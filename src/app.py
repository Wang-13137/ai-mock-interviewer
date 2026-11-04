# -*- coding: utf-8 -*-
"""FastAPI 入口。

本地起服务：
    python -m src.cli serve
    # 等价于 PYTHONPATH=. uvicorn src.app:app --reload

接口：
    GET  /health            健康检查（含当前 provider / model / ready）
    POST /v1/chat           非流式问答
    POST /v1/chat/stream    SSE 流式问答
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .agent.loop import AgentLoop
from .agent.memory import MemoryManager, MemoryPolicy
from .agent.state import SessionStore
from .agent.tools import build_default_registry, load_question_bank
from .clients.llm import BaseLLMClient, EchoClient, LLMError, OpenAICompatClient, Message
from .core.config import Settings, load_settings, resolve_llm
from .core.logger import setup_logging
from .metrics import JsonlSink, LatencyMeter
from .prompts.interviewer import build_messages
from .schemas.agent import ToolSpec
from .schemas.chat import (
    AgentChatResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    Usage,
)

log = logging.getLogger("aimi.app")


def build_client(settings: Settings, role: str = "chat") -> BaseLLMClient:
    """按配置构建 LLM 客户端；条件不满足时降级到离线替身（保证链路永远可跑）。

    role="chat" 用 settings.llm（对话生成），role="judge" 用 settings.judge（评分裁判）。
    """
    raw = settings.llm if role == "chat" else settings.judge
    try:
        llm = resolve_llm(raw)
    except Exception as exc:  # 配置写错不该让服务起不来
        log.error("配置解析失败（role=%s），降级到离线替身：%s", role, exc)
        return EchoClient(model="echo-1")

    if not llm.api_key or not llm.base_url or llm.provider == "mock":
        log.warning(
            "未检测到可用的 LLM 供应商（role=%s provider=%s），当前跑在离线替身 EchoClient 上",
            role,
            raw.provider,
        )
        return EchoClient(model=llm.model or "echo-1")

    return OpenAICompatClient(
        model=llm.model,
        base_url=llm.base_url,
        api_key=llm.api_key,
        provider=llm.provider,
        timeout=llm.timeout,
        max_retries=llm.max_retries,
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
    )


def _build_memory(client: BaseLLMClient, cfg: Settings) -> Optional[MemoryManager]:
    """记忆管理器：关闭或配置缺失时返回 None，AgentLoop 不做压缩。"""
    if not cfg.memory.enabled:
        return None
    return MemoryManager(
        client=client,
        policy=MemoryPolicy(
            enabled=True,
            max_tokens=cfg.memory.max_tokens,
            keep_recent=cfg.memory.keep_recent,
            summary_max_chars=cfg.memory.summary_max_chars,
        ),
    )


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """应用工厂：便于测试时注入离线配置，也便于未来按环境装配不同中间件。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or load_settings()
        setup_logging(cfg.env)
        app.state.settings = cfg
        app.state.client = build_client(cfg)
        app.state.sink = JsonlSink(cfg.metrics.path, enabled=cfg.metrics.enabled)
        app.state.store = SessionStore()
        if cfg.agent.enabled:
            app.state.bank = load_question_bank(cfg.agent.question_bank)
            try:
                app.state.registry = build_default_registry(
                    app.state.bank,
                    strategy=cfg.rag.strategy,
                    embedder_name=cfg.rag.embedder,
                    model_name=cfg.rag.model_name,
                    min_score=cfg.rag.min_score,
                    reranker_name=cfg.rag.rerank,
                    reranker_model=cfg.rag.reranker_model,
                )
                app.state.retriever = app.state.registry.retriever
            except Exception as exc:
                log.error("检索器构建失败，Agent 检索将不可用：%s", exc)
                app.state.registry = build_default_registry()
                app.state.retriever = None
        else:
            app.state.bank = {}
            app.state.registry = None
            app.state.retriever = None
        log.info(
            "boot env=%s provider=%s model=%s ready=%s",
            cfg.env,
            app.state.client.provider_name,
            app.state.client.model,
            app.state.client.ready,
        )
        try:
            yield
        finally:
            await app.state.client.aclose()

    app = FastAPI(
        title="AI Mock Interviewer",
        version="0.1.0",
        description="把毕设《VR + 语音的 AI 模拟面试官》重构成 Python + FastAPI 的 LLM 应用工程",
        lifespan=lifespan,
    )

    # ---------------- 路由 ---------------- #
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        client: BaseLLMClient = request.app.state.client
        cfg: Settings = request.app.state.settings
        retriever = getattr(request.app.state, "retriever", None)
        meta = retriever.describe() if retriever is not None else {}
        return HealthResponse(
            env=cfg.env,
            provider=client.provider_name,
            model=client.model,
            ready=client.ready,
            version=cfg.version,
            agent_enabled=cfg.agent.enabled and request.app.state.registry is not None,
            tools=request.app.state.registry.names() if request.app.state.registry else [],
            retrieval_strategy=meta.get("strategy", "") or cfg.rag.strategy,
            semantic_retrieval=bool(meta.get("semantic_embedding", False)),
        )

    @app.get("/v1/tools", response_model=List[ToolSpec], tags=["agent"])
    def list_tools(request: Request) -> List[ToolSpec]:
        """暴露工具清单：前端要显示「本轮用了哪些工具」，测试要校验 Schema 合法性。"""
        registry = request.app.state.registry
        if registry is None:
            return []
        return registry.specs()

    @app.post("/v1/chat/agent", response_model=AgentChatResponse, tags=["agent"])
    async def chat_agent(req: ChatRequest, request: Request) -> AgentChatResponse:
        """带工具的对话：多打几次模型，中途可能调用检索 / 状态读写工具。"""
        client: BaseLLMClient = request.app.state.client
        sink: JsonlSink = request.app.state.sink
        cfg: Settings = request.app.state.settings
        registry = request.app.state.registry
        trace_id = uuid.uuid4().hex[:12]
        meter = LatencyMeter()

        loop = AgentLoop(
            client=client,
            registry=registry,
            store=request.app.state.store,
            bank=request.app.state.bank,
            max_steps=cfg.agent.max_steps,
            memory=_build_memory(client, cfg),
            protocol=cfg.agent.protocol,
        )

        run = await loop.run(
            _to_messages(req),
            req.profile.model_dump(),
            session_id=req.session_id,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        report = meter.report()
        sink.write(
            {
                **report,
                "trace_id": trace_id,
                "endpoint": "/v1/chat/agent",
                "provider": client.provider_name,
                "model": client.model,
                "turn": req.turn_count,
                "model_calls": run.model_calls,
                "tool_calls": run.tool_calls,
                "tool_errors": run.tool_errors,
                "parse_errors": run.parse_errors,
                "stopped": run.stopped,
                "ok": run.stopped != "error",
            }
        )
        return AgentChatResponse(
            session_id=req.session_id,
            reply=run.reply,
            model=client.model,
            provider=client.provider_name,
            usage=Usage(
                prompt_tokens=run.usage.prompt_tokens,
                completion_tokens=run.usage.completion_tokens,
                total_tokens=run.usage.total_tokens,
            ),
            trace_id=trace_id,
            latency_ms=report["total_ms"],
            steps=run.steps,
            tool_calls=run.tool_calls,
            tool_errors=run.tool_errors,
            parse_errors=run.parse_errors,
            stopped=run.stopped,
        )

    @app.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(req: ChatRequest, request: Request) -> ChatResponse:
        client: BaseLLMClient = request.app.state.client
        sink: JsonlSink = request.app.state.sink
        trace_id = uuid.uuid4().hex[:12]
        meter = LatencyMeter()
        messages = build_messages(_to_messages(req), req.profile.model_dump())

        try:
            result = await client.achat(
                messages, temperature=req.temperature, max_tokens=req.max_tokens
            )
        except LLMError as exc:
            sink.write({**meter.report(), "trace_id": trace_id, "endpoint": "/v1/chat", "ok": False, "error": str(exc)})
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        usage = Usage(
            prompt_tokens=result.usage.prompt_tokens if result.usage else None,
            completion_tokens=result.usage.completion_tokens if result.usage else None,
            total_tokens=result.usage.total_tokens if result.usage else None,
        )
        report = meter.report(result.usage)
        sink.write(
            {
                **report,
                "trace_id": trace_id,
                "endpoint": "/v1/chat",
                "provider": client.provider_name,
                "model": client.model,
                "turn": req.turn_count,
                "ok": True,
            }
        )
        return ChatResponse(
            session_id=req.session_id,
            reply=result.content,
            model=client.model,
            provider=client.provider_name,
            finish_reason=result.finish_reason,
            usage=usage,
            trace_id=trace_id,
            latency_ms=report["total_ms"],
        )

    @app.post("/v1/chat/stream", tags=["chat"])
    async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
        client: BaseLLMClient = request.app.state.client
        sink: JsonlSink = request.app.state.sink
        trace_id = uuid.uuid4().hex[:12]
        meter = LatencyMeter()
        messages = build_messages(_to_messages(req), req.profile.model_dump())

        async def event_gen() -> AsyncIterator[bytes]:
            collected: List[str] = []
            try:
                async for delta in client.astream(
                    messages, temperature=req.temperature, max_tokens=req.max_tokens
                ):
                    collected.append(delta)
                    meter.mark_first_token()
                    yield _sse({"type": "delta", "delta": delta, "trace_id": trace_id})
                report = meter.report()
                sink.write(
                    {
                        **report,
                        "trace_id": trace_id,
                        "endpoint": "/v1/chat/stream",
                        "provider": client.provider_name,
                        "model": client.model,
                        "turn": req.turn_count,
                        "chars": len("".join(collected)),
                        "ok": True,
                    }
                )
                yield _sse({"type": "done", "trace_id": trace_id, **report})
            except LLMError as exc:
                sink.write(
                    {
                        **meter.report(),
                        "trace_id": trace_id,
                        "endpoint": "/v1/chat/stream",
                        "ok": False,
                        "error": str(exc),
                    }
                )
                yield _sse({"type": "error", "trace_id": trace_id, "error": str(exc)})
            finally:
                yield b"data: [DONE]\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Trace-Id": trace_id},
        )

    return app


app = create_app()


# ---------------- 工具 ---------------- #
def _to_messages(req: ChatRequest) -> List[Message]:
    return [{"role": m.role, "content": m.content} for m in req.messages]


def _sse(payload: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
