# -*- coding: utf-8 -*-
"""统一的 LLM 客户端抽象层。

设计要点（也是简历里能讲的点）：
1. 只实现一套 **OpenAI 兼容协议**（POST /chat/completions），智谱 / 硅基流动 / DeepSeek /
   OpenAI 四家都能跑，切换 provider 不用改业务代码；
2. SSE 流式**手写解析**，不依赖厂商 SDK —— 出问题能定位到协议层；
3. `EchoClient` 是离线替身：没 Key、跑 CI、跑端到端冒烟时都能用。
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger("aimi.llm")

Message = Dict[str, str]


class LLMUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    @property
    def total_tokens(self) -> Optional[int]:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


class LLMResult(BaseModel):
    content: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[LLMUsage] = None


class LLMError(RuntimeError):
    """LLM 调用失败。message 里不含密钥。"""

    def __init__(self, message: str, *, status_code: Optional[int] = None, detail: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class BaseLLMClient(ABC):
    """所有 LLM 客户端的共同接口：非流式 achat + 流式 astream。"""

    provider_name: str = "base"
    model: str = "unknown"

    @property
    def ready(self) -> bool:
        """是否连接了真实模型（False 表示处于离线替身模式）。"""
        return True

    @abstractmethod
    async def achat(
        self,
        messages: List[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResult:
        ...

    @abstractmethod
    async def astream(
        self,
        messages: List[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """逐段产出增量文本。实现类必须是 async generator。"""
        yield ""  # pragma: no cover

    async def aclose(self) -> None:  # 预留：未来接连接池 / gRPC
        return None


# --------------------------------------------------------------------------- #
# OpenAI 兼容实现
# --------------------------------------------------------------------------- #
class OpenAICompatClient(BaseLLMClient):
    """面向 /chat/completions 的通用客户端，同时支持非流式与 SSE 流式。"""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        provider: str = "openai-compatible",
        timeout: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url 不能为空")
        if not model:
            raise ValueError("model 不能为空")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider_name = provider
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=httpx.AsyncHTTPTransport(retries=max_retries),
        )

    # ---------------- 内部工具 ---------------- #
    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(
        self,
        messages: List[Message],
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.default_max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }

    @staticmethod
    def _raise_for_http(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            body = resp.text[:300]
        except Exception:  # pragma: no cover
            body = ""
        hint = {
            401: "鉴权失败：检查 API Key 环境变量",
            403: "无权限：账号未开通该模型",
            429: "触发限流：稍后重试或降低并发",
        }.get(resp.status_code, "上游返回错误")
        raise LLMError(f"{hint} (HTTP {resp.status_code})", status_code=resp.status_code, detail=body)

    # ---------------- 非流式 ---------------- #
    async def achat(
        self,
        messages: List[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResult:
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens, stream=False)
        try:
            resp = await self._client.post(self._endpoint, headers=self._headers(), json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError(f"LLM 调用超时（{exc}）") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM 网络异常（{exc}）") from exc

        self._raise_for_http(resp)
        data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            raise LLMError("上游返回为空：choices 缺失", detail=json.dumps(data)[:200])
        message = choices[0].get("message", {}) or {}
        raw_usage = data.get("usage") or {}
        usage = (
            LLMUsage(
                prompt_tokens=raw_usage.get("prompt_tokens"),
                completion_tokens=raw_usage.get("completion_tokens"),
            )
            if raw_usage
            else None
        )
        return LLMResult(
            content=message.get("content") or "",
            finish_reason=choices[0].get("finish_reason"),
            usage=usage,
        )

    # ---------------- 流式 ---------------- #
    async def astream(
        self,
        messages: List[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens, stream=True)
        try:
            async with self._client.stream(
                "POST", self._endpoint, headers=self._headers(), json=payload
            ) as resp:
                self._raise_for_http(resp)
                async for event in self._iter_sse(resp):
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    if delta.get("content"):
                        yield delta["content"]
        except httpx.TimeoutException as exc:
            raise LLMError(f"LLM 流式调用超时（{exc}）") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM 网络异常（{exc}）") from exc

    @staticmethod
    async def _iter_sse(resp: httpx.Response) -> AsyncIterator[Dict[str, Any]]:
        """手写 SSE 帧解析：`data: {...}` / `data: [DONE]`，忽略注释行与心跳。"""
        async for line in resp.aiter_lines():
            if not line:
                continue
            line = line.rstrip("\r")
            if line.startswith(":"):  # SSE 注释行，常用来做心跳
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data or data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                log.debug("丢弃无法解析的 SSE 帧: %s", data[:120])

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


# --------------------------------------------------------------------------- #
# 离线替身
# --------------------------------------------------------------------------- #
class EchoClient(BaseLLMClient):
    """无网络、无 Key 时的替身实现：把用户输入回显成一段「面试官口吻」的回复。

    用途：① 端到端链路冒烟 ② CI 里的流式/非流式测试 ③ 演示接口格式。
    """

    provider_name = "mock"
    ready = False

    def __init__(self, model: str = "echo-1", chunk_delay: float = 0.0) -> None:
        self.model = model
        self.chunk_delay = chunk_delay

    @staticmethod
    def _reply(messages: List[Message]) -> str:
        last = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last = msg.get("content", "")
                break
        return (
            f"[离线替身] 收到你的输入（{len(last)} 字）：{last[:40]}。"
            "请告诉我更多细节，例如你在这个项目里具体负责哪一块？"
            "（当前未配置真实模型，把 provider 改成 zhipu/deepseek 并填入 Key 即可）"
        )

    async def achat(
        self,
        messages: List[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResult:
        content = self._reply(messages)
        approx = max(1, len(content) // 2)  # 粗略估算，替身没有真实 token 计数
        return LLMResult(
            content=content,
            finish_reason="stop",
            usage=LLMUsage(prompt_tokens=approx * 2, completion_tokens=approx),
        )

    async def astream(
        self,
        messages: List[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        import asyncio

        content = self._reply(messages)
        step = 4  # 模拟逐 token 吐字
        for start in range(0, len(content), step):
            if self.chunk_delay:
                await asyncio.sleep(self.chunk_delay)
            yield content[start:start + step]

    # pydantic-free 的 dataclass 兼容 field 声明（保持与基类一致的类型）
    def __init_subclass__(cls, **kwargs):  # pragma: no cover
        super().__init_subclass__(**kwargs)
