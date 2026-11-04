# -*- coding: utf-8 -*-
"""检索层测试。

几条用例是直接从上线的踩坑里长出来的护栏，删掉会让 bug 悄悄回来。
"""

from __future__ import annotations

import pytest

from src.agent.tools import ToolContext, ToolRegistry, ToolSpec, build_default_registry, load_question_bank
from src.rag.bm25 import BM25Index, Hit
from src.rag.corpus import Chunk, Corpus, build_corpus
from src.rag.evaluate import discriminative_power, evaluate, recall_at_k, reciprocal_rank
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.retriever import Retriever, Strategy, available_backends
from src.rag.tokenize import tokenize
from src.rag.vector import HashingEmbedder
from src.schemas.agent import ToolCall


@pytest.fixture(scope="module")
def bank():
    return load_question_bank("config/question_bank.json")


@pytest.fixture(scope="module")
def corpus(bank) -> Corpus:
    return build_corpus(bank)


# --------------------------------------------------------------------------- #
# 一、分词
# --------------------------------------------------------------------------- #
def test_tokenize_english_is_word_level():
    assert tokenize("RAG Plus v1.2") == ["rag", "plus", "v1.2"]


def test_tokenize_chinese_is_bigram():
    """中文按 bigram，不走单字——单字会命中几乎所有条目。"""
    assert tokenize("重排模型") == ["重排", "排模", "模型"]


def test_tokenize_single_han_degrades_to_unigram():
    """单字查询允许 unigram，否则永远召回为空。"""
    assert tokenize("学") == ["学"]


# --------------------------------------------------------------------------- #
# 二、BM25
# --------------------------------------------------------------------------- #
def test_bm25_idf_is_non_negative_on_tiny_corpus():
    """小语料陷阱：教科书 IDF 会让高频词变负数，于是「包含通用词」反而扣分。

    这条测试保护的是：宁可 IDF 接近 0，也不能为负。
    """
    chunks = [
        Chunk(id="a", question="项目管理", dimension="d", difficulty=1, tags=["项目"], tokens=["项目"]),
        Chunk(id="b", question="项目需求", dimension="d", difficulty=1, tags=["项目"], tokens=["项目"]),
        Chunk(id="c", question="项目上线", dimension="d", difficulty=1, tags=["项目"], tokens=["项目"]),
    ]
    idx = BM25Index(Corpus(chunks))
    assert all(v >= 0 for v in idx.idf.values()), "IDF 出现负值，小语料排序会失真"


def test_bm25_universal_term_does_not_outweigh_rare_term():
    """出现在所有文档里的词，IDF 应显著低于只出现一次的词。"""
    chunks = [
        Chunk(id="a", question="通用 罕见甲", dimension="d", difficulty=1, tags=[], tokens=["通用", "罕见甲"]),
        Chunk(id="b", question="通用 其他", dimension="d", difficulty=1, tags=[], tokens=["通用", "其他"]),
        Chunk(id="c", question="通用 别的", dimension="d", difficulty=1, tags=[], tokens=["通用", "别的"]),
    ]
    idx = BM25Index(Corpus(chunks))
    assert idx.idf["罕见甲"] > idx.idf["通用"]


def test_bm25_rejects_empty_corpus():
    with pytest.raises(ValueError):
        BM25Index(Corpus([]))


def test_bm25_ranks_relevant_first(corpus):
    idx = BM25Index(corpus)
    hits = idx.search(corpus, tokenize("向量库 FAISS 选型"), top_k=3)
    assert hits and hits[0].chunk_id == "llm-02"


def test_bm25_no_hit_returns_empty(corpus):
    """「量子佛学」这条 query 是历史事故的化石——它不该命中任何题目。"""
    idx = BM25Index(corpus)
    assert idx.search(corpus, tokenize("zzzz 量子佛学"), top_k=3) == []


# --------------------------------------------------------------------------- #
# 三、向量与融合
# --------------------------------------------------------------------------- #
def test_hashing_embedder_is_deterministic_and_normalized():
    emb = HashingEmbedder(dim=64)
    v1, v2 = emb.embed(["RAG 重排"])[0], emb.embed(["RAG 重排"])[0]
    assert v1 == v2
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-6, "余弦检索要求向量已归一化"


def test_hashing_embedder_is_explicitly_not_semantic():
    """替身身份必须显式可查，防止有人误把它的数字写进报告。"""
    assert HashingEmbedder().semantic is False


def test_rrf_boosts_documents_found_by_both_routes():
    from src.rag.bm25 import Hit

    lists = [[Hit("A", 1.0), Hit("B", 0.9), Hit("C", 0.8)], [Hit("C", 0.7), Hit("A", 0.6)]]
    fused = reciprocal_rank_fusion(lists)
    order = [h.chunk_id for h in fused]
    assert order[0] == "A"          # 两路都命中且名次靠前
    assert order.index("C") < order.index("B")  # 两路都有的 > 只在一路的


# --------------------------------------------------------------------------- #
# 四、Retriever
# --------------------------------------------------------------------------- #
def test_three_strategies_all_retrieve(corpus):
    has_real_embedder = available_backends()["sentence-transformers"]
    for strategy in Strategy:
        if strategy is Strategy.VECTOR and not has_real_embedder:
            # HashingEmbedder 是替身，做不了语义检索——拿替身断言"能检索到"
            # 只会得到一个必然失败的假测试（2026-11-05 CI 实测：195 过 2 挂）。
            # 语义路径由装有 sentence-transformers 的环境负责（本地 / CI semantic job）。
            pytest.skip("无 sentence-transformers：VECTOR 用例依赖真实 embedder，替身测不了语义")
        r = Retriever(corpus, strategy=strategy)
        hits = r.search("向量库 FAISS 选型", top_k=3)
        assert hits, f"{strategy} 应能检索到结果"
        assert hits[0].chunk.id == "llm-02"


def test_nonsense_query_has_no_hit_in_every_strategy(corpus):
    """护栏：任何策略下，无意义 query 都不能假命中。"""
    for strategy in Strategy:
        r = Retriever(corpus, strategy=strategy)
        assert r.search("zzzz 量子佛学", top_k=3) == [], f"{strategy} 出现假命中"


def test_vector_retrieval_requires_similarity_threshold(corpus):
    """稠密检索必须配 `min_score`——这条测试记录的就是设计动机。

    余弦相似度几乎不会等于 0：再不相干的文档也有 0.15 左右的分。
    关掉阈值时「无命中」这个概念在向量检索里根本不存在，
    Agent 就会被一堆毫无关系的召回带偏。（BM25 天然有 >0 语义，不需要这步。）

    只在真实 embedder 下跑：HashingEmbedder 对任何 query 的哈希都不同，
    「关掉阈值必然假命中」这个前提在替身身上根本不成立。
    """
    if not available_backends()["sentence-transformers"]:
        pytest.skip("无 sentence-transformers：min_score 语义只对真实 embedder 成立")
    loose = Retriever(corpus, strategy=Strategy.VECTOR, min_score=0.0)
    assert loose.search("zzzz 量子佛学", top_k=3), "预期：关掉阈值必然假命中"

    strict = Retriever(corpus, strategy=Strategy.VECTOR)
    assert strict.search("zzzz 量子佛学", top_k=3) == [], "开启阈值后必须过滤干净"
    assert strict.search("向量库选型", top_k=3), "阈值不能把正常召回一起掐掉"


def test_retriever_rejects_empty_corpus():
    with pytest.raises(ValueError):
        Retriever(Corpus([]), strategy=Strategy.BM25)


def test_describe_marks_substitute_embedding(bank):
    """`semantic_embedding` 必须在用替身时为 False，这是红线自动化的基础。"""
    r = Retriever.from_bank(bank, strategy=Strategy.HYBRID)
    meta = r.describe()
    assert meta["strategy"] == "hybrid"
    assert isinstance(meta["semantic_embedding"], bool)


def test_available_backends_reports_bm25():
    assert available_backends()["bm25"] is True


def _synthetic_corpus() -> Corpus:
    """两文档、一问深加工维度的最小题库，专用于精确验证加权效果。

    query「甲」下 BM25 原始分：high≈0.2447 > low≈0.2003；
    low 乘 1.3 后 ≈0.2604 > 0.2447——所以 boost B 维应该让它反超。
    （这些数字是手算出来的，改 k1/b/BOOST_FACTOR 时这条测试会先报警。）
    """
    return Corpus(
        [
            Chunk(id="high", question="甲 甲 共有", dimension="A", difficulty=1, tags=["甲"], tokens=["甲", "甲", "共有"]),
            Chunk(id="low", question="乙 共有", dimension="B", difficulty=1, tags=["甲"], tokens=["甲", "共有"]),
        ]
    )


def test_boost_dimensions_flips_ranking_on_synthetic_corpus():
    r = Retriever(_synthetic_corpus(), strategy=Strategy.BM25)
    assert [h.chunk.id for h in r.search("甲", top_k=2)] == ["high", "low"]
    boosted = r.search("甲", top_k=2, boost_dimensions=["B"])
    assert [h.chunk.id for h in boosted] == ["low", "high"]


def test_boost_only_multiplies_matching_dimension():
    r = Retriever(_synthetic_corpus(), strategy=Strategy.BM25)
    base = {h.chunk.id: h.score for h in r.search("甲", top_k=2, boost_dimensions=[])}
    boosted = {h.chunk.id: h.score for h in r.search("甲", top_k=2, boost_dimensions=["B"])}
    assert abs(boosted["low"] - base["low"] * 1.3) < 1e-9
    assert abs(boosted["high"] - base["high"]) < 1e-9


def test_boost_does_not_crash_on_real_bank(corpus):
    """真实题库上，boost 只影响排序，不应改变能否召回、也不应抛异常。"""
    r = Retriever(corpus, strategy=Strategy.BM25)
    hits = r.search("重排", top_k=3, boost_dimensions=["工程基础"])
    assert hits


# --------------------------------------------------------------------------- #
# 五、评测指标（手算校核）
# --------------------------------------------------------------------------- #
def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["b"], 1) == 0.0
    assert recall_at_k(["a", "b", "c"], ["b"], 3) == 1.0
    assert recall_at_k(["a", "b"], ["b", "c"], 5) == 0.5


def test_recall_at_k_is_safe_on_empty_relevant():
    assert recall_at_k(["a"], [], 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert reciprocal_rank(["a"], ["zzz"]) == 0.0


def test_evaluate_reports_all_metrics(bank):
    r = Retriever.from_bank(bank, strategy=Strategy.BM25)
    from src.rag.evaluate import load_cases

    cases = load_cases("config/retrieval_eval.json")
    out = evaluate(r, cases)
    for key in ("recall@1", "recall@3", "recall@5", "mrr@10", "hit@1"):
        assert 0.0 <= out[key] <= 1.0
    assert out["queries"] == len(cases)
    assert set(out["by_kind"]) >= {"literal", "semantic"}


def test_eval_set_has_discriminative_power(bank):
    """评测集如果所有策略都满分，它就废了——这条守护「能不能用它做决策」。"""
    from src.rag.evaluate import load_cases

    cases = load_cases("config/retrieval_eval.json")
    out = evaluate(Retriever.from_bank(bank, strategy=Strategy.BM25), cases)
    diag = discriminative_power([out])
    assert diag["saturated"] is False, (
        "BM25 在评测集上已满分饱和，说明评测集测不出差异，需要补 semantic 类难例"
    )


# --------------------------------------------------------------------------- #
# 六、工具层集成（契约必须不变）
# --------------------------------------------------------------------------- #
async def test_search_tool_uses_retriever(bank):
    from src.agent.state import SessionStore

    ctx = ToolContext(store=SessionStore(), bank=bank, profile={})
    reg = build_default_registry(bank, strategy=Strategy.BM25)
    res = await reg.call(ToolCall(name="search_question_bank", arguments={"query": "向量库 FAISS 选型", "top_k": 2}), ctx)
    assert res.ok, res.error
    assert "llm-02" in res.content or "FAISS" in res.content
    assert len([l for l in res.content.splitlines() if l[:2] in ("1.", "2.")]) <= 2


async def test_search_tool_injects_prebuilt_retriever(bank):
    """registry 预建的检索器应被注入 ctx，避免每轮重建索引。"""
    from src.agent.state import SessionStore

    ctx = ToolContext(store=SessionStore(), bank=bank, profile={})
    reg = build_default_registry(bank, strategy=Strategy.BM25)
    await reg.call(ToolCall(name="search_question_bank", arguments={"query": "FAISS"}), ctx)
    assert ctx.retriever is reg.retriever


async def test_registry_still_usable_without_prebuilt_retriever(bank):
    """老调用方可能不传 bank：此时工具要能即时构建检索器而不是崩掉。"""
    from src.agent.state import SessionStore

    ctx = ToolContext(store=SessionStore(), bank=bank, profile={})
    reg = build_default_registry()
    assert reg.retriever is None
    res = await reg.call(ToolCall(name="search_question_bank", arguments={"query": "FAISS"}), ctx)
    assert res.ok, res.error


async def test_empty_bank_degrades_gracefully():
    """空题库不能让工具抛异常，要返回可执行的建议。"""
    from src.agent.state import SessionStore

    ctx = ToolContext(store=SessionStore(), bank={"items": []}, profile={})
    reg = build_default_registry({"items": []})
    res = await reg.call(ToolCall(name="search_question_bank", arguments={"query": "FAISS"}), ctx)
    assert "题库为空" in res.content


def test_unknown_embedder_name_is_rejected(bank):
    """bm25 不需要 embedder，所以只有 vector/hybrid 才会去解析 embedder 名字。"""
    with pytest.raises(ValueError):
        Retriever.from_bank(bank, strategy=Strategy.VECTOR, embedder_name="nope")


# --------------------------------------------------------------------------- #
# 七、重排（cross-encoder）
# --------------------------------------------------------------------------- #
def test_noop_reranker_preserves_order(corpus):
    from src.rag.rerank import NoopReranker

    hits = [Hit("a", 0.1), Hit("b", 0.9)]
    out = NoopReranker().rerank("q", corpus, hits)
    assert [h.chunk_id for h in out] == ["a", "b"]


def test_reranker_reorders_candidates(corpus):
    """假 reranker：强制把粗排最后一名提到最前，验证重排真的生效。"""
    from src.rag.rerank import Reranker

    class Flipper:
        name = "flipper"
        semantic = False

        def rerank(self, query, corpus, hits):
            return [Hit(h.chunk_id, float(i)) for i, h in enumerate(reversed(hits))]

    r = Retriever(corpus, strategy=Strategy.BM25, reranker=Flipper())
    base = Retriever(corpus, strategy=Strategy.BM25)
    q = "怎么优化"  # BM25 候选池 >=9 条的查询，翻转后 top-3 必然变化
    pool = [h.chunk_id for h in base._rank(q, 9, ())]  # 粗排候选池（top_k*3）
    base_ids = [h.chunk.id for h in base.search(q, top_k=3)]
    flip_ids = [h.chunk.id for h in r.search(q, top_k=3)]
    assert base_ids != flip_ids  # 顺序必须被重排改变
    assert flip_ids == pool[::-1][:3]  # 重排后取 top-3 = 候选池反转的前 3 名
    assert set(flip_ids) <= set(pool)  # 重排不引入候选池之外的新条目


def test_unknown_reranker_name_is_rejected(bank):
    from src.rag.rerank import build_reranker

    with pytest.raises(ValueError):
        build_reranker("nope")


def test_describe_reports_reranker(corpus):
    r = Retriever(corpus, strategy=Strategy.BM25)
    meta = r.describe()
    assert meta["reranker"] == "noop"
    assert meta["semantic_reranker"] is False
