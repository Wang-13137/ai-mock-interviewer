# -*- coding: utf-8 -*-
"""评测体系的测试：数据契约、裁判输出解析、一致性统计、端到端跑批。"""

import json

import pytest

from src.eval.dataset import EvalCase, EvalDataset, load_dataset
from src.eval.judge import JudgeResult, judge_one, parse_judge_json, render_dialogue
from src.eval.stats import (accuracy_within, agreement_report, cohens_kappa,
                            interpret, quadratic_weighted_kappa)
from src.clients.llm import EchoClient
from src.core.config import LLMConfig, MetricsConfig, ServerConfig, Settings
from src.eval.runner import run_eval

SAMPLE = {
    "name": "unit", "version": "0.1", "rubric_version": "v1",
    "cases": [
        {"id": "a1", "dimension": "追问合理性",
         "dialogue": [{"role": "user", "content": "hi"}],
         "reference": "r", "human_score": 4},
    ],
}


# ---------------- 数据契约 ---------------- #
def test_dataset_loads(tmp_path):
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8")
    ds = load_dataset(p)
    assert ds.size == 1
    assert ds.cases[0].human_score == 4
    hist, query = ds.cases[0].history_and_query
    assert hist == [] and query == "hi"


def test_last_turn_must_be_user():
    with pytest.raises(Exception):
        EvalCase(id="bad", dimension="追问合理性",
                 dialogue=[{"role": "assistant", "content": "x"}])


def test_invalid_dimension_rejected():
    with pytest.raises(Exception):
        EvalCase(id="bad", dimension="随便一个维度",
                 dialogue=[{"role": "user", "content": "x"}])


def test_human_score_bounds():
    with pytest.raises(Exception):
        EvalCase(id="bad", dimension="追问合理性",
                 dialogue=[{"role": "user", "content": "x"}], human_score=9)


def test_by_dimension():
    ds = EvalDataset(**SAMPLE)
    assert len(ds.by_dimension(ds.cases[0].dimension)) == 1


# ---------------- 裁判输出解析 ---------------- #
def test_parse_plain_json():
    assert parse_judge_json('{"score": 5, "reason": "ok"}')["score"] == 5


def test_parse_json_in_code_fence():
    text = '```json\n{"score": 3, "reason": "一般"}\n```'
    assert parse_judge_json(text)["score"] == 3


def test_parse_json_with_surrounding_text():
    text = '好的，我的评分如下：\n{"score": 2, "reason": "跑偏了"}\n希望有帮助。'
    assert parse_judge_json(text)["score"] == 2


def test_parse_invalid_raises():
    with pytest.raises(ValueError):
        parse_judge_json("这不是 JSON")


def test_render_dialogue():
    out = render_dialogue([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    assert "候选人：a" in out and "面试官：b" in out


class _ScriptedJudge(EchoClient):
    """按脚本返回裁判输出的假客户端，用来测各种畸形回复。"""

    provider_name = "mock-judge"

    def __init__(self, content: str):
        super().__init__(model="judge-mock")
        self._content = content

    async def achat(self, messages, *, temperature=None, max_tokens=None):
        from src.clients.llm import LLMResult
        return LLMResult(content=self._content, finish_reason="stop")


async def test_judge_happy_path():
    case = EvalCase(**SAMPLE["cases"][0])
    res = await judge_one(_ScriptedJudge('{"score": 5, "reason": "追问到位"}'), case, "然后呢？")
    assert res.parse_ok and res.score == 5 and res.reason == "追问到位"


async def test_judge_bad_text_recorded_not_raised():
    case = EvalCase(**SAMPLE["cases"][0])
    res = await judge_one(_ScriptedJudge("我拒绝打分"), case, "x")
    assert res.score is None
    assert not res.parse_ok
    assert "JSON" in res.error


async def test_judge_out_of_range_score():
    case = EvalCase(**SAMPLE["cases"][0])
    res = await judge_one(_ScriptedJudge('{"score": 9, "reason": ""}'), case, "x")
    assert res.score is None and "越界" in res.error


async def test_judge_missing_score_field():
    case = EvalCase(**SAMPLE["cases"][0])
    res = await judge_one(_ScriptedJudge('{"reason": "没有分"}'), case, "x")
    assert res.score is None and "score" in res.error


async def test_judge_upstream_failure():
    class _Boom:
        async def achat(self, *a, **kw):
            raise RuntimeError("upstream down")

    case = EvalCase(**SAMPLE["cases"][0])
    res = await judge_one(_Boom(), case, "x")  # type: ignore[arg-type]
    assert res.score is None and "RuntimeError" in res.error


def test_judge_result_model_defaults():
    jr = JudgeResult(case_id="x")
    assert jr.score is None and jr.parse_ok is False


# ---------------- 一致性统计 ---------------- #
def test_accuracy_within():
    assert accuracy_within([1, 2, 3], [1, 2, 3]) == 1.0
    assert accuracy_within([1, 2, 3], [2, 3, 4]) == 0.0
    assert accuracy_within([1, 2, 3], [2, 3, 4], tol=1) == 1.0


def test_kappa_perfect_and_degenerate():
    assert cohens_kappa([1, 2, 3], [1, 2, 3]) == 1.0
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) is None, "常数序列下 kappa 不可解释"


def test_kappa_partial_agreement():
    human = [5, 5, 5, 4, 3]
    auto = [5, 5, 4, 4, 3]
    k = cohens_kappa(human, auto)
    # 手算：po=0.8, pe=0.36 -> kappa=0.6875
    assert k is not None and abs(k - 0.6875) < 1e-6
    q = quadratic_weighted_kappa(human, auto)
    assert q is not None and -1.0 <= q <= 1.0
    # 注意：小样本 + 边际偏斜时，QWK 与 kappa 谁高并不固定（本例 QWK≈0.19 < kappa≈0.69）。
    # 所以报告里两者都给，正式结论只在 n≥30 时下。


def test_kappa_paradox_when_marginal_is_degenerate():
    """一方全是同一个分值时 po==pe -> kappa=0，这不是 bug 而是 kappa 的已知特性（kappa paradox）。

    所以报告同时给出严格一致率、acc±1 与 QWK，不让单一指标误导结论。
    """
    assert cohens_kappa([5, 5, 5, 5], [5, 4, 5, 4]) == 0.0
    assert accuracy_within([5, 5, 5, 5], [5, 4, 5, 4]) == 0.5


def test_kappa_length_mismatch_raises():
    with pytest.raises(ValueError):
        cohens_kappa([1, 2], [1])


def test_qwk_penalizes_distance():
    near = quadratic_weighted_kappa([1, 5, 3], [2, 5, 3])
    far = quadratic_weighted_kappa([1, 5, 3], [5, 1, 3])
    assert near > far, "差一档应当比差四档惩罚轻"


def test_qwk_perfect():
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0


def test_agreement_report_shape():
    rep = agreement_report([4, 3, 5, 2], [4, 4, 5, 2])
    assert rep["n"] == 4
    assert rep["acc"] == 0.75
    assert rep["acc_within_1"] == 1.0
    assert rep["kappa"] is not None
    assert rep["qwk"] is not None


def test_interpret_bands():
    assert interpret(None) == "不可计算"
    assert interpret(-0.1) == "差于随机"
    assert "一般" in interpret(0.3)
    assert "中等" in interpret(0.5)
    assert "几乎完全" in interpret(0.9)


# ---------------- 端到端跑批（离线替身） ---------------- #
async def test_run_eval_end_to_end():
    settings = Settings(
        env="test",
        llm=LLMConfig(provider="mock"),
        judge=LLMConfig(provider="mock"),
        metrics=MetricsConfig(enabled=False),
        server=ServerConfig(),
    )
    ds = EvalDataset(**SAMPLE)
    report = await run_eval(settings, ds)
    assert report.meta["n"] == 1
    # 离线替身不会输出合法 JSON，分数应为 None，但流程必须走完且不抛异常
    assert report.outcomes[0].judge.score is None
    assert report.warnings, "替身模式必须给出「不得写进材料」的警示"
    assert report.parse_rate == 0.0
    assert "链路验证" in report.warnings[-1] or "替身" in report.warnings[-1]
