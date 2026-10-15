# -*- coding: utf-8 -*-
"""统一分词。

中文不走分词器（不引入 jieba 这类重依赖），而是切成 **bigram**：

- 纯单字切会让「的」「学」这类高频字命中几乎所有条目，检索退化成噪声；
- bigram 在没有分词器的前提下是性价比最高的折中；
- 单字查询（例如用户只输入「R」以外的单个汉字）才退化为 unigram，否则永远召回为空。

这条规则是从一次真实事故里来的：早期实现用单字匹配，
query「量子佛学」能匹配到「向量库」那道题——**假命中比漏召回更致命**，
因为 Agent 会被一堆无关题目带偏，而且从外部看「工具调用成功」了。
"""

from __future__ import annotations

import re
from typing import List

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#._-]*")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """英文/数字按词切，中文按 bigram 切，统一小写。"""
    text = text or ""
    out: List[str] = [t.lower() for t in WORD_RE.findall(text)]
    han = "".join(HAN_RE.findall(text))
    if len(han) == 1:
        out.append(han)
    else:
        out += [han[i : i + 2] for i in range(len(han) - 1)]
    return out
