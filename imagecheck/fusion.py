"""证据融合层。

核心逻辑（"硬证据优先 + 软证据加权"）：

1. **硬证据一票定性**：
   - 任一 Evidence 为 HARD + AI → verdict=ai_generated, prob>=0.95
   - 任一 Evidence 为 HARD + REAL → verdict=real_capture, prob<=0.05
   - 双方都有 HARD：以最新 / 最高 score 优先（一般为冲突情况，confidence 降低）

2. **加权融合**：
   - 权重表：HARD 1.0 / STRONG 0.6 / MEDIUM 0.35 / WEAK 0.15 / INFO 0.0
   - AI 方向 score 累计为 +，REAL 方向累计为 −
   - 最终 logit = sigmoid(累计 / 归一化系数) → ai_probability

3. **置信度**：
   - HIGH: 有 HARD 证据 / 各 module 方向一致 / 累计权重 > 1.5
   - MEDIUM: 有方向但有少量分歧
   - LOW: 全部 INFO 或方向严重冲突

输出结构与开发计划文档一致：
    ai_probability, verdict, confidence_level, evidence_list,
    raw_module_scores, provenance_summary, limitations_note
"""
from __future__ import annotations

import math
from typing import Any

from .evidence import Direction, Evidence, Strength


_WEIGHTS = {
    Strength.HARD: 1.0,
    Strength.STRONG: 0.6,
    Strength.MEDIUM: 0.35,
    Strength.WEAK: 0.15,
    Strength.INFO: 0.0,
}

_LIMITATIONS = (
    "本工具是辅助识别系统，不替代人工复核与司法鉴定。"
    "对截图、二次压缩、滤镜处理或局部修图后的图片，结论可信度会显著下降。"
    "AI 修图与真实照片混合的灰区案例建议人工复核。"
)


def _summarize_provenance(evidences: list[Evidence]) -> str:
    parts = []
    for ev in evidences:
        if ev.module in ("c2pa", "exif") and ev.strength in (Strength.HARD, Strength.STRONG):
            parts.append(ev.description)
    return "；".join(parts) if parts else "无强来源信号"


def fuse(evidences: list[Evidence]) -> dict[str, Any]:
    if not evidences:
        return {
            "ai_probability": 0.5,
            "verdict": "unknown",
            "confidence_level": "low",
            "evidence_list": [],
            "raw_module_scores": {},
            "provenance_summary": "无任何证据",
            "limitations_note": _LIMITATIONS,
        }

    # 1. 硬证据优先
    hard_ai = [e for e in evidences if e.strength == Strength.HARD and e.direction == Direction.AI]
    hard_real = [e for e in evidences if e.strength == Strength.HARD and e.direction == Direction.REAL]

    raw_scores = {ev.module: round(ev.score if ev.direction == Direction.AI
                                    else (-ev.score if ev.direction == Direction.REAL else 0.0), 3)
                  for ev in evidences}

    if hard_ai and not hard_real:
        prob = max(e.score for e in hard_ai)
        return _build_result(prob, "ai_generated", "high", evidences, raw_scores)
    if hard_real and not hard_ai:
        prob = 1.0 - max(e.score for e in hard_real)
        return _build_result(prob, "real_capture", "high", evidences, raw_scores)
    if hard_ai and hard_real:
        # 罕见冲突
        ai_max = max(e.score for e in hard_ai)
        real_max = max(e.score for e in hard_real)
        prob = (ai_max + (1.0 - real_max)) / 2.0
        return _build_result(prob, "conflicting_evidence", "low", evidences, raw_scores)

    # 2. 软证据加权
    pos = 0.0  # AI
    neg = 0.0  # REAL
    weight_total = 0.0
    directions_seen: set[Direction] = set()
    for ev in evidences:
        w = _WEIGHTS[ev.strength]
        if w == 0 or ev.direction == Direction.NEUTRAL:
            continue
        weight_total += w
        directions_seen.add(ev.direction)
        if ev.direction == Direction.AI:
            pos += w * ev.score
        else:
            neg += w * ev.score

    if weight_total == 0:
        return {
            "ai_probability": 0.5,
            "verdict": "insufficient_evidence",
            "confidence_level": "low",
            "evidence_list": [e.to_dict() for e in evidences],
            "raw_module_scores": raw_scores,
            "provenance_summary": _summarize_provenance(evidences),
            "limitations_note": _LIMITATIONS,
        }

    # logit 归一化：用 (pos - neg) / max(weight_total, 1.0) 限制斜率
    diff = (pos - neg) / max(weight_total, 0.5)
    prob = 1.0 / (1.0 + math.exp(-3.0 * diff))  # 缩放，使中等加权也能拉开

    # 3. 置信度
    if weight_total >= 1.5 and len(directions_seen) == 1:
        confidence = "high"
    elif weight_total >= 0.8:
        confidence = "medium" if len(directions_seen) <= 1 else "low"
    else:
        confidence = "low"

    if prob >= 0.7:
        verdict = "likely_ai_generated"
    elif prob <= 0.3:
        verdict = "likely_real_capture"
    else:
        verdict = "uncertain"

    return _build_result(prob, verdict, confidence, evidences, raw_scores)


def _build_result(prob: float, verdict: str, confidence: str,
                  evidences: list[Evidence], raw_scores: dict) -> dict[str, Any]:
    return {
        "ai_probability": round(max(0.0, min(1.0, prob)), 4),
        "verdict": verdict,
        "confidence_level": confidence,
        "evidence_list": [e.to_dict() for e in evidences],
        "raw_module_scores": raw_scores,
        "provenance_summary": _summarize_provenance(evidences),
        "limitations_note": _LIMITATIONS,
    }
