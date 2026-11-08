"""统一的证据数据结构。

每条证据都必须能独立解释，包含：来源模块、强度、原始数值、人类可读说明。
融合层只信任 Evidence；不要把任何裸 float 直接喂给融合。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Strength(str, Enum):
    """证据强度等级。

    - HARD: 密码学/明确硬证据（C2PA 签名、AI 软件元数据），可一票定性
    - STRONG: 高权重软证据（明显频域峰值 + 元数据缺失）
    - MEDIUM: 中等权重（FFT/ELA 单一异常）
    - WEAK:   仅供参考（噪声分布偏离）
    - INFO:   纯信息，不参与打分
    """

    HARD = "hard"
    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"
    INFO = "info"


class Direction(str, Enum):
    """证据指向。"""

    AI = "ai"            # 指向 AI 生成
    REAL = "real"        # 指向真实拍摄
    NEUTRAL = "neutral"  # 中立 / 未知


@dataclass
class Evidence:
    module: str            # 来源模块名，如 "exif"
    name: str              # 证据短名，如 "ai_software_tag"
    direction: Direction
    strength: Strength
    score: float           # 0~1，越大越强
    description: str       # 人类可读说明
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["strength"] = self.strength.value
        return d
