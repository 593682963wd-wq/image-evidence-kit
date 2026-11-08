"""启发式 / 统计特征分类器（占位实现）。

预留两类接口：
1. `predict_heuristic(image_path)` — 当前 v1.0 默认实现：基于颜色直方图、
   边缘密度、局部熵等手工特征做规则化打分。**不依赖训练好的模型**，
   保证项目零依赖即可跑。
2. `predict_with_model(image_path, model_path)` — 占位钩子，后续接入
   CLIP/CNN/ViT 时实现。

无法替代真正的 CNN 集成，但提供一致的输出结构与融合层接口。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .evidence import Direction, Evidence, Strength


def _features(image_path: Path, target: int = 384) -> dict[str, float]:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        im = im.resize((target, target), Image.LANCZOS)
        arr = np.asarray(im)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # 边缘密度
    edges = cv2.Canny(gray, 80, 200)
    edge_density = float(edges.mean()) / 255.0
    # 颜色饱和度均值（HSV）
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat_mean = float(hsv[..., 1].mean()) / 255.0
    val_std = float(hsv[..., 2].std()) / 255.0
    # 局部纹理熵（用 Laplacian 方差近似）
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    # 颜色直方图熵（归一化）
    hist, _ = np.histogram(gray, bins=64, range=(0, 256), density=False)
    hist = hist[hist > 0].astype(float)
    hist /= hist.sum()
    color_entropy = float(-(hist * np.log2(hist)).sum())

    return {
        "edge_density": edge_density,
        "sat_mean": sat_mean,
        "val_std": val_std,
        "lap_var": lap_var,
        "color_entropy": color_entropy,
    }


def predict_heuristic(image_path: str | Path) -> list[Evidence]:
    """返回最多 1 条证据（统计画像综合得分）。"""
    path = Path(image_path)
    if not path.exists():
        return []
    try:
        f = _features(path)
    except Exception as e:
        return [Evidence(module="classifier", name="error", direction=Direction.NEUTRAL,
                         strength=Strength.INFO, score=0.0,
                         description=f"分类器特征提取失败：{e}")]

    # 经验启发式：AI 图典型特征
    #   - 颜色饱和度偏高（sat_mean > 0.45）
    #   - 全局纹理过于平滑（lap_var < 80）
    #   - 颜色直方图熵偏低（< 5.0）
    score_ai = 0.0
    reasons: list[str] = []
    if f["sat_mean"] > 0.45:
        score_ai += 0.2; reasons.append(f"饱和度偏高 {f['sat_mean']:.2f}")
    if f["lap_var"] < 80:
        score_ai += 0.2; reasons.append(f"纹理过平滑 lap_var={f['lap_var']:.0f}")
    if f["color_entropy"] < 5.0:
        score_ai += 0.15; reasons.append(f"颜色熵偏低 {f['color_entropy']:.2f}")
    if f["edge_density"] < 0.04:
        score_ai += 0.1; reasons.append(f"边缘极少 {f['edge_density']:.3f}")
    if f["edge_density"] > 0.18:
        score_ai -= 0.1  # 边缘极多更像真实复杂场景

    score_ai = max(0.0, min(1.0, score_ai))

    if score_ai >= 0.45:
        return [Evidence(
            module="classifier", name="heuristic_ai_profile",
            direction=Direction.AI, strength=Strength.MEDIUM, score=score_ai,
            description=f"启发式特征画像偏 AI（{', '.join(reasons) or '综合判定'}）",
            raw=f,
        )]
    if score_ai <= 0.15:
        return [Evidence(
            module="classifier", name="heuristic_real_profile",
            direction=Direction.REAL, strength=Strength.WEAK, score=0.4,
            description="启发式特征画像偏真实拍摄",
            raw=f,
        )]
    return [Evidence(
        module="classifier", name="heuristic_neutral",
        direction=Direction.NEUTRAL, strength=Strength.INFO, score=0.0,
        description=f"启发式特征中性（AI 倾向分 {score_ai:.2f}）",
        raw=f,
    )]


def predict_with_model(image_path, model_path):  # pragma: no cover
    """占位：后续接入 CLIP/ViT/CNN 时实现。"""
    raise NotImplementedError("v1.0 未启用学习模型，预留接口")
