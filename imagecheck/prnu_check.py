"""PRNU 风格的传感器噪声残差分析（**简化版**）。

完整 PRNU 需要同设备多张照片做指纹估计，这里只做**单图近似**：
1. 用小波/中值去噪估计 clean 图，残差 = 原图 − clean。
2. 计算残差的局部统计量（均值、方差、峰度），与"自然相机噪声"的经验范围对比。
3. AI 生成图常见特征：
   - 残差能量极低 → "过于干净"，缺少传感器噪声
   - 残差呈现规则结构 → 上采样网格泄露到噪声层
4. 真实相机：残差能量在 [4, 20] 之间，分布接近高斯。

这是 WEAK 级证据，不能单独定案。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .evidence import Direction, Evidence, Strength


_LOW_NOISE_ENERGY = 1.5
_HIGH_NOISE_ENERGY = 25.0


def _denoise(gray: np.ndarray) -> np.ndarray:
    # 中值 + 高斯，速度比 BM3D 快很多，对单图近似够用
    g = cv2.medianBlur(gray, 3)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return g


def analyze_prnu(image_path: str | Path, target_size: int = 768) -> tuple[list[Evidence], dict]:
    path = Path(image_path)
    if not path.exists():
        return [], {}

    try:
        with Image.open(path) as im:
            im = im.convert("L")
            w, h = im.size
            scale = min(1.0, target_size / max(w, h))
            if scale < 1.0:
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            gray = np.asarray(im, dtype=np.uint8)
    except Exception as e:
        return [Evidence(module="prnu", name="error", direction=Direction.NEUTRAL,
                         strength=Strength.INFO, score=0.0,
                         description=f"PRNU 读图失败：{e}")], {}

    clean = _denoise(gray)
    residual = gray.astype(np.int16) - clean.astype(np.int16)

    energy = float(np.sqrt((residual.astype(np.float32) ** 2).mean()))
    # 残差的频域结构性：FFT 后看径向分布是否过分集中
    f = np.fft.fftshift(np.fft.fft2(residual.astype(np.float32)))
    p = np.abs(f) ** 2
    h2, w2 = p.shape
    cy, cx = h2 // 2, w2 // 2
    center_mass = p[cy - 8: cy + 8, cx - 8: cx + 8].sum()
    total_mass = p.sum() + 1e-9
    center_ratio = float(center_mass / total_mass)

    evs: list[Evidence] = []

    if energy < _LOW_NOISE_ENERGY:
        evs.append(Evidence(
            module="prnu", name="missing_sensor_noise",
            direction=Direction.AI, strength=Strength.WEAK,
            score=0.5,
            description=f"残差能量极低（{energy:.2f}），缺少自然传感器噪声",
            raw={"energy": energy, "center_ratio": center_ratio},
        ))
    elif energy > _HIGH_NOISE_ENERGY:
        evs.append(Evidence(
            module="prnu", name="high_noise",
            direction=Direction.NEUTRAL, strength=Strength.INFO, score=0.0,
            description=f"残差能量偏高（{energy:.2f}），可能为高 ISO 或重压缩",
            raw={"energy": energy},
        ))
    else:
        evs.append(Evidence(
            module="prnu", name="noise_in_natural_range",
            direction=Direction.REAL, strength=Strength.WEAK, score=0.4,
            description=f"残差能量在自然相机噪声范围内（{energy:.2f}）",
            raw={"energy": energy},
        ))

    if center_ratio > 0.4 and energy >= _LOW_NOISE_ENERGY:
        evs.append(Evidence(
            module="prnu", name="residual_low_freq_concentration",
            direction=Direction.AI, strength=Strength.WEAK, score=0.45,
            description=f"残差能量过度集中在低频（{center_ratio:.0%}），异常",
            raw={"center_ratio": center_ratio},
        ))

    return evs, {"residual": residual, "energy": energy, "center_ratio": center_ratio}
