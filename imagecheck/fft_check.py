"""FFT 频域分析。

思路：
1. 灰度化 → 减去均值 → 加 Hann 窗 → 2D FFT → 对数谱
2. 计算径向能量谱（高频能量占比）。真实相机经 ISP/JPEG 处理后高频快速衰减；
   多数 diffusion / GAN 上采样会留下相对**更平、更规则**的高频能量分布。
3. 检测谱图里的规则性"网格峰值"（GAN/上采样 checkerboard artifact）。

输出 1~2 条 Evidence。所有阈值都是经验值，标在常量里方便后续调。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .evidence import Direction, Evidence, Strength


# 经验阈值
_HF_RATIO_AI_HIGH = 0.55      # 高频能量占比 > 此值 → 偏 AI
_HF_RATIO_REAL_LOW = 0.35     # 高频能量占比 < 此值 → 偏真实
_PEAK_Z_AI = 4.0              # 谱峰值 z-score > 此值 → 怀疑规则伪影


def _to_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    # Rec.601
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)


def _radial_profile(power: np.ndarray) -> np.ndarray:
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    r_max = min(cy, cx)
    r = np.clip(r, 0, r_max)
    tbin = np.bincount(r.ravel(), weights=power.ravel())
    n = np.bincount(r.ravel())
    n = np.where(n == 0, 1, n)
    return (tbin / n)[: r_max + 1]


def analyze_fft(image_path: str | Path, target_size: int = 512) -> tuple[list[Evidence], dict]:
    """返回 (证据列表, 中间结果字典) — 中间结果给 UI 画频谱图用。"""
    path = Path(image_path)
    if not path.exists():
        return [], {}

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            # 居中裁正方形 + 缩放，保证 FFT 各向同性
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            im = im.resize((target_size, target_size), Image.LANCZOS)
            arr = np.asarray(im)
    except Exception as e:
        return [Evidence(module="fft", name="error", direction=Direction.NEUTRAL,
                         strength=Strength.INFO, score=0.0,
                         description=f"FFT 读图失败：{e}")], {}

    gray = _to_gray(arr)
    gray = gray - gray.mean()
    # Hann 窗抑制边界泄漏
    win_y = np.hanning(gray.shape[0])[:, None]
    win_x = np.hanning(gray.shape[1])[None, :]
    gray = gray * (win_y * win_x)

    f = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(f) ** 2
    log_spec = np.log10(np.maximum(power, 1e-9))

    radial = _radial_profile(power)
    total = radial.sum() + 1e-9
    mid = len(radial) // 4
    high_freq_ratio = float(radial[mid:].sum() / total)

    # 规则峰值检测：取径向曲线的 1 阶差分，找 z-score 异常高峰
    diff = np.diff(radial)
    if diff.size > 4:
        dz = (diff - diff.mean()) / (diff.std() + 1e-9)
        peak_z = float(dz.max())
    else:
        peak_z = 0.0

    evs: list[Evidence] = []

    if not np.any(power):
        evs.append(Evidence(
            module="fft", name="insufficient_signal",
            direction=Direction.NEUTRAL, strength=Strength.INFO, score=0.0,
            description="图像没有可分析的频谱能量",
            raw={"hf_ratio": high_freq_ratio},
        ))
    elif high_freq_ratio > _HF_RATIO_AI_HIGH:
        evs.append(Evidence(
            module="fft", name="high_freq_excess",
            direction=Direction.AI, strength=Strength.MEDIUM,
            score=min(1.0, (high_freq_ratio - _HF_RATIO_AI_HIGH) * 4 + 0.5),
            description=f"高频能量占比异常偏高（{high_freq_ratio:.2%}），可能为上采样伪影",
            raw={"hf_ratio": high_freq_ratio},
        ))
    elif high_freq_ratio < _HF_RATIO_REAL_LOW:
        evs.append(Evidence(
            module="fft", name="high_freq_natural_decay",
            direction=Direction.REAL, strength=Strength.WEAK,
            score=0.4,
            description=f"高频能量自然衰减（占比 {high_freq_ratio:.2%}），符合相机 ISP 特征",
            raw={"hf_ratio": high_freq_ratio},
        ))
    else:
        evs.append(Evidence(
            module="fft", name="high_freq_neutral",
            direction=Direction.NEUTRAL, strength=Strength.INFO, score=0.0,
            description=f"高频能量占比中性（{high_freq_ratio:.2%}）",
            raw={"hf_ratio": high_freq_ratio},
        ))

    if peak_z > _PEAK_Z_AI:
        # 单个峰值不应封顶；上限 0.7，需要其它证据共同支撑
        evs.append(Evidence(
            module="fft", name="spectral_regular_peak",
            direction=Direction.AI, strength=Strength.MEDIUM,
            score=min(0.7, (peak_z - _PEAK_Z_AI) * 0.05 + 0.5),
            description=f"频谱径向曲线出现规则性峰值（z={peak_z:.1f}），疑似生成模型上采样网格",
            raw={"peak_z": peak_z},
        ))

    return evs, {
        "log_spectrum": log_spec,
        "radial": radial,
        "hf_ratio": high_freq_ratio,
        "peak_z": peak_z,
    }
