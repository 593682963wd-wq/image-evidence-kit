"""ELA — Error Level Analysis。

将图像以固定 quality 重存为 JPEG，再与原图相减取绝对差。真实图各区域压缩差应均匀；
被局部修改/合成的区域往往会出现"亮斑"。AI 生成图整体重压缩痕迹普遍**更均匀偏低**
（因为没有任何"原始"压缩痕迹），可作为一项弱~中证据。
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from .evidence import Direction, Evidence, Strength


_QUALITY = 90
_HIGH_LOCAL_VAR = 25.0   # 局部方差极高 → 可能局部 inpaint
_LOW_GLOBAL_MEAN = 2.0   # 全局均值极低 → 完全无原始压缩历史
_HIGH_GLOBAL_MEAN = 12.0 # 全局均值偏高 → 多次重压缩


def analyze_ela(image_path: str | Path, target_size: int = 1024) -> tuple[list[Evidence], dict]:
    path = Path(image_path)
    if not path.exists():
        return [], {}

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = min(1.0, target_size / max(w, h))
            if scale < 1.0:
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=_QUALITY)
            buf.seek(0)
            recompressed = Image.open(buf).convert("RGB")
            ela = np.abs(np.asarray(im, dtype=np.int16) -
                         np.asarray(recompressed, dtype=np.int16)).astype(np.uint8)
    except Exception as e:
        return [Evidence(module="ela", name="error", direction=Direction.NEUTRAL,
                         strength=Strength.INFO, score=0.0,
                         description=f"ELA 读图失败：{e}")], {}

    ela_gray = ela.mean(axis=2)
    g_mean = float(ela_gray.mean())
    g_max = float(ela_gray.max())

    # 局部方差：分块算 mean，再看块间标准差
    h2, w2 = ela_gray.shape
    block = min(32, h2, w2)
    yblk = max(1, h2 // block)
    xblk = max(1, w2 // block)
    blocks = ela_gray[: yblk * block, : xblk * block].reshape(yblk, block, xblk, block).mean(axis=(1, 3))
    local_var = float(blocks.std())

    evs: list[Evidence] = []

    if local_var > _HIGH_LOCAL_VAR:
        evs.append(Evidence(
            module="ela", name="local_inconsistency",
            direction=Direction.AI, strength=Strength.MEDIUM,
            score=min(1.0, (local_var - _HIGH_LOCAL_VAR) / 25.0 + 0.55),
            description=f"ELA 局部不一致（块方差 {local_var:.1f}），疑似局部修改/合成",
            raw={"block_std": local_var, "global_mean": g_mean},
        ))

    if g_mean < _LOW_GLOBAL_MEAN:
        evs.append(Evidence(
            module="ela", name="suspiciously_clean",
            direction=Direction.AI, strength=Strength.WEAK, score=0.4,
            description=f"ELA 全局响应极低（均值 {g_mean:.2f}），无原始压缩历史",
            raw={"global_mean": g_mean},
        ))
    elif g_mean > _HIGH_GLOBAL_MEAN:
        evs.append(Evidence(
            module="ela", name="multi_recompression",
            direction=Direction.NEUTRAL, strength=Strength.INFO, score=0.0,
            description=f"ELA 全局响应偏高（均值 {g_mean:.2f}），可能经过多次压缩",
            raw={"global_mean": g_mean},
        ))
    else:
        evs.append(Evidence(
            module="ela", name="ela_normal",
            direction=Direction.NEUTRAL, strength=Strength.INFO, score=0.0,
            description=f"ELA 响应正常（均值 {g_mean:.2f}，块方差 {local_var:.1f}）",
            raw={"global_mean": g_mean, "block_std": local_var},
        ))

    return evs, {"ela_image": ela, "global_mean": g_mean,
                 "global_max": g_max, "block_std": local_var}
