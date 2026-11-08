from __future__ import annotations
import json,time
from pathlib import Path
from typing import Any
from . import __version__
from .c2pa_check import detect_c2pa
from .classifier import predict_heuristic
from .ela_check import analyze_ela
from .evidence import Direction,Evidence,Strength
from .exif_check import detect_exif
from .fft_check import analyze_fft
from .fusion import fuse
from .prnu_check import analyze_prnu


def detect(image_path: str | Path, *, include_intermediate: bool = False,
           use_ml: bool = False) -> dict[str, Any]:
    """运行全流水线。

    此发行版拒绝 use_ml=True；不附带预训练模型或自动下载入口。
    include_intermediate=True 时返回 numpy 中间结果给 GUI 可视化。
    """
    t0 = time.time()
    path = Path(image_path)
    all_ev: list[Evidence] = []
    intermediate: dict[str, Any] = {}
    module_status: dict[str, str] = {}

    pipeline_steps: list[tuple[str, Any]] = [
        ("c2pa", lambda: (detect_c2pa(path), {})),
        ("exif", lambda: (detect_exif(path), {})),
        ("fft", lambda: analyze_fft(path)),
        ("ela", lambda: analyze_ela(path)),
        ("prnu", lambda: analyze_prnu(path)),
        ("classifier", lambda: (predict_heuristic(path), {})),
    ]
    if use_ml:
        raise ValueError("此发行版不包含模型或自动下载")

    # 严格分模块 try/except，单一模块失败不影响其他模块
    for name, fn in pipeline_steps:
        try:
            evs, inter = fn()
            all_ev.extend(evs)
            if inter:
                intermediate[name] = inter
            module_status[name] = "ok"
        except Exception as exc:  # pragma: no cover
            module_status[name] = f"error: {exc}"
            all_ev.append(Evidence(
                module=name, name="module_error",
                direction=Direction.NEUTRAL, strength=Strength.INFO, score=0.0,
                description=f"模块 {name} 运行失败：{exc}",
            ))

    result = fuse(all_ev)
    result["meta"] = {
        "version": __version__,
        "image_path": str(path),
        "image_name": path.name,
        "elapsed_sec": round(time.time() - t0, 3),
        "module_status": module_status,
    }
    if include_intermediate:
        result["intermediate"] = intermediate
    return result

def to_json(result: dict[str, Any], indent: int = 2) -> str:
    """安全序列化（去掉 numpy 中间结果）。"""
    cleaned = {k: v for k, v in result.items() if k != "intermediate"}
    return json.dumps(cleaned, ensure_ascii=False, indent=indent, default=str)
