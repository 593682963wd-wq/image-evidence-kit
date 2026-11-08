"""EXIF / 元数据分析。

硬规则：
- Software / Make / Model / XMP 中明确出现已知 AI 生成器关键字 -> HARD AI
- 含完整相机三件套 (Make+Model+ExposureTime+ISO+FNumber+DateTime) -> STRONG REAL
- EXIF 完全为空 -> WEAK AI（很多 AI 图导出为纯净 PNG，但社交平台压缩也会清空，所以只算 WEAK）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from .evidence import Direction, Evidence, Strength


_AI_KEYWORDS = (
    "stable diffusion", "stablediffusion", "sdxl", "sd1.5", "sd 1.5",
    "midjourney", "mj v",
    "dall-e", "dalle", "dall·e",
    "openai",
    "comfyui", "comfy ui",
    "automatic1111", "a1111",
    "novelai",
    "firefly",
    "imagen",
    "gemini image",
    "runway",
    "flux", "flux.1",
    "leonardo.ai", "leonardo ai",
    "playground v",
    "ideogram",
    "即梦", "可灵", "豆包", "通义万相", "文心一格", "天工",
    "civitai",
)

_CAMERA_FIELDS = ("Make", "Model", "ExposureTime", "ISOSpeedRatings", "FNumber", "DateTimeOriginal")


def _read_exif(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as im:
            raw = im.getexif()
            if not raw:
                return {}
            out: dict[str, Any] = {}
            fields = dict(raw.items())
            # Camera exposure and original time normally live in the nested Exif IFD.
            try:
                fields.update(raw.get_ifd(34665))
            except (KeyError, OSError, ValueError, TypeError):
                pass
            for tag_id, val in fields.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                # 解码 bytes
                if isinstance(val, bytes):
                    try:
                        val = val.decode("utf-8", errors="ignore").strip("\x00").strip()
                    except Exception:
                        val = repr(val)
                out[tag] = val
            return out
    except Exception:
        return {}


def _read_xmp_text(path: Path) -> str:
    """快速读取 XMP packet（在文件前 256KB 内查找）。"""
    try:
        with path.open("rb") as f:
            blob = f.read(256 * 1024)
        start = blob.find(b"<x:xmpmeta")
        end = blob.find(b"</x:xmpmeta>")
        if start != -1 and end != -1:
            return blob[start:end].decode("utf-8", errors="ignore")
    except Exception:
        pass
    return ""


def _read_png_text(path: Path) -> dict[str, str]:
    """PNG tEXt / iTXt 块（A1111/ComfyUI 把 prompt 写在这里）。"""
    out: dict[str, str] = {}
    try:
        with Image.open(path) as im:
            if im.format == "PNG":
                for k, v in (im.info or {}).items():
                    if isinstance(v, str):
                        out[k] = v
    except Exception:
        pass
    return out


def detect_exif(image_path: str | Path) -> list[Evidence]:
    path = Path(image_path)
    if not path.exists():
        return []

    exif = _read_exif(path)
    xmp = _read_xmp_text(path).lower()
    png_text = _read_png_text(path)
    png_blob = " ".join(png_text.values()).lower()

    evs: list[Evidence] = []

    # 硬规则 1：AI 软件关键字命中
    blob_parts = [
        str(exif.get("Software", "")),
        str(exif.get("Artist", "")),
        str(exif.get("ImageDescription", "")),
        str(exif.get("UserComment", "")),
        str(exif.get("HostComputer", "")),
        xmp,
        png_blob,
    ]
    blob = " ".join(blob_parts).lower()
    hits = [k for k in _AI_KEYWORDS if k in blob]
    if hits:
        evs.append(Evidence(
            module="exif", name="ai_software_tag",
            direction=Direction.AI, strength=Strength.HARD, score=0.97,
            description=f"元数据明确包含 AI 生成器标识：{', '.join(hits[:3])}",
            raw={"hits": hits, "software": exif.get("Software"),
                 "png_text_keys": list(png_text.keys())},
        ))
        # PNG A1111/ComfyUI 的 parameters 段
        if "parameters" in png_text or "prompt" in png_text:
            evs.append(Evidence(
                module="exif", name="ai_workflow_payload",
                direction=Direction.AI, strength=Strength.HARD, score=0.99,
                description="PNG 包含 AI 生成工作流参数（A1111/ComfyUI 风格）",
                raw={"keys": list(png_text.keys())},
            ))
        return evs

    # 硬规则 2：完整相机三件套
    cam_present = sum(1 for k in _CAMERA_FIELDS if exif.get(k))
    if cam_present >= 4 and exif.get("Make") and exif.get("Model"):
        evs.append(Evidence(
            module="exif", name="camera_metadata_complete",
            direction=Direction.REAL, strength=Strength.STRONG, score=0.85,
            description=f"包含完整相机元数据：{exif.get('Make')} {exif.get('Model')}（{cam_present}/6 字段）",
            raw={"make": exif.get("Make"), "model": exif.get("Model"),
                 "fields_present": cam_present},
        ))
    elif not exif:
        evs.append(Evidence(
            module="exif", name="exif_empty",
            direction=Direction.AI, strength=Strength.WEAK, score=0.45,
            description="EXIF 完全为空（注意：社交平台压缩与截图也会导致此现象）",
            raw={},
        ))
    else:
        evs.append(Evidence(
            module="exif", name="exif_partial",
            direction=Direction.NEUTRAL, strength=Strength.INFO, score=0.0,
            description=f"EXIF 信息不完整（{len(exif)} 个字段）",
            raw={"fields": list(exif.keys())},
        ))

    return evs
