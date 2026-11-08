"""C2PA / Content Credentials 检测。

完整 C2PA 验证依赖 `c2pa-python`（rust 绑定，安装较重）。本模块采用**渐进策略**：
1. 若 `c2pa` 可用，调用其 API 读取声明；本模块不授予签名有效或来源真实的结论。
2. 否则退化为 JUMBF/manifest 标记探测：扫描文件字节里是否包含 C2PA 标识
   （`jumb`/`c2pa`/`c2ma`/`c2sh`），可以判断"这张图嵌入了 C2PA 元数据"，
   但不能验证签名有效性 — 此时输出 INFO 级证据，不参与打分。

返回 Evidence 列表（可能为空）。
"""
from __future__ import annotations

from pathlib import Path

from .evidence import Direction, Evidence, Strength


def _scan_jumbf_marker(path: Path) -> bool:
    """快速扫描文件前 1MB 是否含 JUMBF / C2PA 标识。"""
    try:
        with path.open("rb") as f:
            head = f.read(1024 * 1024)
    except OSError:
        return False
    for marker in (b"jumb", b"c2pa", b"c2ma", b"c2sh", b"contentauth"):
        if marker in head:
            return True
    return False


def _try_official_c2pa(path: Path) -> tuple[bool, dict | None]:
    """如果安装了 c2pa-python，返回 (success, manifest_dict)。"""
    try:
        import c2pa  # type: ignore
    except Exception:
        return False, None
    # 仅使用支持显式离线Context的SDK；旧版回退字节标记，绝不采用默认联网读取。
    if not hasattr(c2pa, "Context") or not hasattr(c2pa, "Reader"):
        return False, None
    context = reader = None
    try:
        context = c2pa.Context.from_dict({"verify": {"remote_manifest_fetch": False, "ocsp_fetch": False}})
        reader = c2pa.Reader(str(path), context=context)
        import json
        return True, json.loads(reader.json())
    except Exception:
        return False, None
    finally:
        for item in (reader, context):
            if item is not None and callable(getattr(item, "close", None)):
                item.close()


def detect_c2pa(image_path: str | Path) -> list[Evidence]:
    path = Path(image_path)
    if not path.exists():
        return []

    evs: list[Evidence] = []
    ok, manifest = _try_official_c2pa(path)

    if ok and manifest:
        # 解析签名者
        signer = ""
        try:
            active = manifest.get("active_manifest") or ""
            mans = manifest.get("manifests") or {}
            if active and active in mans:
                cert = mans[active].get("signature_info", {}) or {}
                signer = (cert.get("issuer") or cert.get("common_name") or "").lower()
        except Exception:
            signer = ""

        # 解析到manifest不等于签名链通过；issuer不能证明画面真实或AI生成。
        # 本离线组件不维护信任根/撤销/版本状态合同，因此始终作为中性元数据。
        evs.append(Evidence(
            module="c2pa", name="manifest_present",
            direction=Direction.NEUTRAL, strength=Strength.INFO, score=0.0,
            description=f"读到C2PA声明（主体：{signer or '未知'}）；未独立验证来源真实性",
            raw={"signer": signer, "verified": False,
                 "validation_status_reported": manifest.get("validation_status"),
                 "validation_state_reported": manifest.get("validation_state"),
                 "verification_scope": "metadata-only"},
        ))
        return evs

    # 退化路径：仅探测 marker
    if _scan_jumbf_marker(path):
        evs.append(Evidence(
            module="c2pa", name="manifest_marker_detected",
            direction=Direction.NEUTRAL, strength=Strength.INFO, score=0.0,
            description="文件包含 C2PA/JUMBF 标识，但未启用官方 c2pa 库做签名验证",
            raw={"verified": False},
        ))
    return evs
