# Author: Tom Sapletta · https://tom.sapletta.com
# Part of the ifURI solution.

"""Smart document/receipt crop connector for urirun.

Routes:

* ``smartcrop://host/document/query/crop``   -- detect + save cropped document image
* ``smartcrop://host/document/query/detect`` -- detect only, return bounding box
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import urirun

CONNECTOR_ID = "smart-crop"
conn = urirun.connector(CONNECTOR_ID, scheme="smartcrop")


def _path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _candidate_components(
    raw: bytes,
    width: int,
    height: int,
    threshold: int,
    *,
    min_component_area_ratio: float,
) -> list[dict[str, Any]]:
    from PIL import Image, ImageFilter

    mask = bytearray(width * height)
    out = 0
    for idx in range(0, len(raw), 3):
        r, g, b = raw[idx], raw[idx + 1], raw[idx + 2]
        hi = max(r, g, b)
        lo = min(r, g, b)
        lum = (299 * r + 587 * g + 114 * b) // 1000
        sat = hi - lo
        if lum >= threshold and sat <= 90:
            mask[out] = 255
        out += 1

    closed = Image.frombytes("L", (width, height), bytes(mask)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    data = bytearray(closed.tobytes())
    seen = bytearray(len(data))
    min_area = max(120, int(width * height * max(0.0001, min_component_area_ratio)))
    found: list[dict[str, Any]] = []

    for start, value in enumerate(data):
        if not value or seen[start]:
            continue
        queue = [start]
        seen[start] = 1
        area = 0
        minx = width
        maxx = 0
        miny = height
        maxy = 0
        for cur in queue:
            area += 1
            y, x = divmod(cur, width)
            if x < minx:
                minx = x
            if x > maxx:
                maxx = x
            if y < miny:
                miny = y
            if y > maxy:
                maxy = y
            for nxt in (cur - 1, cur + 1, cur - width, cur + width):
                if nxt < 0 or nxt >= len(data) or seen[nxt] or not data[nxt]:
                    continue
                ny, nx = divmod(nxt, width)
                if abs(nx - x) + abs(ny - y) != 1:
                    continue
                seen[nxt] = 1
                queue.append(nxt)
        if area < min_area:
            continue

        bw = maxx - minx + 1
        bh = maxy - miny + 1
        aspect = bw / float(bh)
        bbox_area = (bw * bh) / float(width * height)
        touches_edge = minx <= 2 or miny <= 2 or maxx >= width - 3 or maxy >= height - 3
        if touches_edge or bw < 35 or bh < 35 or not (0.18 <= aspect <= 6.5) or bbox_area < 0.02 or bbox_area > 0.82:
            continue

        fill = area / float(bw * bh)
        aspect_penalty = 0.22 if aspect > 3.2 else 0.55 if aspect > 2.6 else 1.0
        score = area * min(1.0, fill * 1.6) * aspect_penalty
        found.append({
            "threshold": threshold,
            "area": area,
            "fill": fill,
            "score": score,
            "left": minx,
            "top": miny,
            "right": maxx,
            "bottom": maxy,
            "width": bw,
            "height": bh,
            "aspect": aspect,
            "bboxArea": bbox_area,
        })
    return found


def _projection_peakiness(values: list[int]) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    mean = total / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return variance / float(total + 1)


def _line_orientation_score(image) -> dict[str, Any]:
    max_side = 420
    scale = min(1.0, max_side / max(image.size))
    sample = image.resize((max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale)))) if scale < 1.0 else image
    gray = sample.convert("L")
    width, height = gray.size
    pixels = gray.tobytes()
    threshold = 170
    row_counts = [0] * height
    col_counts = [0] * width
    dark = 0
    for idx, value in enumerate(pixels):
        if value >= threshold:
            continue
        y, x = divmod(idx, width)
        row_counts[y] += 1
        col_counts[x] += 1
        dark += 1
    if dark < max(24, int(width * height * 0.003)):
        return {"score": 0.0, "darkFraction": dark / float(width * height), "rowPeak": 0.0, "colPeak": 0.0}
    row_peak = _projection_peakiness(row_counts)
    col_peak = _projection_peakiness(col_counts)
    return {
        "score": row_peak - col_peak,
        "darkFraction": dark / float(width * height),
        "rowPeak": row_peak,
        "colPeak": col_peak,
    }


def _orient_document_image(image, *, auto_orient: bool = True, prefer_portrait: bool = True) -> tuple[Any, dict[str, Any]]:
    if not auto_orient:
        return image, {"enabled": False, "angle": 0, "reason": "disabled", "width": image.size[0], "height": image.size[1]}

    candidates: list[dict[str, Any]] = []
    for angle in (0, 90, 180, 270):
        rotated = image.rotate(angle, expand=True) if angle else image
        width, height = rotated.size
        projection = _line_orientation_score(rotated)
        portrait_bonus = 0.28 if prefer_portrait and height >= width else 0.0
        score = float(projection["score"]) + portrait_bonus
        candidates.append({
            "angle": angle,
            "score": score,
            "width": width,
            "height": height,
            "portrait": height >= width,
            "projection": projection,
        })

    pool = candidates
    if prefer_portrait:
        portrait_candidates = [item for item in candidates if item["portrait"]]
        if portrait_candidates:
            pool = portrait_candidates
    best = max(pool, key=lambda item: item["score"])
    oriented = image.rotate(int(best["angle"]), expand=True) if best["angle"] else image
    return oriented, {
        "enabled": True,
        "angle": int(best["angle"]),
        "rotated": bool(best["angle"]),
        "preferPortrait": prefer_portrait,
        "width": oriented.size[0],
        "height": oriented.size[1],
        "score": round(float(best["score"]), 6),
        "candidates": [
            {
                "angle": item["angle"],
                "score": round(float(item["score"]), 6),
                "width": item["width"],
                "height": item["height"],
                "portrait": item["portrait"],
            }
            for item in candidates
        ],
    }


def detect_document_crop(
    image: str | Path,
    *,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    save: bool = True,
    auto_orient: bool = True,
    prefer_portrait: bool = True,
    max_side: int = 700,
    margin_ratio: float = 0.035,
    quality: int = 94,
    min_component_area_ratio: float = 0.004,
) -> dict[str, Any]:
    """Detect and optionally write a cropped document image.

    Returns a metadata dict. ``ok`` is true only when a reliable crop is written
    or, with ``save=False``, when a reliable box is detected.
    """
    from PIL import Image, ImageOps

    source = Path(image).expanduser().resolve()
    if not source.is_file():
        return {"ok": False, "reason": f"image not found: {source}", "originalPath": str(source)}

    try:
        with Image.open(source) as opened:
            full = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"image decode failed: {exc}", "originalPath": str(source)}

    original_width, original_height = full.size
    if original_width < 80 or original_height < 80:
        return {
            "ok": False,
            "reason": "image too small",
            "originalPath": str(source),
            "originalWidth": original_width,
            "originalHeight": original_height,
        }

    scale = min(1.0, max(64, int(max_side)) / max(original_width, original_height))
    analysis = full.resize((max(1, int(original_width * scale)), max(1, int(original_height * scale)))) if scale < 1.0 else full
    aw, ah = analysis.size
    raw = analysis.tobytes()

    candidates: list[dict[str, Any]] = []
    for threshold in (185, 180, 190, 175, 200, 170):
        candidates.extend(_candidate_components(
            raw,
            aw,
            ah,
            threshold,
            min_component_area_ratio=min_component_area_ratio,
        ))
    if not candidates:
        return {
            "ok": False,
            "reason": "no reliable document component",
            "originalPath": str(source),
            "originalWidth": original_width,
            "originalHeight": original_height,
        }

    best = max(candidates, key=lambda item: item["score"])
    left = int(best["left"])
    top = int(best["top"])
    right = int(best["right"])
    bottom = int(best["bottom"])
    bw = max(1, right - left + 1)
    bh = max(1, bottom - top + 1)
    coverage = best["area"] / float(aw * ah)
    bbox_area = best["bboxArea"]
    if bbox_area > 0.92:
        return {
            "ok": False,
            "reason": "document already fills frame",
            "originalPath": str(source),
            "coverage": round(coverage, 4),
            "bboxArea": round(bbox_area, 4),
            "originalWidth": original_width,
            "originalHeight": original_height,
        }

    pad_x = max(4, int(bw * max(0.0, margin_ratio)))
    pad_y = max(4, int(bh * max(0.0, margin_ratio)))
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(aw - 1, right + pad_x)
    bottom = min(ah - 1, bottom + pad_y)

    inv_scale = 1.0 / scale
    box = (
        max(0, int(left * inv_scale)),
        max(0, int(top * inv_scale)),
        min(original_width, int((right + 1) * inv_scale)),
        min(original_height, int((bottom + 1) * inv_scale)),
    )
    crop_width = box[2] - box[0]
    crop_height = box[3] - box[1]
    if crop_width < 50 or crop_height < 50:
        return {"ok": False, "reason": "crop too small", "originalPath": str(source), "box": list(box)}
    if box[0] <= 3 and box[1] <= 3 and box[2] >= original_width - 3 and box[3] >= original_height - 3:
        return {"ok": False, "reason": "crop equals original", "originalPath": str(source), "box": list(box)}

    cropped = full.crop(box)
    oriented, orientation = _orient_document_image(cropped, auto_orient=auto_orient, prefer_portrait=prefer_portrait)

    target = ""
    if save:
        if output_path:
            target_path = Path(output_path).expanduser().resolve()
        elif output_dir:
            out_dir = Path(output_dir).expanduser().resolve()
            target_path = out_dir / f"{source.stem}-document-crop.jpg"
        else:
            target_path = source.with_name(f"{source.stem}-document-crop.jpg")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        oriented.save(target_path, format="JPEG", quality=max(1, min(100, int(quality))), optimize=True)
        target = str(target_path)

    return {
        "ok": True,
        "method": "connected-component",
        "path": target,
        "originalPath": str(source),
        "box": list(box),
        "coverage": round(coverage, 4),
        "bboxArea": round(bbox_area, 4),
        "threshold": best["threshold"],
        "fill": round(best["fill"], 4),
        "component": {
            "area": best["area"],
            "aspect": round(best["aspect"], 4),
            "score": round(best["score"], 4),
        },
        "orientation": orientation,
        "originalWidth": original_width,
        "originalHeight": original_height,
        "cropWidth": crop_width,
        "cropHeight": crop_height,
        "width": oriented.size[0],
        "height": oriented.size[1],
    }


@conn.handler("document/query/detect", isolated=True, meta={"label": "Detect document crop box", "cliAlias": "detect"})
def document_detect(
    image: str = "",
    auto_orient: bool = True,
    prefer_portrait: bool = True,
    max_side: int = 700,
    margin_ratio: float = 0.035,
    min_component_area_ratio: float = 0.004,
) -> dict[str, Any]:
    """Detect a document/receipt bounding box without writing a cropped file."""
    if not image:
        return urirun.fail("image is required", connector=CONNECTOR_ID)
    result = detect_document_crop(
        image,
        save=False,
        auto_orient=auto_orient,
        prefer_portrait=prefer_portrait,
        max_side=max_side,
        margin_ratio=margin_ratio,
        min_component_area_ratio=min_component_area_ratio,
    )
    if result.get("ok"):
        return urirun.ok(connector=CONNECTOR_ID, crop=result, image=str(_path(image)))
    return urirun.fail(str(result.get("reason", "document not detected")), connector=CONNECTOR_ID, crop=result, image=str(_path(image)))


@conn.handler("document/query/crop", isolated=True, meta={"label": "Crop document from image", "cliAlias": "crop"})
def document_crop(
    image: str = "",
    output_path: str = "",
    output_dir: str = "",
    max_side: int = 700,
    margin_ratio: float = 0.035,
    quality: int = 94,
    auto_orient: bool = True,
    prefer_portrait: bool = True,
    fail_if_uncertain: bool = False,
    min_component_area_ratio: float = 0.004,
) -> dict[str, Any]:
    """Detect and crop a document/receipt from an image.

    With ``fail_if_uncertain=false`` (default), the route succeeds and returns the
    original path when no reliable crop is available. That makes it safe as a
    preprocessing step before OCR.
    """
    if not image:
        return urirun.fail("image is required", connector=CONNECTOR_ID)
    result = detect_document_crop(
        image,
        output_path=output_path or None,
        output_dir=output_dir or None,
        save=True,
        auto_orient=auto_orient,
        prefer_portrait=prefer_portrait,
        max_side=max_side,
        margin_ratio=margin_ratio,
        quality=quality,
        min_component_area_ratio=min_component_area_ratio,
    )
    if result.get("ok"):
        return urirun.ok(
            connector=CONNECTOR_ID,
            image=str(_path(image)),
            path=result.get("path") or str(_path(image)),
            originalPath=result.get("originalPath") or str(_path(image)),
            crop=result,
        )
    if fail_if_uncertain:
        return urirun.fail(str(result.get("reason", "document not detected")), connector=CONNECTOR_ID, image=str(_path(image)), crop=result)
    return urirun.ok(
        connector=CONNECTOR_ID,
        image=str(_path(image)),
        path=str(_path(image)),
        originalPath=str(_path(image)),
        crop=result,
    )


def connector_manifest() -> dict[str, Any]:
    manifest_path = Path(__file__).with_name("connector.manifest.json")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def urirun_bindings() -> dict[str, Any]:
    return conn.bindings()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"--help", "-h"}:
        print("usage: urirun-smart-crop <image> [output-dir]", file=sys.stderr)
        return 2
    image = argv[0]
    output_dir = argv[1] if len(argv) > 1 else ""
    print(json.dumps(document_crop(image=image, output_dir=output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
