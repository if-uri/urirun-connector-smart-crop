# Author: Tom Sapletta · https://tom.sapletta.com
# Part of the ifURI solution.

"""Smart document/receipt crop connector for urirun.

Routes:

* ``smartcrop://host/document/query/crop``   -- detect + save cropped document image
* ``smartcrop://host/document/query/detect`` -- detect only, return bounding box
"""

from __future__ import annotations

import json
import math
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


def _distance(a: Any, b: Any) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _order_quad_points(points: Any) -> Any:
    import numpy as np

    pts = np.asarray(points, dtype="float32").reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(4)
    ordered[0] = pts[int(np.argmin(sums))]
    ordered[2] = pts[int(np.argmax(sums))]
    ordered[1] = pts[int(np.argmin(diffs))]
    ordered[3] = pts[int(np.argmax(diffs))]
    return ordered


def _expand_quad(points: Any, margin_ratio: float, width: int, height: int) -> Any:
    import numpy as np

    pts = np.asarray(points, dtype="float32").reshape(4, 2)
    center = pts.mean(axis=0)
    factor = 1.0 + (2.0 * max(0.0, float(margin_ratio)))
    expanded = center + (pts - center) * factor
    expanded[:, 0] = np.clip(expanded[:, 0], 0, max(0, width - 1))
    expanded[:, 1] = np.clip(expanded[:, 1], 0, max(0, height - 1))
    return expanded.astype("float32")


def _quad_dimensions(points: Any) -> tuple[int, int]:
    ordered = _order_quad_points(points)
    width_a = _distance(ordered[2], ordered[3])
    width_b = _distance(ordered[1], ordered[0])
    height_a = _distance(ordered[1], ordered[2])
    height_b = _distance(ordered[0], ordered[3])
    return max(1, int(round(max(width_a, width_b)))), max(1, int(round(max(height_a, height_b))))


def _warp_quad(full: Any, points: Any) -> Any:
    import cv2
    import numpy as np
    from PIL import Image

    ordered = _order_quad_points(points)
    width, height = _quad_dimensions(ordered)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(np.asarray(full.convert("RGB")), matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def _contour_quad(contour: Any) -> Any | None:
    import cv2
    import numpy as np

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    for epsilon_ratio in (0.015, 0.025, 0.035, 0.05, 0.075):
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype("float32")
    box_area = max(1.0, float(rect[1][0] * rect[1][1]))
    fill = float(cv2.contourArea(contour)) / box_area
    if fill >= 0.45:
        return box
    return None


def _opencv_document_crop(
    full: Any,
    source: Path,
    *,
    output_path: str | Path | None,
    output_dir: str | Path | None,
    save: bool,
    auto_orient: bool,
    prefer_portrait: bool,
    max_side: int,
    margin_ratio: float,
    quality: int,
    min_component_area_ratio: float,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"opencv unavailable: {exc}"}

    original_width, original_height = full.size
    scale = min(1.0, max(64, int(max_side)) / max(original_width, original_height))
    rgb = np.asarray(full.convert("RGB"))
    if scale < 1.0:
        analysis = cv2.resize(rgb, (max(1, int(original_width * scale)), max(1, int(original_height * scale))), interpolation=cv2.INTER_AREA)
    else:
        analysis = rgb
    ah, aw = analysis.shape[:2]
    gray = cv2.cvtColor(analysis, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    hsv = cv2.cvtColor(analysis, cv2.COLOR_RGB2HSV)
    light_mask = cv2.inRange(hsv, (0, 0, 135), (179, 105, 255))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    edges = cv2.Canny(blur, 45, 135)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge_mask = cv2.dilate(edges, edge_kernel, iterations=1)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    adaptive_edges = cv2.Canny(adaptive, 40, 120)
    adaptive_edges = cv2.dilate(adaptive_edges, edge_kernel, iterations=1)
    adaptive_edges = cv2.morphologyEx(adaptive_edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    masks = (
        ("light-region", light_mask),
        ("edge-contour", edge_mask),
        ("adaptive-edge", adaptive_edges),
    )
    min_area = max(180.0, float(aw * ah) * max(0.0001, float(min_component_area_ratio)))
    candidates: list[dict[str, Any]] = []
    for mask_name, mask in masks:
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 35 or h < 35:
                continue
            bbox_ratio = (w * h) / float(aw * ah)
            if bbox_ratio < 0.018 or bbox_ratio > 0.985:
                continue
            aspect = w / float(h)
            long_aspect = max(aspect, 1.0 / max(0.001, aspect))
            if long_aspect > 8.5:
                continue
            quad = _contour_quad(contour)
            if quad is None:
                continue
            quad_width, quad_height = _quad_dimensions(quad)
            if quad_width < 35 or quad_height < 35:
                continue
            quad_area = abs(float(cv2.contourArea(quad.astype("float32"))))
            quad_ratio = quad_area / float(aw * ah)
            if quad_ratio < 0.018 or quad_ratio > 0.985:
                continue
            fill = area / max(1.0, quad_area)
            touches_edge = x <= 2 or y <= 2 or x + w >= aw - 3 or y + h >= ah - 3
            if touches_edge and (bbox_ratio > 0.66 or quad_ratio > 0.66):
                continue
            region = gray[y:y + h, x:x + w]
            region_edges = edges[y:y + h, x:x + w]
            dark_fraction = float(np.count_nonzero(region < 115)) / float(max(1, region.size))
            edge_density = float(np.count_nonzero(region_edges)) / float(max(1, region_edges.size))
            edge_penalty = 0.86 if touches_edge else 1.0
            top_band_penalty = 0.18 if aspect > 3.0 and y < ah * 0.12 and h < ah * 0.38 else 1.0
            wide_strip_penalty = 0.48 if aspect > 4.0 else 0.76 if aspect > 3.2 else 1.0
            content_bonus = max(0.18, min(1.25, (dark_fraction * 18.0) + (edge_density * 10.0)))
            area_bonus = min(1.0, quad_ratio * 2.4)
            shape_bonus = min(1.25, max(0.35, fill))
            mask_bonus = 1.18 if mask_name != "light-region" else 1.0
            score = area * area_bonus * shape_bonus * edge_penalty * top_band_penalty * wide_strip_penalty * content_bonus * mask_bonus
            candidates.append({
                "mask": mask_name,
                "area": area,
                "fill": fill,
                "score": score,
                "bbox": (x, y, w, h),
                "bboxArea": bbox_ratio,
                "quadArea": quad_ratio,
                "aspect": aspect,
                "darkFraction": dark_fraction,
                "edgeDensity": edge_density,
                "touchesEdge": touches_edge,
                "quad": quad,
            })

    if not candidates:
        return {"ok": False, "reason": "no reliable opencv document contour"}

    best = max(candidates, key=lambda item: item["score"])
    inv_scale = 1.0 / scale
    quad = np.asarray(best["quad"], dtype="float32") * inv_scale
    quad = _expand_quad(quad, margin_ratio, original_width, original_height)
    min_xy = quad.min(axis=0)
    max_xy = quad.max(axis=0)
    box = (
        max(0, int(math.floor(float(min_xy[0])))),
        max(0, int(math.floor(float(min_xy[1])))),
        min(original_width, int(math.ceil(float(max_xy[0])))),
        min(original_height, int(math.ceil(float(max_xy[1])))),
    )
    crop_width = box[2] - box[0]
    crop_height = box[3] - box[1]
    if crop_width < 50 or crop_height < 50:
        return {"ok": False, "reason": "opencv crop too small", "box": list(box)}

    warped = _warp_quad(full, quad)
    oriented, orientation = _orient_document_image(warped, auto_orient=auto_orient, prefer_portrait=prefer_portrait)

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
        "method": "opencv-perspective",
        "path": target,
        "originalPath": str(source),
        "box": list(box),
        "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in _order_quad_points(quad)],
        "coverage": round(float(best["area"]) / float(aw * ah), 4),
        "bboxArea": round(float(best["bboxArea"]), 4),
        "quadArea": round(float(best["quadArea"]), 4),
        "threshold": None,
        "fill": round(float(best["fill"]), 4),
        "component": {
            "area": int(round(float(best["area"]))),
            "aspect": round(float(best["aspect"]), 4),
            "score": round(float(best["score"]), 4),
            "mask": best["mask"],
            "darkFraction": round(float(best["darkFraction"]), 4),
            "edgeDensity": round(float(best["edgeDensity"]), 4),
            "touchesEdge": bool(best["touchesEdge"]),
        },
        "orientation": orientation,
        "originalWidth": original_width,
        "originalHeight": original_height,
        "cropWidth": crop_width,
        "cropHeight": crop_height,
        "width": oriented.size[0],
        "height": oriented.size[1],
    }


def _text_region_document_crop(
    full: Any,
    source: Path,
    *,
    output_path: str | Path | None,
    output_dir: str | Path | None,
    save: bool,
    auto_orient: bool,
    prefer_portrait: bool,
    max_side: int,
    margin_ratio: float,
    quality: int,
) -> dict[str, Any]:
    """Find receipt-like paper by locating dark text near a bright document body.

    This fallback handles camera scenes where a bright background object touches
    the receipt, causing plain contour detection to return a near-full-frame
    connected component.
    """
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"opencv unavailable: {exc}"}

    original_width, original_height = full.size
    scale = min(1.0, max(64, int(max_side)) / max(original_width, original_height))
    rgb = np.asarray(full.convert("RGB"))
    if scale < 1.0:
        analysis = cv2.resize(rgb, (max(1, int(original_width * scale)), max(1, int(original_height * scale))), interpolation=cv2.INTER_AREA)
    else:
        analysis = rgb
    ah, aw = analysis.shape[:2]
    gray = cv2.cvtColor(analysis, cv2.COLOR_RGB2GRAY)

    text_candidates: list[dict[str, Any]] = []
    for dark_threshold in (90, 80, 70, 60):
        for bright_threshold in (175, 165, 155, 145):
            dark = cv2.inRange(gray, 0, dark_threshold - 1)
            bright = cv2.inRange(gray, bright_threshold, 255)
            near_bright = cv2.dilate(bright, cv2.getStructuringElement(cv2.MORPH_RECT, (33, 33)), iterations=1)
            mask = cv2.bitwise_and(dark, near_bright)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)), iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (19, 9)), iterations=2)
            contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < max(180.0, aw * ah * 0.0025):
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if w < 24 or h < 18:
                    continue
                bbox_ratio = (w * h) / float(aw * ah)
                if bbox_ratio < 0.0018 or bbox_ratio > 0.42:
                    continue
                aspect = w / float(h)
                if not (0.22 <= aspect <= 7.8):
                    continue
                touches_edge = x <= 2 or y <= 2 or x + w >= aw - 3 or y + h >= ah - 3
                if touches_edge and bbox_ratio > 0.18:
                    continue
                region = gray[y:y + h, x:x + w]
                ink_density = float(np.count_nonzero(region < dark_threshold)) / float(max(1, region.size))
                paper_density = float(np.count_nonzero(region > bright_threshold)) / float(max(1, region.size))
                if ink_density < 0.006:
                    continue
                score = area * min(1.4, ink_density * 14.0) * max(0.25, min(1.2, paper_density * 4.0))
                if 0.35 <= aspect <= 3.8:
                    score *= 1.25
                if y > ah * 0.18:
                    score *= 1.12
                text_candidates.append({
                    "bbox": (x, y, w, h),
                    "score": score,
                    "area": area,
                    "aspect": aspect,
                    "bboxArea": bbox_ratio,
                    "darkThreshold": dark_threshold,
                    "brightThreshold": bright_threshold,
                    "inkDensity": ink_density,
                    "paperDensity": paper_density,
                })

    if not text_candidates:
        return {"ok": False, "reason": "no receipt-like text region"}

    text = max(text_candidates, key=lambda item: item["score"])
    tx, ty, tw, th = text["bbox"]
    paper_candidates: list[dict[str, Any]] = []
    for bright_threshold in (205, 195, 185, 175, 165):
        bright = cv2.inRange(gray, bright_threshold, 255)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        contours, _hierarchy = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < max(220.0, aw * ah * 0.008):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            bbox_ratio = (w * h) / float(aw * ah)
            if bbox_ratio < 0.015 or bbox_ratio > 0.68:
                continue
            ix = max(0, min(x + w, tx + tw) - max(x, tx))
            iy = max(0, min(y + h, ty + th) - max(y, ty))
            overlap = (ix * iy) / float(max(1, tw * th))
            if overlap < 0.28:
                continue
            aspect = w / float(h)
            if not (0.18 <= aspect <= 5.0):
                continue
            touches_edge = x <= 2 or y <= 2 or x + w >= aw - 3 or y + h >= ah - 3
            if touches_edge and bbox_ratio > 0.36:
                continue
            top_overhang = max(0, ty - y) / float(max(1, h))
            bottom_overhang = max(0, (y + h) - (ty + th)) / float(max(1, h))
            side_overhang = (
                max(0, tx - x) + max(0, (x + w) - (tx + tw))
            ) / float(max(1, w))
            overhang_penalty = max(0.18, 1.0 - (max(0.0, top_overhang - 0.24) * 1.45) - (max(0.0, side_overhang - 0.72) * 0.65))
            score = area * overlap * overhang_penalty
            if top_overhang > 0.48:
                score *= 0.55
            if bottom_overhang < 0.03:
                score *= 0.82
            if 0.25 <= aspect <= 2.2:
                score *= 1.18
            paper_candidates.append({
                "bbox": (x, y, w, h),
                "score": score,
                "area": area,
                "aspect": aspect,
                "bboxArea": bbox_ratio,
                "overlap": overlap,
                "brightThreshold": bright_threshold,
                "topOverhang": top_overhang,
                "bottomOverhang": bottom_overhang,
                "sideOverhang": side_overhang,
            })

    if paper_candidates:
        paper = max(paper_candidates, key=lambda item: item["score"])
        x, y, w, h = paper["bbox"]
        component = {"text": text, "paper": paper}
    else:
        margin_x = max(8, int(tw * (0.16 + margin_ratio)))
        top_margin = max(18, int(th * (0.42 + margin_ratio)))
        bottom_margin = max(12, int(th * (0.16 + margin_ratio)))
        x = max(0, tx - margin_x)
        y = max(0, ty - top_margin)
        w = min(aw - x, tw + (2 * margin_x))
        h = min(ah - y, th + top_margin + bottom_margin)
        component = {"text": text, "paper": None}

    pad_x = max(3, int(w * max(0.0, margin_ratio)))
    pad_y = max(3, int(h * max(0.0, margin_ratio)))
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(aw, x + w + pad_x)
    bottom = min(ah, y + h + pad_y)
    inv_scale = 1.0 / scale
    box = (
        max(0, int(math.floor(left * inv_scale))),
        max(0, int(math.floor(top * inv_scale))),
        min(original_width, int(math.ceil(right * inv_scale))),
        min(original_height, int(math.ceil(bottom * inv_scale))),
    )
    crop_width = box[2] - box[0]
    crop_height = box[3] - box[1]
    if crop_width < 50 or crop_height < 50:
        return {"ok": False, "reason": "text-region crop too small", "box": list(box)}
    bbox_area = (crop_width * crop_height) / float(original_width * original_height)
    if bbox_area > 0.72:
        return {"ok": False, "reason": "text-region crop too large", "box": list(box), "bboxArea": round(bbox_area, 4)}

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
        "method": "text-region",
        "path": target,
        "originalPath": str(source),
        "box": list(box),
        "coverage": round(float(component["text"]["area"]) / float(aw * ah), 4),
        "bboxArea": round(bbox_area, 4),
        "quadArea": None,
        "threshold": component["text"]["darkThreshold"],
        "fill": round(float(component["text"]["inkDensity"]), 4),
        "component": {
            "area": int(round(float(component["text"]["area"]))),
            "aspect": round(float(component["text"]["aspect"]), 4),
            "score": round(float(component["text"]["score"]), 4),
            "mask": "text-region",
            "darkThreshold": component["text"]["darkThreshold"],
            "brightThreshold": component["text"]["brightThreshold"],
            "inkDensity": round(float(component["text"]["inkDensity"]), 4),
            "paperDensity": round(float(component["text"]["paperDensity"]), 4),
            "paper": None if component["paper"] is None else {
                "area": int(round(float(component["paper"]["area"]))),
                "aspect": round(float(component["paper"]["aspect"]), 4),
                "bboxArea": round(float(component["paper"]["bboxArea"]), 4),
                "overlap": round(float(component["paper"]["overlap"]), 4),
                "brightThreshold": component["paper"]["brightThreshold"],
                "topOverhang": round(float(component["paper"]["topOverhang"]), 4),
                "bottomOverhang": round(float(component["paper"]["bottomOverhang"]), 4),
                "sideOverhang": round(float(component["paper"]["sideOverhang"]), 4),
            },
        },
        "orientation": orientation,
        "originalWidth": original_width,
        "originalHeight": original_height,
        "cropWidth": crop_width,
        "cropHeight": crop_height,
        "width": oriented.size[0],
        "height": oriented.size[1],
    }


def _fill_ratio_document_crop(
    full: Any,
    source: Path,
    *,
    output_path: str | Path | None,
    output_dir: str | Path | None,
    save: bool,
    auto_orient: bool,
    prefer_portrait: bool,
    max_side: int,
    margin_ratio: float,
    quality: int,
) -> dict[str, Any]:
    """Pick the SOLID bright rectangle (the paper sheet) by connected-component fill-ratio.

    Robust to the hard real-world case the contour tier fails on: a small receipt next to a
    large bright distractor (tape, table edge, light wall). The receipt is a high-fill
    rectangle; the distractor is an L-shape/streak with a low fill-ratio, so it loses even
    when it is bigger or brighter. Threshold the brightest pixels at a few high percentiles,
    morphologically close text gaps, then keep the component maximising fill x area."""
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"opencv unavailable: {exc}"}

    original_width, original_height = full.size
    scale = min(1.0, max(64, int(max_side)) / max(original_width, original_height))
    rgb = np.asarray(full.convert("RGB"))
    analysis = cv2.resize(rgb, (max(1, int(original_width * scale)), max(1, int(original_height * scale))),
                          interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb
    ah, aw = analysis.shape[:2]
    gray = cv2.cvtColor(analysis, cv2.COLOR_RGB2GRAY).astype(np.float32)
    total = float(gray.size)
    close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    best: tuple[float, int, int, int, int, float, float] | None = None
    for pct in (96, 94, 92, 90):
        thr = float(np.percentile(gray, pct))
        mask = (gray >= thr).astype("uint8")
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close, iterations=1)
        count, _labels, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, count):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            box = bw * bh
            if box < 0.01 * total or box > 0.6 * total or bw < 20 or bh < 20:
                continue
            touches = x <= 1 or y <= 1 or x + bw >= aw - 1 or y + bh >= ah - 1
            if touches:
                continue
            fill = area / float(box)
            aspect = max(bw, bh) / float(max(1, min(bw, bh)))
            if fill > 0.55 and aspect < 6:
                score = fill * area
                if best is None or score > best[0]:
                    best = (score, x, y, bw, bh, fill, aspect)
    if best is None:
        return {"ok": False, "reason": "no solid sheet component"}

    score, x, y, bw, bh, fill, aspect = best
    inv = 1.0 / scale
    px = int(bw * max(0.0, margin_ratio))
    py = int(bh * max(0.0, margin_ratio))
    left = max(0, int((x - px) * inv))
    top = max(0, int((y - py) * inv))
    right = min(original_width, int((x + bw + px) * inv))
    bottom = min(original_height, int((y + bh + py) * inv))
    crop_width, crop_height = right - left, bottom - top
    if crop_width < 50 or crop_height < 50:
        return {"ok": False, "reason": "fill-ratio crop too small"}
    bbox_area = (crop_width * crop_height) / float(original_width * original_height)

    oriented, orientation = _orient_document_image(full.crop((left, top, right, bottom)),
                                                   auto_orient=auto_orient, prefer_portrait=prefer_portrait)
    target = ""
    if save:
        if output_path:
            target_path = Path(output_path).expanduser().resolve()
        elif output_dir:
            target_path = Path(output_dir).expanduser().resolve() / f"{source.stem}-document-crop.jpg"
        else:
            target_path = source.with_name(f"{source.stem}-document-crop.jpg")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        oriented.save(target_path, format="JPEG", quality=max(1, min(100, int(quality))), optimize=True)
        target = str(target_path)

    return {
        "ok": True,
        "method": "fill-ratio",
        "path": target,
        "originalPath": str(source),
        "box": [left, top, right, bottom],
        "coverage": round((bw * bh * fill) / total, 4),
        "bboxArea": round(bbox_area, 4),
        "quadArea": None,
        "threshold": None,
        "fill": round(fill, 4),
        "component": {
            "area": int(round(bw * bh * fill)),
            "aspect": round(aspect, 4),
            "score": round(score, 4),
            "mask": "fill-ratio",
            "touchesEdge": False,
        },
        "orientation": orientation,
        "originalWidth": original_width,
        "originalHeight": original_height,
        "cropWidth": crop_width,
        "cropHeight": crop_height,
        "width": oriented.size[0],
        "height": oriented.size[1],
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

    opencv_result = _opencv_document_crop(
        full,
        source,
        output_path=output_path,
        output_dir=output_dir,
        save=save,
        auto_orient=auto_orient,
        prefer_portrait=prefer_portrait,
        max_side=max_side,
        margin_ratio=margin_ratio,
        quality=quality,
        min_component_area_ratio=min_component_area_ratio,
    )
    # A confident, tight opencv crop wins (preserves perspective/edge-document handling).
    # Exception: a large, edge-touching component is often a bright background strip
    # (tape/table/fabric) next to a smaller receipt. Let fill-ratio compete first.
    opencv_component = opencv_result.get("component") if isinstance(opencv_result.get("component"), dict) else {}
    opencv_area = float(opencv_result.get("bboxArea") or 1.0) if opencv_result.get("ok") else 1.0
    large_edge_component = bool(opencv_component.get("touchesEdge")) and opencv_area > 0.38
    if opencv_result.get("ok") and opencv_area <= 0.55 and not large_edge_component:
        return opencv_result

    # opencv missing or kept most of the frame (small sheet + big bright distractor). The
    # fill-ratio detector finds the solid paper rectangle; prefer it only when it is clearly
    # tighter, so a document that legitimately fills the frame still uses opencv-perspective.
    fill_result = _fill_ratio_document_crop(
        full,
        source,
        output_path=output_path,
        output_dir=output_dir,
        save=save,
        auto_orient=auto_orient,
        prefer_portrait=prefer_portrait,
        max_side=max_side,
        margin_ratio=margin_ratio,
        quality=quality,
    )
    if fill_result.get("ok"):
        fill_area = float(fill_result.get("bboxArea") or 1.0)
        if fill_area < 0.5 and (not opencv_result.get("ok") or fill_area < 0.6 * opencv_area):
            return fill_result

    if opencv_result.get("ok"):
        return opencv_result

    text_region_result = _text_region_document_crop(
        full,
        source,
        output_path=output_path,
        output_dir=output_dir,
        save=save,
        auto_orient=auto_orient,
        prefer_portrait=prefer_portrait,
        max_side=max_side,
        margin_ratio=margin_ratio,
        quality=quality,
    )
    if text_region_result.get("ok"):
        return text_region_result

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
