from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import urirun
from urirun import v2

from urirun_connector_smart_crop import (
    connector_manifest,
    detect_document_crop,
    document_crop,
    document_detect,
    urirun_bindings,
)
from urirun_connector_smart_crop.core import (
    _orient_document_image,
    _partial_edge_document_reason,
    _prefer_geometry_over_text_boundary,
    _text_boundary_document_crop,
)

ROUTE_CROP = "smartcrop://host/document/query/crop"
ROUTE_DETECT = "smartcrop://host/document/query/detect"
ALL_ROUTES = {ROUTE_CROP, ROUTE_DETECT}


def _fake_tesseract(monkeypatch, data: dict) -> None:
    fake = types.ModuleType("pytesseract")
    fake.Output = types.SimpleNamespace(DICT="dict")
    fake.image_to_data = lambda _image, output_type=None: data
    monkeypatch.setitem(sys.modules, "pytesseract", fake)


def _receipt_scene(path: Path) -> Path:
    image = Image.new("RGB", (240, 180), (45, 47, 50))
    draw = ImageDraw.Draw(image)
    draw.rectangle((72, 20, 164, 158), fill=(246, 244, 235))
    draw.line((86, 42, 150, 42), fill=(35, 35, 35), width=2)
    draw.line((86, 70, 142, 70), fill=(35, 35, 35), width=2)
    image.save(path)
    return path


def _bright_band_scene(path: Path) -> Path:
    image = Image.new("RGB", (480, 360), (92, 92, 82))
    draw = ImageDraw.Draw(image)
    draw.rectangle((45, 15, 470, 108), fill=(186, 184, 168))
    draw.rectangle((145, 142, 312, 296), fill=(246, 242, 224))
    for x in (178, 210, 242, 276):
        draw.line((x, 162, x, 276), fill=(34, 34, 34), width=3)
    image.save(path)
    return path


def test_detect_document_crop_writes_crop(tmp_path: Path) -> None:
    source = _receipt_scene(tmp_path / "receipt.jpg")

    crop = detect_document_crop(source)

    assert crop["ok"] is True
    assert Path(crop["path"]).is_file()
    assert crop["width"] < 180
    assert crop["height"] > 120
    assert crop["box"][0] > 40
    assert crop["method"] in {"opencv-perspective", "connected-component"}


def test_text_boundary_keeps_already_cropped_document_frame(monkeypatch, tmp_path: Path) -> None:
    image = Image.new("RGB", (420, 720), (246, 246, 238))
    draw = ImageDraw.Draw(image)
    for y in range(36, 660, 82):
        draw.line((24, y, 394, y), fill=(20, 20, 20), width=3)
    source = tmp_path / "already-cropped.jpg"
    image.save(source)
    _fake_tesseract(monkeypatch, {
        "conf": ["91"] * 8,
        "text": [f"word{i}" for i in range(8)],
        "left": [28, 44, 26, 38, 30, 34, 40, 32],
        "top": [34, 120, 220, 310, 400, 500, 585, 642],
        "width": [330, 300, 350, 320, 345, 330, 310, 340],
        "height": [28, 30, 28, 30, 28, 30, 28, 28],
    })

    with Image.open(source) as full:
        crop = _text_boundary_document_crop(
            full.convert("RGB"),
            source,
            output_path=None,
            output_dir=None,
            save=False,
            auto_orient=True,
            prefer_portrait=True,
            margin_ratio=0.035,
            quality=94,
        )

    assert crop["ok"] is True
    assert crop["box"] == [0, 0, 420, 720]
    assert crop["bboxArea"] == 1.0


def test_text_boundary_expands_to_background_document_edges(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    image = Image.new("RGB", (520, 760), (116, 86, 55))
    draw = ImageDraw.Draw(image)
    draw.rectangle((128, 42, 402, 708), fill=(246, 245, 236))
    for y in range(92, 430, 58):
        draw.line((154, y, 374, y), fill=(18, 18, 18), width=4)
    source = tmp_path / "paper-on-background.jpg"
    image.save(source)
    _fake_tesseract(monkeypatch, {
        "conf": ["88"] * 7,
        "text": [f"line{i}" for i in range(7)],
        "left": [154, 160, 152, 166, 158, 162, 155],
        "top": [84, 140, 198, 254, 312, 368, 420],
        "width": [210, 190, 220, 185, 205, 198, 200],
        "height": [24, 24, 24, 24, 24, 24, 24],
    })

    with Image.open(source) as full:
        crop = _text_boundary_document_crop(
            full.convert("RGB"),
            source,
            output_path=None,
            output_dir=None,
            save=False,
            auto_orient=True,
            prefer_portrait=True,
            margin_ratio=0.035,
            quality=94,
        )

    assert crop["ok"] is True
    assert crop["background"]["ok"] is True
    assert crop["box"][0] <= 130
    assert crop["box"][2] >= 400
    assert crop["box"][3] >= 705


def test_text_boundary_extends_large_background_component_per_safe_side(monkeypatch, tmp_path: Path) -> None:
    image = Image.new("RGB", (1000, 1000), (122, 94, 66))
    draw = ImageDraw.Draw(image)
    draw.rectangle((285, 155, 835, 900), fill=(244, 241, 226))
    source = tmp_path / "receipt-on-edge-connected-background.jpg"
    image.save(source)
    _fake_tesseract(monkeypatch, {
        "conf": ["90"] * 8,
        "text": [f"item{i}" for i in range(8)],
        "left": [340, 360, 342, 358, 350, 362, 345, 355],
        "top": [300, 360, 430, 500, 570, 650, 720, 790],
        "width": [420, 380, 410, 400, 390, 395, 405, 385],
        "height": [28] * 8,
    })

    def fake_background_box(_full, _text_box):
        return {
            "ok": True,
            "box": [100, 0, 870, 910],
            "bboxArea": 0.7007,
            "textContainment": 1.0,
        }

    monkeypatch.setattr("urirun_connector_smart_crop.core._background_box_around_text", fake_background_box)

    with Image.open(source) as full:
        crop = _text_boundary_document_crop(
            full.convert("RGB"),
            source,
            output_path=None,
            output_dir=None,
            save=False,
            auto_orient=True,
            prefer_portrait=True,
            margin_ratio=0.035,
            quality=94,
        )

    assert crop["ok"] is True
    assert crop["backgroundUsedForCrop"] is True
    assert crop["background"]["usedForCrop"] is True
    assert crop["backgroundExtendedSides"] == {"left": False, "top": False, "right": True, "bottom": True}
    assert crop["box"][0] > 200
    assert crop["box"][1] > 100
    # Right/bottom snap to the detected paper edge (870 / 910) plus only a hairline
    # safety margin -- no longer over-padded out to the frame on the paper-backed sides.
    assert crop["box"][2] >= 880
    assert 910 <= crop["box"][3] <= 970


def test_text_boundary_does_not_extend_top_to_frame_touching_background(monkeypatch, tmp_path: Path) -> None:
    image = Image.new("RGB", (1440, 1440), (118, 96, 72))
    draw = ImageDraw.Draw(image)
    draw.rectangle((380, 300, 1020, 1280), fill=(247, 246, 238))
    source = tmp_path / "receipt-with-frame-touching-background.jpg"
    image.save(source)

    text_boxes = [
        [590, 344, 796, 387],
        [458, 377, 920, 426],
        [471, 415, 920, 456],
        [435, 456, 612, 490],
        [440, 1073, 975, 1113],
        [577, 1113, 829, 1151],
        [439, 1184, 631, 1221],
    ]

    monkeypatch.setattr(
        "urirun_connector_smart_crop.core._detect_text_boxes",
        lambda *_a, **_kw: {
            "ok": True,
            "backend": "paddleocr",
            "boxes": text_boxes,
            "wordCount": len(text_boxes),
            "lineHeight": 40,
            "clean": True,
        },
    )
    monkeypatch.setattr(
        "urirun_connector_smart_crop.core._background_box_around_text",
        lambda *_a, **_kw: {
            "ok": True,
            "box": [403, 0, 1440, 1227],
            "bboxArea": 0.6136,
            "textContainment": 1.0,
        },
    )

    with Image.open(source) as full:
        crop = _text_boundary_document_crop(
            full.convert("RGB"),
            source,
            output_path=None,
            output_dir=None,
            save=False,
            auto_orient=True,
            prefer_portrait=True,
            margin_ratio=0.035,
            quality=94,
            backend="paddleocr",
        )

    assert crop["ok"] is True
    assert crop["backgroundExtendedSides"]["top"] is False
    assert crop["box"][1] > 250
    assert crop["box"][1] < 330


def test_geometry_can_win_when_text_boundary_cuts_document_bottom() -> None:
    text = {
        "ok": True,
        "method": "text-boundary",
        "box": [440, 0, 1162, 1187],
        "originalHeight": 1440,
    }
    geometry = {
        "ok": True,
        "method": "opencv-perspective",
        "box": [453, 0, 1150, 1439],
        "originalHeight": 1440,
    }

    assert _prefer_geometry_over_text_boundary(text, geometry) is True


def test_geometry_can_win_when_low_word_text_crop_is_too_narrow() -> None:
    text = {
        "ok": True,
        "method": "text-boundary",
        "box": [357, 0, 940, 1440],
        "originalHeight": 1440,
        "wordCount": 10,
    }
    geometry = {
        "ok": True,
        "method": "opencv-perspective",
        "box": [354, 0, 1050, 1439],
        "originalHeight": 1440,
    }

    assert _prefer_geometry_over_text_boundary(text, geometry) is True


def test_geometry_can_win_when_text_boundary_starts_below_header() -> None:
    text = {
        "ok": True,
        "method": "text-boundary",
        "box": [383, 446, 985, 1440],
        "originalHeight": 1440,
        "wordCount": 29,
    }
    geometry = {
        "ok": True,
        "method": "opencv-perspective",
        "box": [377, 96, 1089, 1439],
        "originalHeight": 1440,
    }

    assert _prefer_geometry_over_text_boundary(text, geometry) is True


def test_geometry_does_not_win_when_it_cuts_text_or_is_too_wide() -> None:
    text = {
        "ok": True,
        "method": "text-boundary",
        "box": [367, 215, 1119, 1265],
        "originalHeight": 1440,
    }
    geometry = {
        "ok": True,
        "method": "opencv-perspective",
        "box": [0, 0, 1125, 1202],
        "originalHeight": 1440,
    }

    assert _prefer_geometry_over_text_boundary(text, geometry) is False


def test_partial_edge_guard_rejects_clipped_strip_but_keeps_wide_edge_document() -> None:
    clipped = {
        "box": [1137, 0, 1440, 1440],
        "originalWidth": 1440,
        "originalHeight": 1440,
        "wordCount": 29,
    }
    valid_edge_document = {
        "box": [2, 12, 260, 610],
        "originalWidth": 420,
        "originalHeight": 620,
        "wordCount": 20,
    }

    assert _partial_edge_document_reason(clipped)
    assert _partial_edge_document_reason(valid_edge_document) == ""


def test_text_boundary_backend_can_be_disabled(monkeypatch, tmp_path: Path) -> None:
    source = _receipt_scene(tmp_path / "receipt.jpg")
    _fake_tesseract(monkeypatch, {
        "conf": ["91"] * 6,
        "text": [f"word{i}" for i in range(6)],
        "left": [10, 20, 30, 40, 50, 60],
        "top": [10, 20, 30, 40, 50, 60],
        "width": [20] * 6,
        "height": [10] * 6,
    })

    crop = detect_document_crop(source, save=False, text_boundary_backend="none")

    assert crop["ok"] is True
    assert crop["method"] != "text-boundary"


def test_text_boundary_reports_unsupported_backend() -> None:
    from urirun_connector_smart_crop.core import _detect_text_boxes

    result = _detect_text_boxes(Image.new("RGB", (200, 200), (255, 255, 255)), backend="doctr")

    assert result["ok"] is False
    assert "unsupported" in result["reason"]
    assert set(result["supportedBackends"]) == {"auto", "paddleocr", "tesseract", "off"}


def test_paddleocr_backend_falls_back_when_unavailable(tmp_path: Path) -> None:
    # PaddleOCR is stubbed unavailable by the autouse fixture, so an explicit paddleocr
    # request must fall through to the geometric cascade rather than error.
    source = _receipt_scene(tmp_path / "receipt.jpg")

    crop = detect_document_crop(source, save=False, use_text_boundary=True, text_boundary_backend="paddleocr")

    assert crop["ok"] is True
    assert crop["method"] != "text-boundary"


def test_detect_document_crop_ignores_bright_background_band(tmp_path: Path) -> None:
    source = _bright_band_scene(tmp_path / "bright-band.jpg")

    crop = detect_document_crop(source)

    assert crop["ok"] is True
    assert crop["box"][1] > 100
    assert crop["box"][0] > 110
    assert crop["box"][2] < 340


def test_detect_document_crop_ignores_connected_bright_tape_and_crops_receipt(tmp_path: Path) -> None:
    image = Image.new("RGB", (480, 640), (104, 103, 92))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 180, 640), fill=(172, 170, 155))
    draw.rectangle((92, 0, 480, 64), fill=(190, 184, 158))
    draw.rectangle((140, 312, 352, 572), fill=(248, 246, 235))
    for y in (365, 392, 420, 452, 490, 532):
        draw.line((166, y, 322, y), fill=(24, 24, 24), width=5)
    draw.text((166, 330), "Polskie ePlatnosci", fill=(24, 24, 24))
    source = tmp_path / "receipt-with-tape.jpg"
    image.save(source)

    crop = detect_document_crop(source)

    assert crop["ok"] is True
    assert crop["method"] in {"text-region", "opencv-perspective", "fill-ratio", "connected-component"}
    assert crop["bboxArea"] < 0.45
    assert crop["box"][0] > 90
    assert crop["box"][1] > 250
    assert crop["cropWidth"] < 280
    assert crop["cropHeight"] < 340


def test_detect_document_crop_prefers_receipt_over_large_edge_fabric_strip(tmp_path: Path) -> None:
    image = Image.new("RGB", (720, 720), (104, 106, 96))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 300, 720), fill=(176, 174, 158))
    draw.rectangle((0, 0, 520, 58), fill=(188, 184, 164))
    for x in range(28, 292, 42):
        draw.line((x, 0, x + 24, 720), fill=(152, 151, 137), width=2)
    draw.rectangle((315, 325, 565, 638), fill=(250, 248, 236))
    draw.text((344, 372), "Polskie ePlatnosci", fill=(22, 22, 22))
    for y in (420, 444, 470, 502, 542, 586, 616):
        draw.line((344, y, 528, y), fill=(20, 20, 20), width=4)
    source = tmp_path / "receipt-edge-fabric-strip.jpg"
    image.save(source)

    crop = detect_document_crop(source, save=False)

    assert crop["ok"] is True
    assert crop["box"][0] > 250
    assert crop["box"][1] > 280
    assert crop["bboxArea"] < 0.35
    assert crop["method"] in {"fill-ratio", "text-region", "opencv-perspective", "connected-component"}


def test_detect_document_crop_rectifies_perspective_document(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    image = Image.new("RGB", (600, 420), (45, 47, 50))
    draw = ImageDraw.Draw(image)
    draw.polygon([(180, 40), (420, 75), (390, 370), (145, 330)], fill=(247, 245, 235), outline=(20, 20, 20))
    for y in (120, 170, 220, 270):
        draw.line((200, y, 360, y + 18), fill=(30, 30, 30), width=4)
    source = tmp_path / "perspective.jpg"
    image.save(source)

    crop = detect_document_crop(source)

    assert crop["ok"] is True
    assert crop["method"] == "opencv-perspective"
    assert len(crop["quad"]) == 4
    assert crop["box"][0] > 100
    assert crop["box"][2] < 460
    assert crop["width"] > 220
    assert crop["height"] > 280


def test_detect_document_crop_keeps_document_touching_frame_edge(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    image = Image.new("RGB", (420, 620), (38, 39, 42))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 12, 260, 610), fill=(248, 247, 240), outline=(20, 20, 20), width=3)
    for y in range(60, 560, 55):
        draw.line((28, y, 230, y), fill=(30, 30, 30), width=4)
    source = tmp_path / "edge-receipt.jpg"
    image.save(source)

    crop = detect_document_crop(source)

    assert crop["ok"] is True
    assert crop["method"] == "opencv-perspective"
    assert crop["box"][0] <= 5
    assert crop["cropHeight"] > 560
    assert crop["component"]["touchesEdge"] is True


def test_detect_document_crop_auto_orients_sideways_receipt_to_portrait(tmp_path: Path) -> None:
    image = Image.new("RGB", (360, 240), (45, 47, 50))
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 72, 310, 154), fill=(246, 244, 235))
    for y in (92, 116, 138):
        draw.line((82, y, 270, y), fill=(35, 35, 35), width=3)
    source = tmp_path / "sideways.jpg"
    image.save(source)

    crop = detect_document_crop(source)
    with Image.open(crop["path"]) as saved:
        width, height = saved.size

    assert crop["ok"] is True
    assert height > width
    assert crop["orientation"]["angle"] in {90, 270}
    assert crop["orientation"]["rotated"] is True


def test_orient_document_image_uses_osd_when_portrait_dimension_is_misleading(monkeypatch) -> None:
    # A nearly square crop can be geometrically "portrait" while its text is
    # still sideways. Projection scoring alone then prefers the wrong 0/180
    # candidates. OSD supplies the text-upright direction and must be allowed to
    # override the portrait-only pool.
    image = Image.new("RGB", (120, 124), (245, 245, 238))

    monkeypatch.setattr(
        "urirun_connector_smart_crop.core._line_orientation_score",
        lambda _image: {"score": 0.25, "darkFraction": 0.02, "rowPeak": 0.3, "colPeak": 0.05},
    )
    monkeypatch.setattr(
        "urirun_connector_smart_crop.core._tesseract_orientation_correction",
        lambda _image: {
            "ok": True,
            "rotate": 90,  # clockwise tesseract hint -> PIL angle 270
            "confidence": 2.0,
            "script": "Latin",
            "scriptConfidence": 0.8,
        },
    )

    oriented, meta = _orient_document_image(image, auto_orient=True, prefer_portrait=True)

    assert meta["angle"] == 270
    assert meta["osd"]["appliedAngle"] == 270
    assert oriented.size == (124, 120)


def test_orient_document_image_trusts_paddle_orientation(monkeypatch) -> None:
    # The trained orientation classifier is authoritative: when it picks an angle, the
    # geometric/OSD cascade is bypassed entirely. This is the fix for receipts that the
    # row/col heuristic wrongly laid on their side.
    image = Image.new("RGB", (124, 120), (245, 245, 238))

    monkeypatch.setattr(
        "urirun_connector_smart_crop.core._paddle_document_angle",
        lambda _image, **_kw: {"angle": 90, "score": 0.92},
    )
    # If the cascade below ran it would raise; it must not be reached.
    monkeypatch.setattr(
        "urirun_connector_smart_crop.core._line_orientation_score",
        lambda _image: (_ for _ in ()).throw(AssertionError("geometry must not run")),
    )

    oriented, meta = _orient_document_image(image, auto_orient=True, prefer_portrait=True)

    assert meta["source"] == "paddle-doc-orientation"
    assert meta["angle"] == 90 and meta["rotated"] is True
    assert meta["score"] == 0.92
    assert oriented.size == (120, 124)  # 124x120 rotated 90 -> 120x124


def test_document_crop_route_returns_original_when_uncertain(tmp_path: Path) -> None:
    source = tmp_path / "dark.jpg"
    Image.new("RGB", (160, 120), (20, 20, 20)).save(source)

    result = document_crop(image=str(source), fail_if_uncertain=False)

    assert result["ok"] is True
    assert result["path"] == str(source.resolve())
    assert result["crop"]["ok"] is False


def test_document_crop_route_can_fail_when_uncertain(tmp_path: Path) -> None:
    source = tmp_path / "dark.jpg"
    Image.new("RGB", (160, 120), (20, 20, 20)).save(source)

    result = document_crop(image=str(source), fail_if_uncertain=True)

    assert result["ok"] is False
    assert "document" in result["error"] or "component" in result["error"]


def test_document_detect_does_not_write_crop(tmp_path: Path) -> None:
    source = _receipt_scene(tmp_path / "receipt.jpg")

    result = document_detect(image=str(source))

    assert result["ok"] is True
    assert result["crop"]["ok"] is True
    assert result["crop"]["path"] == ""


def test_bindings_and_runtime() -> None:
    bindings = urirun_bindings()["bindings"]
    assert set(bindings) == ALL_ROUTES
    for route in ALL_ROUTES:
        assert bindings[route]["adapter"] == "local-function-subprocess"
        assert bindings[route]["python"]["module"] == "urirun_connector_smart_crop.core"
    json.dumps(urirun_bindings())


def test_runtime_executes_from_compiled_registry(tmp_path: Path) -> None:
    source = _receipt_scene(tmp_path / "receipt.jpg")
    registry = urirun.compile_registry(json.loads(json.dumps(urirun_bindings())))

    env = v2.run(
        ROUTE_CROP,
        registry,
        payload={"image": str(source), "output_dir": str(tmp_path)},
        mode="execute",
        policy=urirun.policy(allow=["smartcrop://*"]),
    )

    assert env["ok"] is True
    data = urirun.result_data(env)
    assert data["ok"] is True
    assert data["crop"]["ok"] is True
    assert Path(data["path"]).is_file()


def test_manifest() -> None:
    manifest = connector_manifest()
    assert manifest["id"] == "smart-crop"
    assert manifest["uriSchemes"] == ["smartcrop"]
    assert set(manifest["routes"]) == ALL_ROUTES
    json.dumps(manifest)


def test_contract_output_shape() -> None:
    """Contract examples and a cheap live error output must satisfy the declared schema."""
    from urirun_connectors_toolkit.contract_gate import conform, validate_output
    from urirun_connector_smart_crop.contracts import CONTRACTS

    conform(CONTRACTS)
    validate_output(CONTRACTS["document/query/detect"], document_detect(image=""))
    validate_output(CONTRACTS["document/query/crop"], document_crop(image=""))


def test_detect_document_crop_handles_receipt_with_brightness_gradient(tmp_path: Path) -> None:
    """The real QUO CAFE failure: a small receipt with a top-to-bottom brightness gradient
    (top whiter than bottom) next to a large light duct-tape 'L'. The fill-ratio solid-sheet
    detector must crop the WHOLE receipt (incl. the dimmer bottom with the amount), not the tape."""
    image = Image.new("RGB", (1000, 1400), (70, 75, 72))
    draw = ImageDraw.Draw(image)
    draw.rectangle((110, 70, 250, 900), fill=(190, 188, 182))     # vertical tape strip
    draw.rectangle((110, 70, 770, 210), fill=(190, 188, 182))     # horizontal tape strip
    # receipt with a vertical brightness gradient: top ~252 → bottom ~218
    for i, y in enumerate(range(980, 1330, 5)):
        v = int(252 - (y - 980) / 350.0 * 34)
        draw.rectangle((360, y, 720, y + 5), fill=(v, v, v - 2))
    for y in range(1010, 1300, 26):
        draw.line((385, y, 695, y), fill=(15, 15, 15), width=4)
    source = tmp_path / "receipt-gradient.jpg"
    image.save(source)

    crop = detect_document_crop(source, save=False)
    assert crop["ok"] is True
    x0, y0, x1, y1 = crop["box"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    assert 350 <= cx <= 730 and 960 <= cy <= 1350, f"crop centred on {cx},{cy} (distractor?)"
    assert (y1 - y0) > 250, "crop must span the full receipt height, not just the bright top"
    assert crop["bboxArea"] < 0.4


def test_text_boundary_crop_follows_detected_text(tmp_path: Path) -> None:
    """A real text document is cropped to the union of its text, via Tesseract."""
    import shutil

    import pytest

    pytest.importorskip("pytesseract")
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract binary not installed")
    from PIL import ImageFont

    image = Image.new("RGB", (760, 1040), (58, 60, 62))  # dark surface
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 170, 610, 880), fill=(249, 248, 241))  # white sheet
    try:
        font = ImageFont.load_default(size=34)
    except TypeError:  # very old Pillow without size arg
        font = ImageFont.load_default()
    lines = ["PARAGON FISKALNY", "SKLEP TESTOWY ABC", "RAZEM 42,00 PLN",
             "KARTA VISA", "DZIEKUJEMY ZAPRASZAMY"]
    y = 220
    for line in lines:
        draw.text((190, y), line, fill=(12, 12, 12), font=font)
        y += 120
    source = tmp_path / "textdoc.png"
    image.save(source)

    crop = detect_document_crop(source, save=False)

    assert crop["ok"] is True
    if crop["method"] != "text-boundary":
        pytest.skip(f"tesseract did not detect synthetic text (method={crop['method']})")
    # Cropped to the text/sheet region, not the whole dark frame.
    assert crop["box"][0] > 90 and crop["box"][1] > 90
    assert crop["bboxArea"] < 0.75
    assert crop["wordCount"] >= 6


def test_text_boundary_disabled_falls_back_to_geometry(tmp_path: Path) -> None:
    """With use_text_boundary=False the geometric cascade still produces a crop."""
    image = Image.new("RGB", (480, 640), (40, 42, 45))
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 160, 330, 470), fill=(247, 246, 236))
    for y in (200, 240, 280, 330, 380, 430):
        draw.line((175, y, 305, y), fill=(20, 20, 20), width=4)
    source = tmp_path / "geom.png"
    image.save(source)

    crop = detect_document_crop(source, save=False, use_text_boundary=False)

    assert crop["ok"] is True
    assert crop["method"] != "text-boundary"


def _fake_paddle_detector(monkeypatch, polys: list) -> None:
    """Stub the cached PaddleOCR engine with one returning fixed detection polygons.

    Overrides the autouse `unavailable` stub for this test only.
    """
    class _Det:
        def predict(self, _image):
            return [{"dt_polys": [list(p) for p in polys]}]

    monkeypatch.setattr(
        "urirun_connector_smart_crop.core._get_paddle_detector",
        lambda *a, **k: _Det(),
    )


def test_paddleocr_backend_crops_to_detected_text(monkeypatch, tmp_path: Path) -> None:
    """The PaddleOCR (line-polygon) path drives a text-boundary crop on the paper sheet."""
    image = Image.new("RGB", (760, 1040), (60, 62, 64))  # dark surface
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 170, 610, 880), fill=(248, 247, 240))  # white sheet
    for y in range(230, 740, 120):
        draw.rectangle((190, y, 420, y + 26), fill=(15, 15, 15))
    source = tmp_path / "paddle-doc.png"
    image.save(source)

    # Quad polygons (x,y per corner) as PP-OCRv6 returns them, one per text line.
    polys = [
        [[190, y], [420, y], [420, y + 26], [190, y + 26]]
        for y in range(230, 740, 120)
    ]
    _fake_paddle_detector(monkeypatch, polys)

    crop = detect_document_crop(source, save=False, text_boundary_backend="paddleocr")

    assert crop["ok"] is True
    assert crop["method"] == "text-boundary"
    assert crop["textBoundaryBackend"] == "paddleocr"
    assert crop["wordCount"] == len(polys)
    # Snaps out to the detected white-sheet edges, not the dark frame.
    assert crop["box"][0] >= 130 and crop["box"][2] <= 630
    assert crop["bboxArea"] < 0.6


def test_paddleocr_partial_edge_text_strip_is_rejected(monkeypatch, tmp_path: Path) -> None:
    image = Image.new("RGB", (1000, 1000), (92, 88, 82))
    draw = ImageDraw.Draw(image)
    draw.rectangle((850, 0, 999, 999), fill=(248, 247, 240))
    for y in range(60, 900, 70):
        draw.rectangle((880, y, 986, y + 24), fill=(18, 18, 18))
    source = tmp_path / "partial-edge-strip.png"
    image.save(source)

    polys = [
        [[880, y], [986, y], [986, y + 24], [880, y + 24]]
        for y in range(60, 900, 70)
    ]
    _fake_paddle_detector(monkeypatch, polys)

    crop = detect_document_crop(source, save=False, text_boundary_backend="paddleocr")

    assert crop["ok"] is False
    assert crop.get("partialEdge") is True
    assert "partial" in crop["reason"]


@pytest.mark.real_paddle
def test_paddleocr_backend_real_engine_detects_text(tmp_path: Path) -> None:
    """Integration: the actual PaddleOCR detector produces a text-boundary crop.

    Skips cleanly when the package or its model (first run downloads it) is unavailable.
    """
    pytest.importorskip("paddleocr")
    from PIL import ImageFont

    from urirun_connector_smart_crop.core import _get_paddle_detector

    try:
        _get_paddle_detector()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"paddleocr detector unavailable: {exc}")

    image = Image.new("RGB", (760, 1040), (58, 60, 62))
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 170, 610, 880), fill=(249, 248, 241))
    try:
        font = ImageFont.load_default(size=34)
    except TypeError:
        font = ImageFont.load_default()
    for i, line in enumerate(["PARAGON FISKALNY", "SKLEP TESTOWY ABC", "RAZEM 42,00 PLN",
                              "KARTA VISA", "DZIEKUJEMY ZAPRASZAMY"]):
        draw.text((190, 220 + i * 120), line, fill=(12, 12, 12), font=font)
    source = tmp_path / "real-paddle.png"
    image.save(source)

    crop = detect_document_crop(source, save=False, text_boundary_backend="paddleocr")

    if crop["method"] != "text-boundary":
        pytest.skip(f"paddle detected too little synthetic text (method={crop['method']})")
    assert crop["textBoundaryBackend"] == "paddleocr"
    assert crop["box"][0] > 90 and crop["box"][1] > 90
    assert crop["bboxArea"] < 0.75


def test_crop_route_tags_output_as_frozen_artifact(tmp_path: Path) -> None:
    source = _receipt_scene(tmp_path / "receipt.jpg")

    result = document_crop(image=str(source), output_dir=str(tmp_path))

    assert result["ok"] is True
    # Shared urirun.tag contract: a finished crop is a frozen artifact, never a live widget.
    assert result["kind"] == "document-crop"
    assert result["live"] is False


def test_detect_route_tags_output_as_frozen_artifact(tmp_path: Path) -> None:
    source = _receipt_scene(tmp_path / "receipt.jpg")

    result = document_detect(image=str(source))

    assert result["ok"] is True
    assert result["kind"] == "crop-detection"
    assert result["live"] is False


def test_auto_prefers_tesseract_when_confident(monkeypatch) -> None:
    """auto = tesseract-first: a confident Tesseract read wins and PaddleOCR is NOT invoked."""
    from urirun_connector_smart_crop import core
    _fake_tesseract(monkeypatch, {
        "conf": ["90"] * 6, "text": [f"w{i}" for i in range(6)],
        "left": [10, 20, 30, 40, 50, 60], "top": [10, 20, 30, 40, 50, 60],
        "width": [20] * 6, "height": [10] * 6,
    })
    called = {"paddle": False}

    def _no_paddle(*_a, **_k):
        called["paddle"] = True
        return {"ok": False, "backend": "paddleocr"}

    monkeypatch.setattr(core, "_paddleocr_text_boxes", _no_paddle)
    res = core._detect_text_boxes(Image.new("RGB", (200, 300), (255, 255, 255)), backend="auto")
    assert res["ok"] and res["backend"] == "tesseract"
    assert called["paddle"] is False  # fast path, no escalation


def test_auto_escalates_to_paddle_when_tesseract_uncertain(monkeypatch) -> None:
    """auto escalates to the trained detector only when Tesseract finds too few words."""
    from urirun_connector_smart_crop import core
    _fake_tesseract(monkeypatch, {
        "conf": ["90"] * 2, "text": ["a", "b"],
        "left": [10, 20], "top": [10, 20], "width": [20, 20], "height": [10, 10],
    })
    monkeypatch.setattr(core, "_paddleocr_text_boxes", lambda *_a, **_k: {
        "ok": True, "backend": "paddleocr", "boxes": [[1, 1, 9, 9]] * 3,
        "wordCount": 3, "lineHeight": 8.0, "clean": True,
    })
    res = core._detect_text_boxes(Image.new("RGB", (200, 300), (255, 255, 255)), backend="auto")
    assert res["ok"] and res["backend"] == "paddleocr"  # escalated
