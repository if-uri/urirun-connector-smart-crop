from __future__ import annotations

import json
from pathlib import Path

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

ROUTE_CROP = "smartcrop://host/document/query/crop"
ROUTE_DETECT = "smartcrop://host/document/query/detect"
ALL_ROUTES = {ROUTE_CROP, ROUTE_DETECT}


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


def test_detect_document_crop_ignores_bright_background_band(tmp_path: Path) -> None:
    source = _bright_band_scene(tmp_path / "bright-band.jpg")

    crop = detect_document_crop(source)

    assert crop["ok"] is True
    assert crop["box"][1] > 100
    assert crop["box"][0] > 110
    assert crop["box"][2] < 340


def test_detect_document_crop_rectifies_perspective_document(tmp_path: Path) -> None:
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
