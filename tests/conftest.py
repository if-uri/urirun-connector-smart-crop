from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _stub_paddle_detector(request, monkeypatch):
    """Disable the heavy PaddleOCR detector in unit tests.

    The trained detector is the default ``auto`` backend in production, but loading and
    running it in every unit test would be slow and non-deterministic on synthetic
    line-art fixtures. Stubbing it ``unavailable`` makes ``auto`` fall back to Tesseract
    (itself faked per-test) then the geometric cascade -- exactly the path these tests
    target. A test that needs the real engine opts back in with ``@pytest.mark.real_paddle``.
    """
    if request.node.get_closest_marker("real_paddle"):
        return

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("paddleocr detector stubbed off in unit tests")

    monkeypatch.setattr(
        "urirun_connector_smart_crop.core._get_paddle_detector",
        _unavailable,
    )
