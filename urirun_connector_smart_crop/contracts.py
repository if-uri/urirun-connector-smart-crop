# Author: Tom Sapletta · https://tom.sapletta.com
# Part of the ifURI solution.
"""Route contracts for the smart-crop connector — document boundary detection, read-only."""
from __future__ import annotations

from urirun_connectors_toolkit.contract_gate import Contract

_BOX = {"x": "int", "y": "int", "w": "int", "h": "int"}

CONTRACTS: dict[str, Contract] = {
    "document/query/detect": Contract(
        version="v1",
        effect="query",
        reversible=False,
        inp={"image": "str", "auto_orient": "?bool", "prefer_portrait": "?bool"},
        out={"ok": "bool", "box": "?obj", "confidence": "?num", "method": "?str"},
        errors=("precondition-unmet",),
        examples=(
            {
                "payload": {"image": "/tmp/scan.jpg"},
                "result": {
                    "ok": True,
                    "connector": "smart-crop",
                    "box": {"x": 10, "y": 10, "w": 800, "h": 600},
                    "confidence": 0.92,
                    "method": "paddleocr",
                },
            },
        ),
    ),
    "document/query/crop": Contract(
        version="v1",
        effect="query",
        reversible=False,
        inp={"image": "str", "output_path": "?str", "output_dir": "?str"},
        out={"ok": "bool", "path": "?str", "box": "?obj", "method": "?str"},
        errors=("precondition-unmet",),
        examples=(
            {
                "payload": {"image": "/tmp/scan.jpg"},
                "result": {
                    "ok": True,
                    "connector": "smart-crop",
                    "path": "/tmp/scan_cropped.jpg",
                    "box": {"x": 10, "y": 10, "w": 800, "h": 600},
                    "method": "paddleocr",
                },
            },
        ),
    ),
}
