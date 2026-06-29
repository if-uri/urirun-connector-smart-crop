# Author: Tom Sapletta · https://tom.sapletta.com
# Part of the ifURI solution.
"""Route contracts for the smart-crop connector — document boundary detection, read-only."""
from __future__ import annotations

from urirun_connectors_toolkit.contract_gate import Contract

_CROP_RESULT = {
    "ok": "bool",
    "connector": "const:smart-crop",
    "image": "?str",
    "crop": "?obj",
    "path": "?str",
    "originalPath": "?str",
    "kind": "?str",
    "live": "?bool",
    "error": "?str",
}

CONTRACTS: dict[str, Contract] = {
    "document/query/detect": Contract(
        version="v1",
        effect="query",
        reversible=False,
        inp={"image": "str", "auto_orient": "?bool", "prefer_portrait": "?bool"},
        out=_CROP_RESULT,
        errors=("precondition-unmet",),
        examples=(
            {
                "payload": {"image": "/tmp/scan.jpg"},
                "result": {
                    "ok": True,
                    "connector": "smart-crop",
                    "image": "/tmp/scan.jpg",
                    "crop": {"ok": True, "box": [10, 10, 800, 600], "method": "paddleocr"},
                    "kind": "crop-detection",
                    "live": False,
                },
            },
        ),
    ),
    "document/query/crop": Contract(
        version="v1",
        effect="query",
        reversible=False,
        inp={"image": "str", "output_path": "?str", "output_dir": "?str"},
        out=_CROP_RESULT,
        errors=("precondition-unmet",),
        examples=(
            {
                "payload": {"image": "/tmp/scan.jpg"},
                "result": {
                    "ok": True,
                    "connector": "smart-crop",
                    "image": "/tmp/scan.jpg",
                    "path": "/tmp/scan_cropped.jpg",
                    "originalPath": "/tmp/scan.jpg",
                    "crop": {"ok": True, "box": [10, 10, 800, 600], "method": "paddleocr"},
                    "kind": "document-crop",
                    "live": False,
                },
            },
        ),
    ),
}
