# urirun-connector-smart-crop

Detect, crop and auto-orient a document, receipt or invoice from a larger image
before OCR.

This connector owns the reusable document-crop logic used by:

- phone scanner captures in the urirun host dashboard,
- OCR flows that need a stable crop before `ocr://...`,
- camera/photo pipelines that should preserve the original frame while OCR runs on
  the detected document.

## Routes

| URI | Purpose |
| --- | --- |
| `smartcrop://host/document/query/crop` | Detect a document/receipt in an image and write a cropped image when reliable. |
| `smartcrop://host/document/query/detect` | Detect only; return bounding box and confidence metadata without writing a crop. |

## Example

```bash
urirun run smartcrop://host/document/query/crop \
  --payload '{"image":"/tmp/photo.jpg","output_dir":"/tmp/crops"}'
```

The crop route defaults to `auto_orient=true` and `prefer_portrait=true`, so a
sideways receipt is saved as a portrait image with horizontal text lines. Orientation
is decided by a **trained classifier first** (PaddleOCR `PP-LCNet_x1_0_doc_ori`) and only
falls back to tesseract OSD / the geometric heuristic when the model is unavailable — the
geometric `rowPeak − colPeak` signal is unreliable on narrow receipts (it can rotate an
already-upright scan onto its side), so the classifier is authoritative when confident
(`crop.orientation.source == "paddle-doc-orientation"`).
Text-boundary cropping is enabled by default with `text_boundary_backend=auto`,
which tries fast **Tesseract** first (~0.5s) and escalates to the trained
**PaddleOCR** detector (~4s on CPU) only when Tesseract is uncertain (too few
confident words: faint/low-contrast/non-Latin text) — keeping the common case fast
while still recovering hard scans. Force a specific backend with
`text_boundary_backend=paddleocr` or `=tesseract`, or use `use_text_boundary=false`
/ `text_boundary_backend=none` to force the geometric cascade.

The result includes:

- `path`: cropped image path when detected, otherwise the original path,
- `originalPath`: original input path,
- `crop.ok`: whether a reliable crop was written,
- `crop.box`: `[left, top, right, bottom]` in original image pixels,
- `crop.orientation`: applied rotation `angle`, `rotated`, and `source`
  (`paddle-doc-orientation` | `osd` | `geometry`) plus candidate scores,
- `crop.reason`: why detection was skipped when not reliable.
- `crop.partialEdge`: true when the detector saw only a clipped fragment of a
  document at the frame edge; the caller should ask for another frame instead of
  saving a bad scan.

The detector intentionally prefers a conservative "no crop" over a wrong crop.
It first detects text-line boxes (fast Tesseract, escalating to the trained
PaddleOCR detector only when Tesseract is uncertain), then expands that boundary out to the paper component
detected from background contrast, per side. The residual margin is a fixed
multiple of the detected text-line height — a scale-invariant typographic unit,
not a hand-tuned fraction of the frame — so the same logic holds for a small
receipt and a full A4 scan. This keeps barcodes, logos and footer text from being
cut off. If the document already fills the frame, it keeps the frame instead of
running a destructive second crop.

Paper-background expansion is side-aware. When a background component touches the
camera frame, only a small extension is allowed on that side; this prevents a
receipt merged with the table/background from pulling the crop up to `y=0` while
still keeping real frame-filling documents intact.

Candidate detectors run in probe mode first and write a crop only after the final
winner passes validation. This prevents a rejected fallback from leaving a stale
or overly tight `*-receipt-crop.jpg` on disk.

Backends are selectable behind `text_boundary_backend` (`auto`, `paddleocr`,
`tesseract`, `none`/`off`). Unsupported values fall back to geometry with metadata
rather than failing the whole crop. PaddleOCR runs detection only (no recognition
or orientation models), so it is language-agnostic; install it with the
`text-boundary-paddle` extra.

When OCR boxes are not available, the connector falls back to the geometric
cascade: OpenCV document quadrilateral/perspective crop, bright-sheet fill-ratio,
text-region and connected-component thresholds. Candidates are scored using
contour shape, text/edge density, and penalties for false positives such as
bright horizontal background bands.
