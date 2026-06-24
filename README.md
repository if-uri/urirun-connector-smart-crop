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
sideways receipt is saved as a portrait image with horizontal text lines.
Text-boundary cropping is enabled by default with `text_boundary_backend=auto`
(`tesseract` today). Use `use_text_boundary=false` or
`text_boundary_backend=none` to force the geometric cascade.

The result includes:

- `path`: cropped image path when detected, otherwise the original path,
- `originalPath`: original input path,
- `crop.ok`: whether a reliable crop was written,
- `crop.box`: `[left, top, right, bottom]` in original image pixels,
- `crop.orientation`: applied rotation angle and candidate scores,
- `crop.reason`: why detection was skipped when not reliable.

The detector intentionally prefers a conservative "no crop" over a wrong crop.
When Tesseract is available, it first uses OCR word boxes as a text boundary,
then expands that boundary to the paper component detected from background
contrast. This keeps receipt barcodes, logos and low-confidence footer text from
being cut off. If the document already fills the frame, it keeps the frame
instead of running a destructive second crop.

The text-boundary backend is intentionally explicit. Unsupported values such as
`paddleocr` or `doctr` fall back to geometry today; those backends can be added
behind the same parameter later without changing URI flows.

When OCR boxes are not available, the connector falls back to the geometric
cascade: OpenCV document quadrilateral/perspective crop, bright-sheet fill-ratio,
text-region and connected-component thresholds. Candidates are scored using
contour shape, text/edge density, and penalties for false positives such as
bright horizontal background bands.
