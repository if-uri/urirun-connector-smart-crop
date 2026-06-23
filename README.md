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

The result includes:

- `path`: cropped image path when detected, otherwise the original path,
- `originalPath`: original input path,
- `crop.ok`: whether a reliable crop was written,
- `crop.box`: `[left, top, right, bottom]` in original image pixels,
- `crop.orientation`: applied rotation angle and candidate scores,
- `crop.reason`: why detection was skipped when not reliable.

The detector intentionally prefers a conservative "no crop" over a wrong crop.
It uses Pillow only: bright low-saturation connected components are scored as
document candidates, edge-touching regions and wide background bands are rejected,
and the selected component is padded slightly before saving.
