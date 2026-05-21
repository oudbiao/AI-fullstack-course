# Chapter 7 Embedding Image Sync QA

Date: 2026-05-21

## Scope

- `ch07-embedding-cosine-retrieval-context-result-map-en.png`
- `ch07-embedding-cosine-retrieval-context-result-map.png`
- `ch07-embedding-cosine-retrieval-context-result-map-ja.png`

## Generation

- Source prompt group: `scripts/generate_course_images.py`
- Temporary review directory: `tmp/ch07-embedding-sync-20260521/`
- Official asset directory: `static/img/course/`
- Generation errors: `[]`

## QA

- All three candidate PNG files are `1024x1792`.
- OCR and visual review confirmed the retrieval ranking now uses:
  - `B password reset 1.000`
  - `C coupon return 0.335`
  - `A refund policy 0.333`
- No `banana` or fruit example remains in the accepted candidate images.
- Visual review confirmed readable layout, no obvious fake metrics, no watermark, and no stretched text.
- Accepted candidates were converted to official WebP assets:
  - `static/img/course/ch07-embedding-cosine-retrieval-context-result-map-en.webp`
  - `static/img/course/ch07-embedding-cosine-retrieval-context-result-map.webp`
  - `static/img/course/ch07-embedding-cosine-retrieval-context-result-map-ja.webp`

## Result

Accepted.
