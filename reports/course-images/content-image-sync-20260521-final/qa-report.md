# ch02/ch03 Content-Image Sync QA Report

Date: 2026-05-21

## Scope

- Content changes:
  - `ch02-python/ch03-projects/04-ai-api-experience.md`
  - `ch03-data-analysis/ch05-database/04-db-design.md`
  - Matching `zh-Hans` and `ja` localized pages
- Image changes:
  - `ch02-ai-api-request-response*.webp`
  - `ch03-database-design-erd-normalization*.webp`

## Accepted PNG Candidates

| Locale | Source PNG | Official WebP |
|---|---|---|
| zh-Hans | `tmp/content-image-sync-20260521-v2/ch02-ai-api-request-response.png` | `static/img/course/ch02-ai-api-request-response.webp` |
| en | `tmp/content-image-sync-20260521-v2/ch02-ai-api-request-response-en.png` | `static/img/course/ch02-ai-api-request-response-en.webp` |
| ja | `tmp/content-image-sync-20260521-v3/ch02-ai-api-request-response-ja.png` | `static/img/course/ch02-ai-api-request-response-ja.webp` |
| zh-Hans | `tmp/content-image-sync-20260521-final/ch03-database-design-erd-normalization.png` | `static/img/course/ch03-database-design-erd-normalization.webp` |
| en | `tmp/content-image-sync-20260521-final/ch03-database-design-erd-normalization-en.png` | `static/img/course/ch03-database-design-erd-normalization-en.webp` |
| ja | `tmp/content-image-sync-20260521-final/ch03-database-design-erd-normalization-ja.png` | `static/img/course/ch03-database-design-erd-normalization-ja.webp` |

All accepted images are `1024x1792` vertical assets and were converted to same-stem WebP files. Markdown references remain WebP-only.

## QA Decisions

### ch02 API request/response image

- v1 candidates were rejected for concrete token/cost-like numbers and brand/logo-like UI signals.
- v2 zh-Hans and English candidates passed:
  - real vertical composition, not stretched landscape
  - role selector matches the updated engineering roles
  - no invented usage metrics, token counts, costs, or fake outputs
  - no unreadable tiny filler text
- v2 Japanese candidate was rejected because the top text mark/title area was cropped.
- v3 Japanese candidate passed:
  - title and labels are readable
  - Japanese content matches the updated engineering-role exercise
  - no fake metrics or concrete API usage values

### ch03 database design image

- Generated candidates were rejected for text reliability issues, including field-name typos, concrete-looking IDs/numbers, or language/layout drift.
- Final ch03 images were rendered locally with `scripts/render_ch03_db_design_images.py` because table and field names must be exact teaching content.
- Final deterministic render passed:
  - field names are exact: `customer_id`, `assignee_id`, `ticket_messages`, `ticket_tags`, `category_id`, `tag_id`
  - no invented real-looking names, dates, prices, metrics, or dashboard numbers
  - zh-Hans/Japanese explanatory text is localized; English table names and schema fields are retained as technical identifiers
  - layout is vertical, readable, and not compressed
  - relationship lines do not hide table field text

## Rejection Notes

- `tmp/content-image-sync-20260521/`: rejected as a batch because the ch02 images included concrete usage/brand-like signals and the ch03 images contained too much generated microtext and concrete-looking values.
- `tmp/content-image-sync-20260521-v2/ch02-ai-api-request-response-ja.png`: rejected due to cropped/stray top text.
- `tmp/content-image-sync-20260521-v2/ch03-database-design-erd-normalization*.png` and `tmp/content-image-sync-20260521-v3/ch03-database-design-erd-normalization*.png`: not promoted; deterministic local rendering was chosen for exact schema text.

## Follow-Up Notes

- `scripts/generate_course_images.py` now has tighter prompts for these image keys so future regeneration avoids fake metrics, logo-like marks, and table/field typos.
- If future content changes alter schema fields or exercise requirements, update `scripts/render_ch03_db_design_images.py`, regenerate the three PNGs, QA them, then reconvert the same-stem WebP assets.
