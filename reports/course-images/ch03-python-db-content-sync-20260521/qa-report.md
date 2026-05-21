# ch03 Python Database Content-Image Sync QA Report

Date: 2026-05-21

## Scope

- Content changes:
  - `docs/ch03-data-analysis/ch05-database/03-python-db.md`
  - Matching `zh-Hans` and `ja` localized pages
- Image changes:
  - `static/img/course/ch03-python-database-safety-vertical*.webp`

## Accepted PNG Candidates

| Locale | Source PNG | Official WebP |
|---|---|---|
| zh-Hans | `tmp/ch03-python-db-content-sync-20260521-final2/ch03-python-database-safety-vertical.png` | `static/img/course/ch03-python-database-safety-vertical.webp` |
| en | `tmp/ch03-python-db-content-sync-20260521-final2/ch03-python-database-safety-vertical-en.png` | `static/img/course/ch03-python-database-safety-vertical-en.webp` |
| ja | `tmp/ch03-python-db-content-sync-20260521-final2/ch03-python-database-safety-vertical-ja.png` | `static/img/course/ch03-python-database-safety-vertical-ja.webp` |

All accepted images are `1024x1792` vertical assets and were converted to same-stem WebP files. Markdown references remain WebP-only.

## QA Decisions

- The final images were rendered locally with `scripts/render_ch03_python_db_images.py` so the schema text and field names stay exact.
- All three candidates passed visual QA:
  - readable vertical recomposition, not a stretched landscape image
  - no garbled small text
  - no invented metrics, fake outputs, or wrong-language residue
  - matching three-language structure for the same lesson
- The visual flow matches the updated page:
  - connect and model
  - safe query
  - hand off to Pandas
  - write back
- The footer evidence card matches the page's `tickets` / `TicketDB` workflow and keeps the teaching evidence compact.

## Follow-Up Notes

- If the page changes again, regenerate the three PNGs with `scripts/render_ch03_python_db_images.py`, QA them, and reconvert the same-stem WebP assets.
- The existing bridge image was kept because it still provides a valid contextual overview for the updated lesson.
