# Ch08 SOP Assistant Image Sync QA

Date: 2026-05-21

## Scope

Replaced the project-page images for the former courseware assistant page while keeping the existing slugs and asset filenames:

- `courseware-assistant-workflow-en.webp`
- `courseware-assistant-workflow.webp`
- `courseware-assistant-workflow-ja.webp`
- `ch08-courseware-assistant-production-line-map-en.webp`
- `ch08-courseware-assistant-production-line-map.webp`
- `ch08-courseware-assistant-production-line-map-ja.webp`

The new visuals show a support-operations SOP document assistant: incident topic input, policy-document retrieval, evidence extraction, SOP schema, Word template export, trace, and evaluation.

## Generation

Generated candidate PNGs in:

- `tmp/ch08-sop-assistant-sync-20260521/`

All accepted candidates are `1024x1792`.

## QA Notes

- Visual contact sheet reviewed manually.
- All six candidates use SOP/support-operations/Word-template scenes.
- No school, classroom, math worksheet, discount problem, or old courseware visual motif was observed.
- English OCR scan found no old-context terms: `courseware`, `lesson`, `discount`, `classroom`, `school`, `student`, `worksheet`, `math`, `blackboard`.
- Local OCR only had the English language pack available; Chinese and Japanese candidates were checked visually for the same old-context residues.

## Promotion

Accepted PNGs were converted with `cwebp -q 90` and promoted to:

- `static/img/course/courseware-assistant-workflow-en.webp`
- `static/img/course/courseware-assistant-workflow.webp`
- `static/img/course/courseware-assistant-workflow-ja.webp`
- `static/img/course/ch08-courseware-assistant-production-line-map-en.webp`
- `static/img/course/ch08-courseware-assistant-production-line-map.webp`
- `static/img/course/ch08-courseware-assistant-production-line-map-ja.webp`
