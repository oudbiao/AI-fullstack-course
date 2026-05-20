# CH09 Vertical Image QA Report

Updated: 2026-05-20 Asia/Taipei

## Scope
- Chapter: `ch09` official Markdown image references across zh/base, en, and ja docs.
- Candidate directory: `tmp/ch09-vertical-images/`
- Official output directory: `static/img/course/`
- Candidate targets: 153 PNGs, split into 51 zh/base, 51 en, and 51 ja files.

## QA Checks
- All 153 candidate PNGs exist.
- All 153 candidate PNGs are native `1024x1792` vertical images.
- Contact-sheet review covered the full candidate set.
- OCR risk scans were reviewed at:
  - `tmp/ch09-ocr-risk-hits.json`
  - `tmp/ch09-regenerated-ocr-risk-hits.json`
  - `tmp/ch09-full-postregen-ocr-risk-hits.json`
- High-risk candidates were also checked full-size after regeneration.
- Visual QA confirmed no forced landscape stretch, squeezed text, wrong-language labels, watermarks, real brand logos, decorative-only posters, or unreadable gibberish text in accepted candidates.
- Content QA confirmed no invented secrets, timestamps, ids, fake citations, fake percentages, or other synthetic metrics in accepted high-risk candidates.

## Rejections And Regeneration
The initial QA pass rejected a small high-risk subset before final promotion:
- security prompt-injection map: fake secrets / sensitive-looking values
- persistence checkpoint/event-log maps: concrete timestamps and ids
- code-agent sandbox review maps: overly concrete review output
- data-analysis notebook-loop maps: invented notebook / table values
- memory chapter / lifecycle maps in ja: concrete dates and importance values
- multi-agent communication contract map in ja: overly concrete trace data
- project delivery loop en: fake business/task values
- reasoning-eval failure taxonomy maps: invented metrics and labels
- research-assistant citation trace maps: fictional citation details

Those stems were regenerated with stricter prompt constraints and then rechecked.

## Accepted Candidate Set
All 153 listed PNGs are accepted for conversion to same-stem WebP files.

Accepted candidate lists are tracked in:
- `reports/course-images/ch09-vertical-refine/candidate-targets-zh.txt`
- `reports/course-images/ch09-vertical-refine/candidate-targets-en.txt`
- `reports/course-images/ch09-vertical-refine/candidate-targets-ja.txt`

## Conversion Status
- QA status: passed.
- Official WebP conversion: completed for all 153 accepted PNGs.
- Verification: all promoted `static/img/course/ch09-*.webp` files now match the accepted candidate stems and remain `1024x1792`.
- Markdown image references remain `.webp`; no Markdown image links were changed to PNG.
