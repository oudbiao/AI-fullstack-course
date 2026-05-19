# CH09 Vertical Image Refinement Handoff

Updated: 2026-05-19 Asia/Taipei

## Status
- Inventory complete; candidate generation and QA are pending.
- Scope is the actual Markdown image references under `ch09` docs and zh/ja i18n trees.
- Existing vertical WebP files are marked `keep`; existing landscape/square WebP files are marked `candidate` and must be regenerated as native vertical PNG candidates before conversion.

## Counts
- Official image refs: 345 unique WebP filenames.
- Existing non-vertical candidates: 153.
- Missing official files: 0.

## Files
- Inventory: `reports/course-images/ch09-vertical-refine/inventory.tsv`
- Candidate targets: `reports/course-images/ch09-vertical-refine/candidate-targets.txt`
- Candidate PNG directory: `tmp/ch09-vertical-images/`
- QA report target: `reports/course-images/ch09-vertical-refine/qa-report.md`

## Required QA Rules
- Candidate PNGs are not official assets until QA passes.
- Do not convert forced-stretched landscape images.
- Reject unreadable tiny/gibberish text, wrong-language text, fake numeric metrics, watermarks, brand logos, or decorative-only posters.
- Accepted PNGs must be converted to same-stem `.webp` files in `static/img/course/`; Markdown references remain `.webp`.
- After chapter conversion, run the project validators and build before committing.

## Next Step
Generate `ch09` candidate PNGs listed in `candidate-targets.txt`, QA them, write `qa-report.md`, convert accepted files to official WebP, then update this handoff to complete.
