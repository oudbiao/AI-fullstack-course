# CH08 Vertical Image Refinement Handoff

Updated: 2026-05-19 Asia/Taipei

## Status
- Inventory complete; candidate generation and QA are pending.
- Scope is the actual Markdown image references under `ch08` docs and zh/ja i18n trees.
- Existing vertical WebP files are marked `keep`; existing landscape/square WebP files are marked `candidate` and must be regenerated as native vertical PNG candidates before conversion.

## Counts
- Official image refs: 237 unique WebP filenames.
- Existing non-vertical candidates: 123.
- Missing official files: 0.

## Files
- Inventory: `reports/course-images/ch08-vertical-refine/inventory.tsv`
- Candidate targets: `reports/course-images/ch08-vertical-refine/candidate-targets.txt`
- Candidate PNG directory: `tmp/ch08-vertical-images/`
- QA report target: `reports/course-images/ch08-vertical-refine/qa-report.md`

## Required QA Rules
- Candidate PNGs are not official assets until QA passes.
- Do not convert forced-stretched landscape images.
- Reject unreadable tiny/gibberish text, wrong-language text, fake numeric metrics, watermarks, brand logos, or decorative-only posters.
- Accepted PNGs must be converted to same-stem `.webp` files in `static/img/course/`; Markdown references remain `.webp`.
- After chapter conversion, run the project validators and build before committing.

## Next Step
Generate `ch08` candidate PNGs listed in `candidate-targets.txt`, QA them, write `qa-report.md`, convert accepted files to official WebP, then update this handoff to complete.

## Progress Log
- 2026-05-19: ch07 vertical image refinement was committed as `cc16ef5e` (`Refine ch07 vertical course images`). ch08 scope confirmed at 123 candidates, split into 41 zh/base, 41 en, and 41 ja targets. Candidate generation is starting with the zh/base list in `tmp/ch08-vertical-images/`; no ch08 official WebP files have been replaced yet.
- 2026-05-19: ch08 zh/base candidate generation was paused to address localized title/sidebar English residue reported by the user. `tmp/ch08-vertical-images/` currently contains 19 zh/base PNG candidates; no ch08 official WebP files have been replaced. Resume with the same `generate_course_images.py --force-vertical ... --only $(tr '\n' ' ' < reports/course-images/ch08-vertical-refine/candidate-targets-zh.txt)` command after zh-Hans/ja text QA fixes; existing valid PNGs should be skipped by default.
- 2026-05-19: zh-Hans/ja localized sidebar taxonomy QA fix completed for ch06-ch09 suffixed labels. Added translations for the new `(Core)`, `(Extension)`, `(Deep Dive)`, `(Core Evidence)`, `(Advanced / Elective)`, and related labels in both locale `current.json` files. Also localized visible zh/ja table rows such as "Advanced RAG", "Advanced planning", and "Electives" where they appeared as learner-facing list labels. Follow-up scans found 0 missing zh-Hans/ja sidebar translations for these labels and 0 visible localized message/title hits for the English taxonomy residue.
