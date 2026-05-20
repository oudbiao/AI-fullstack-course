# CH08 Vertical Image Refinement Handoff

Updated: 2026-05-19 Asia/Taipei

## Status
- Complete and committed as `ce710626` (`Refine ch08 vertical course images`).
- Scope was the actual Markdown image references under `ch08` docs and zh/ja i18n trees.
- All 123 non-vertical WebP candidates were regenerated as native vertical PNG candidates, QA'd, converted to same-stem official WebP assets, validated, built, and committed.

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
No ch08 image work remains. Continue with ch09 candidate generation, QA, conversion, validation/build, and commit.

## Progress Log
- 2026-05-19: ch07 vertical image refinement was committed as `cc16ef5e` (`Refine ch07 vertical course images`). ch08 scope confirmed at 123 candidates, split into 41 zh/base, 41 en, and 41 ja targets. Candidate generation is starting with the zh/base list in `tmp/ch08-vertical-images/`; no ch08 official WebP files have been replaced yet.
- 2026-05-19: ch08 zh/base candidate generation was paused to address localized title/sidebar English residue reported by the user. `tmp/ch08-vertical-images/` currently contains 19 zh/base PNG candidates; no ch08 official WebP files have been replaced. Resume with the same `generate_course_images.py --force-vertical ... --only $(tr '\n' ' ' < reports/course-images/ch08-vertical-refine/candidate-targets-zh.txt)` command after zh-Hans/ja text QA fixes; existing valid PNGs should be skipped by default.
- 2026-05-19: zh-Hans/ja localized sidebar taxonomy QA fix completed for ch06-ch09 suffixed labels. Added translations for the new `(Core)`, `(Extension)`, `(Deep Dive)`, `(Core Evidence)`, `(Advanced / Elective)`, and related labels in both locale `current.json` files. Also localized visible zh/ja table rows such as "Advanced RAG", "Advanced planning", and "Electives" where they appeared as learner-facing list labels. Follow-up scans found 0 missing zh-Hans/ja sidebar translations for these labels and 0 visible localized message/title hits for the English taxonomy residue.
- 2026-05-19: User clarified the English residue examples were not limited to one visible title. A broader rendered-content scan across zh-Hans and ja found 0 learner-facing `current.json` message hits and 0 Markdown heading/front matter/list/table hits for `(Core)`, `(Extension)`, `(Deep Dive)`, `(Core Evidence)`, `Advanced / Elective`, `Advanced RAG`, `Advanced planning`, `Electives`, and related taxonomy labels.
- 2026-05-19: ch08 zh/base candidate generation completed. All 41 zh/base PNGs listed in `candidate-targets-zh.txt` exist in `tmp/ch08-vertical-images/` and passed automatic dimension checks at `1024x1792`. These are still candidates only: no visual QA report has been written yet and no official ch08 WebP files have been replaced.
- 2026-05-19: ch08 English candidate generation completed with `generate_localized_course_images.py --locale en --force-vertical`. All 41 PNGs listed in `candidate-targets-en.txt` exist in `tmp/ch08-vertical-images/` and passed automatic dimension checks at `1024x1792`. These are still candidates only: no visual QA report has been written yet and no official ch08 WebP files have been replaced.
- 2026-05-19: ch08 Japanese candidate generation completed with `generate_localized_course_images.py --locale ja --force-vertical`. All 41 PNGs listed in `candidate-targets-ja.txt` exist in `tmp/ch08-vertical-images/` and passed automatic dimension checks at `1024x1792`. A combined automatic check now reports zh/en/ja `targets=41 missing=0 bad_dimensions=0` for each locale, `total_pngs=123`. These are still candidates only: visual QA/contact sheets and official WebP replacement remain pending.
- 2026-05-19: Visual QA found and regenerated high-risk candidates before official adoption. Initial `function-calling-workflow-en.png` and `function-calling-workflow-ja.png` invented concrete tool outputs, so the source prompt was hardened to allow only abstract placeholders and all three locale variants were regenerated at `1024x1792`. Initial Docker compose candidates used recognizable brand-style icons, so the prompt was hardened to forbid Docker whale/Kubernetes/Redis logos and all three locale variants were regenerated at `1024x1792`. Initial `ch08-huggingface-ecosystem-layers-map-ja.png` used a brand-like HuggingFace mark, so the prompt was hardened to forbid official HuggingFace emoji/logo/mascot and the Japanese variant was regenerated at `1024x1792`. These regenerated PNGs still need full final QA before conversion to official WebP.
- 2026-05-19: Final visual QA found two more high-risk patterns and rejected/regenerated the full zh/en/ja sets for those concepts. `ch08-dialog-state-slot-memory-map-en.png` contained a concrete fake invoice number, so the prompt now forbids invoice/order/city/name/address/money/date/product placeholders and uses only abstract conversation labels. `ch08-study-guide-four-layer-map-en.png` contained a fake `92%` metric and Docker/Kubernetes-style icons, so the prompt now forbids concrete percentages/scores/cost/latency/ports/versions plus real brand-style deployment icons. All six regenerated files passed full-size visual QA and the combined candidate check remains `targets=123 unique=123 missing=0 bad_dimensions=0`.
- 2026-05-19: ch08 vertical image QA report was written to `reports/course-images/ch08-vertical-refine/qa-report.md`. All 123 accepted PNG candidates were converted to same-stem official WebP files in `static/img/course/`; Markdown references remain `.webp`. Post-conversion dimension check reports `official_webps=123 missing=0 bad_dimensions=0`. Required validators, `git diff --check`, and `npm run build` passed, then the work was committed as `ce710626`.
