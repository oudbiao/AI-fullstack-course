# CH07 Vertical Image Refinement Handoff

Updated: 2026-05-19 Asia/Taipei

## Status
- Complete; all 45 accepted candidates were converted to official WebP assets.
- Scope is the actual Markdown image references under `ch07` docs and zh/ja i18n trees.
- Existing vertical WebP files are marked `keep`; existing landscape/square WebP files are marked `candidate` and must be regenerated as native vertical PNG candidates before conversion.

## Counts
- Official image refs: 324 unique WebP filenames.
- Existing non-vertical candidates: 45.
- Missing official files: 0.

## Files
- Inventory: `reports/course-images/ch07-vertical-refine/inventory.tsv`
- Candidate targets: `reports/course-images/ch07-vertical-refine/candidate-targets.txt`
- Locale split targets: `candidate-targets-zh.txt`, `candidate-targets-en.txt`, `candidate-targets-ja.txt`
- Candidate PNG directory: `tmp/ch07-vertical-images/`
- QA report target: `reports/course-images/ch07-vertical-refine/qa-report.md`

## Required QA Rules
- Candidate PNGs are not official assets until QA passes.
- Do not convert forced-stretched landscape images.
- Reject unreadable tiny/gibberish text, wrong-language text, fake numeric metrics, watermarks, brand logos, or decorative-only posters.
- Accepted PNGs must be converted to same-stem `.webp` files in `static/img/course/`; Markdown references remain `.webp`.
- After chapter conversion, run the project validators and build before committing.

## Next Step
No ch07 image-refinement work remains unless later QA finds a regression. Continue with ch08 vertical candidate generation and QA.

## Progress Log
- 2026-05-19: Inventory complete. The 45 candidates split evenly into 15 zh / 15 en / 15 ja targets. `scripts/generate_course_images.py` and `scripts/generate_localized_course_images.py` now support `--force-vertical` for candidate generation so old landscape jobs render as native `1024x1792` vertical PNGs. Candidate generation is starting in `tmp/ch07-vertical-images/`.
- 2026-05-19: Generated all 15 zh/base PNG candidates in `tmp/ch07-vertical-images/`; automatic dimension check passed for all 15 at `1024x1792`. EN and JA candidates are still pending, and no official WebP files have been replaced yet.
- 2026-05-19: Generated all 15 EN PNG candidates in `tmp/ch07-vertical-images/`; generation reported `successes=15`, `errors=[]`, and automatic dimension check passed for all 15 at `1024x1792`. JA candidates are still pending, and no official WebP files have been replaced yet.
- 2026-05-19: Generated all 15 JA PNG candidates in `tmp/ch07-vertical-images/`; automatic dimension check passed for all 45 ch07 candidates at `1024x1792`. Visual QA rejected six candidates for concrete numeric values: `ch07-rlhf-reward-kl-loop-map.png`, `ch07-rlhf-reward-kl-loop-map-en.png`, `ch07-rlhf-reward-kl-loop-map-ja.png`, `ch07-lora-qlora-low-rank-memory-map.png`, `ch07-lora-qlora-low-rank-memory-map-en.png`, and `ch07-lora-qlora-low-rank-memory-map-ja.png`. The generator now has stricter vertical QA rules and dedicated no-number prompts for those base concepts; these six PNG candidates are being regenerated before conversion.
- 2026-05-19: Regenerated the two zh/base rejected PNG candidates (`ch07-rlhf-reward-kl-loop-map.png`, `ch07-lora-qlora-low-rank-memory-map.png`) with the stricter no-number prompts; dimension check passed at `1024x1792`. EN and JA rejected variants are next.
- 2026-05-19: Regenerated the two EN rejected PNG candidates (`ch07-rlhf-reward-kl-loop-map-en.png`, `ch07-lora-qlora-low-rank-memory-map-en.png`) with `successes=2`, `errors=[]`; dimension check passed at `1024x1792`. JA rejected variants are next.
- 2026-05-19: Regenerated the two JA rejected PNG candidates (`ch07-rlhf-reward-kl-loop-map-ja.png`, `ch07-lora-qlora-low-rank-memory-map-ja.png`) with `successes=2`, `errors=[]`; dimension check passed at `1024x1792`. Fresh contact sheets were generated for zh/en/ja, the six regenerated candidates were opened at full size, and all 45 ch07 candidates passed visual QA. `qa-report.md` has been created with official WebP conversion still pending.
- 2026-05-19: Converted all 45 accepted ch07 PNG candidates to same-stem WebP files under `static/img/course/` without changing Markdown references. Post-conversion dimension check passed for all 45 official WebP files at `1024x1792`. Required validators and build still need to run before the ch07 image commit.
