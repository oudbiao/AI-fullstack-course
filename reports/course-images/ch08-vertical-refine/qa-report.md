# CH08 Vertical Image QA Report

Updated: 2026-05-19 Asia/Taipei

## Scope
- Chapter: `ch08` official Markdown image references across zh/base, en, and ja docs.
- Candidate directory: `tmp/ch08-vertical-images/`
- Official output directory: `static/img/course/`
- Candidate targets: 123 PNGs, split into 41 zh/base, 41 en, and 41 ja files.

## QA Checks
- All 123 candidate PNGs exist.
- All 123 candidate PNGs are native `1024x1792` vertical images.
- Contact sheets reviewed:
  - `tmp/ch08-vertical-images/qa-contact-sheet-zh.jpg`
  - `tmp/ch08-vertical-images/qa-contact-sheet-en.jpg`
  - `tmp/ch08-vertical-images/qa-contact-sheet-ja.jpg`
- Full-size visual QA was performed for high-risk workflow, deployment, HuggingFace, evaluation, inference, state, and study-guide candidates after contact sheet review.
- Visual QA confirmed no forced landscape stretch, squeezed text, wrong-language labels, watermarks, real brand logos, decorative-only posters, or unreadable gibberish text in accepted candidates.
- Numeric/content QA confirmed no invented tool outputs, fake invoice/order data, random business values, fake percentages, concrete service metrics, concrete latency/cost/ports, token IDs, vector values, brand marks, or dashboard values in accepted high-risk candidates.

## Rejections And Regeneration
Initial visual QA rejected and regenerated these high-risk groups before official adoption:
- `function-calling-workflow-en.png` and `function-calling-workflow-ja.png`: invented concrete tool outputs. The prompt now allows only abstract placeholders such as `tool_result`, `validated result`, `observation`, and `final answer`; all three locale variants were regenerated.
- `ch08-docker-image-container-compose-map.png`, `ch08-docker-image-container-compose-map-en.png`, and `ch08-docker-image-container-compose-map-ja.png`: recognizable brand-style Docker/Redis deployment icons and overly concrete config risk. The prompt now forbids Docker whale, Kubernetes wheel, Redis brand-style marks, concrete ports, secrets, URLs, IPs, versions, and environment values.
- `ch08-huggingface-ecosystem-layers-map-ja.png`: brand-like HuggingFace mark. The prompt now forbids official HuggingFace emoji/logo/mascot and concrete vector/token/sample/score values; affected locale variants were regenerated and rechecked.
- `ch08-rag-evaluation-layered-dashboard-map.png`, `ch08-rag-evaluation-layered-dashboard-map-en.png`, and `ch08-rag-evaluation-layered-dashboard-map-ja.png`: fake numeric evaluation risk. The prompt now allows metric names, blank ticks, abstract trend lines, and no-number status points only.
- `ch08-inference-serving-queue-batch-map.png`, `ch08-inference-serving-queue-batch-map-en.png`, and `ch08-inference-serving-queue-batch-map-ja.png`: concrete service metric risk. The prompt now uses only abstract labels such as high/low, fast/slow, small/large, and balanced.
- `ch08-engineering-chapter-flow.png`, `ch08-engineering-chapter-flow-en.png`, and `ch08-engineering-chapter-flow-ja.png`: deployment logo and concrete metric risk. The prompt now forbids real brand-style infrastructure logos and concrete port/cost/latency/percentage/version values.
- `ch08-dialog-state-slot-memory-map-en.png`: fake invoice number in chat history. The prompt now forbids invoice/order/city/name/address/money/date/product placeholders and uses abstract conversation labels only; all three locale variants were regenerated.
- `ch08-study-guide-four-layer-map-en.png`: fake `92%` metric and Docker/Kubernetes-style icons. The prompt now forbids concrete percentages/scores/cost/latency/ports/versions and real brand-style deployment icons; all three locale variants were regenerated.

All regenerated candidates passed dimension, language, stretch, readability, no-fake-value, and no-brand-logo checks.

## Accepted Candidate Set
Accepted candidate lists are tracked in:
- `reports/course-images/ch08-vertical-refine/candidate-targets-zh.txt`
- `reports/course-images/ch08-vertical-refine/candidate-targets-en.txt`
- `reports/course-images/ch08-vertical-refine/candidate-targets-ja.txt`

All 123 listed PNGs are accepted for conversion to same-stem WebP files.

## Conversion Status
- QA status: passed.
- Official WebP conversion: completed for all 123 accepted candidates.
- Post-conversion dimension check: passed for all 123 official WebP files at `1024x1792`.
- Markdown image references remain `.webp`; no Markdown image links were changed to PNG.
