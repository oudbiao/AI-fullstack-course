# SVG Teaching Image Audit

Date: 2026-05-07

Scope: `static/img/course/*.svg` and active `docs` / `i18n` references.

Summary:

- Total SVG files: 99
- Three-language SVG groups: 33
- Missing referenced SVG files: 0
- A retain as SVG: 9 groups
- B simplify as SVG: 8 groups
- C recommend image2 replacement: 16 groups

Classification rules:

- A retain SVG: formulas, coordinates, matrices, exact code behavior, command paths, and diagrams where editable vector precision is useful.
- B simplify SVG: keep vector, but reduce repeated title text, long prose, and same-looking colored-box templates.
- C recommend image2: entry-page big diagrams, concept comics, stage/process/comparison maps, and current colored-box arrow diagrams that feel templated.

## C. Recommend Replacing With image2 First

| Priority | SVG group | References | Why replace | image2 prompt direction |
|---|---|---|---|---|
| P0 | `ch09-agent-execution-loop-en.svg`, `ch09-agent-execution-loop.svg`, `ch09-agent-execution-loop-ja.svg` | `docs/ch09-agent/index.md`; `i18n/zh-Hans/.../ch09-agent/index.md`; `i18n/ja/.../ch09-agent/index.md` | Screenshot-reported issue: colored boxes plus arrows plus long text. This is a chapter-index concept loop, not a precision diagram. | Make a vertical Agent operations storyboard: user goal enters a bounded workbench, agent keeps state, plans one step, calls a visible tool, observes result, updates memory, stops or retries, and leaves a trace log. Use minimal labels, visual trace cards, safety boundary, and replay timeline. |
| P0 | `ch09-agent-boundary-map-en.svg`, `ch09-agent-boundary-map.svg`, `ch09-agent-boundary-map-ja.svg` | `docs/ch09-agent/index.md`; zh/ja matching index pages | Conceptual decision map for workflow/RAG/function calling/Agent; current layout is mostly text cards. | Show four adjacent workbenches: fixed workflow conveyor, RAG evidence desk, one-shot function button, autonomous agent loop. Emphasize choosing the simplest reliable control structure. |
| P0 | `ch08-rag-app-loop-en.svg`, `ch08-rag-app-loop.svg`, `ch08-rag-app-loop-ja.svg` | `docs/ch08-rag/index.md`; zh/ja matching index pages | RAG chapter entry loop should feel like retrieval and evidence, not abstract boxes. | Create a RAG application scene: question, query rewrite, document shelves/vector index, retrieved snippets with citations, answer composer, evaluation feedback loop. |
| P0 | `ch10-vision-pipeline-loop-en.svg`, `ch10-vision-pipeline-loop.svg`, `ch10-vision-pipeline-loop-ja.svg` | `docs/ch10-computer-vision/index.md`; zh/ja matching index pages | Vision pipeline benefits from actual visual artifacts: image, crop, boxes, masks, failure examples. | Use one image moving through preprocessing, model output overlays, metrics panel, and error review board. Keep labels sparse and show real visual outputs. |
| P0 | `ch11-text-to-task-pipeline-en.svg`, `ch11-text-to-task-pipeline.svg`, `ch11-text-to-task-pipeline-ja.svg` | `docs/ch11-nlp/index.md`; `docs/ch11-nlp/ch01-text-basics/00-roadmap.md`; zh/ja matching pages | NLP entry pipeline is currently text describing text; image2 can show documents becoming tokens, embeddings, task outputs, and metrics. | Show raw text documents flowing into token strips, vector space, task heads for classification/extraction/generation/QA, then evaluation cards. |
| P0 | `ch12-multimodal-workflow-loop-en.svg`, `ch12-multimodal-workflow-loop.svg`, `ch12-multimodal-workflow-loop-ja.svg` | `docs/ch12-multimodal/index.md`; zh/ja matching index pages | Multimodal workflow should visually show text/image/audio/video assets; current SVG is generic boxes. | Create a production board with text, screenshot, PDF, audio waveform, and video frame feeding parse/align, generate/understand, human review, export package. |
| P1 | `ch07-token-to-answer-lifecycle-en.svg`, `ch07-token-to-answer-lifecycle.svg`, `ch07-token-to-answer-lifecycle-ja.svg` | `docs/ch07-llm-principles/index.md`; zh/ja matching index pages | LLM lifecycle is a concept-stage page image; current flow is too schematic. | Show prompt tokens entering a transformer stack, context window budget, next-token probabilities, decoded answer, and validation. Keep formulas/code labels small. |
| P1 | `ch07-prompt-experiment-loop-en.svg`, `ch07-prompt-experiment-loop.svg`, `ch07-prompt-experiment-loop-ja.svg` | `docs/ch07-llm-principles/index.md`; zh/ja matching index pages | Prompt experiment loop is better as lab notebook / iteration scene than box diagram. | Show prompt versions, test cases, model output samples, scoring checklist, revision notes, and final prompt card. |
| P1 | `ch07-solution-choice-map-en.svg`, `ch07-solution-choice-map.svg`, `ch07-solution-choice-map-ja.svg` | `docs/ch07-llm-principles/index.md`; zh/ja matching index pages | Chapter-level solution chooser can be more memorable as route selection. | Show route signs for prompt-only, RAG, fine-tuning, tool use, and agent; include cost/risk/latency badges and simple decision cues. |
| P1 | `ch08-rag-debug-ladder-en.svg`, `ch08-rag-debug-ladder.svg`, `ch08-rag-debug-ladder-ja.svg` | `docs/ch08-rag/index.md`; zh/ja matching index pages | Debug ladder is a process concept; image2 can show failures at each layer. | Show a diagnostic staircase: document quality, chunking, embedding/index, retrieval, rerank, prompt, answer faithfulness, citations. Include red flags and checkmarks. |
| P1 | `ch10-vision-task-granularity-ladder-en.svg`, `ch10-vision-task-granularity-ladder.svg`, `ch10-vision-task-granularity-ladder-ja.svg` | `docs/ch10-computer-vision/index.md`; zh/ja matching index pages | Needs actual output granularity examples, not text-only ladder. | Use the same image with progressively richer overlays: class label, bounding boxes, segmentation mask, OCR text, visual QA response. |
| P1 | `ch11-nlp-task-output-map-en.svg`, `ch11-nlp-task-output-map.svg`, `ch11-nlp-task-output-map-ja.svg` | `docs/ch11-nlp/index.md`; `docs/ch11-nlp/ch01-text-basics/00-roadmap.md`; zh/ja matching pages | Output-type map is mostly text cards; image2 can show sample outputs. | Show one document splitting into category label, entity JSON, generated summary, retrieval QA answer, and model comparison panel. |
| P1 | `ch12-multimodal-rag-agent-capstone-map-en.svg`, `ch12-multimodal-rag-agent-capstone-map.svg`, `ch12-multimodal-rag-agent-capstone-map-ja.svg` | `docs/ch12-multimodal/index.md`; zh/ja matching index pages | Capstone map is a broad course-connection concept and currently template-heavy. | Show a final capstone control room connecting multimodal records to RAG evidence, Agent tools, prompt versions, review/export, and project README. |
| P2 | `ch05-kaggle-validation-leaderboard-loop-en.svg`, `ch05-kaggle-validation-leaderboard-loop.svg`, `ch05-kaggle-validation-leaderboard-loop-ja.svg` | `docs/ch05-machine-learning/ch06-projects/04-kaggle.md`; zh/ja matching pages | Project workflow can be more concrete with local CV, notebook, submission, leaderboard, and experiment log visuals. | Show a Kaggle-style project desk: local validation chart, one-change experiment card, submission file, public leaderboard, disagreement diagnosis notes. Avoid real Kaggle logos. |
| P2 | `ch05-churn-imbalance-threshold-map-en.svg`, `ch05-churn-imbalance-threshold-map.svg`, `ch05-churn-imbalance-threshold-map-ja.svg` | `docs/ch05-machine-learning/ch06-projects/02-customer-churn.md`; zh/ja matching pages | Project decision graphic; current version is mostly text blocks and would benefit from a concrete imbalance/threshold scene. | Show an imbalanced customer table, minority churn cases, training lever, threshold slider, and cost tradeoff panel. |
| P2 | `ch05-house-price-residual-review-map-en.svg`, `ch05-house-price-residual-review-map.svg`, `ch05-house-price-residual-review-map-ja.svg` | `docs/ch05-machine-learning/ch06-projects/01-house-price.md`; zh/ja matching pages | Project review graphic; image2 can show charts and error buckets instead of text boxes. | Show scatter plot, residual buckets, neighborhood/price-range error cards, and next feature actions. |

## B. Simplify But Keep SVG

| SVG group | References | Suggested simplification |
|---|---|---|
| `ch01-git-merge-conflict-resolution-en.svg`, `.svg`, `-ja.svg` | `docs/ch01-tools/ch02-git/04-branches.md`; zh/ja matching pages | Keep exact conflict markers and branch timeline, but remove the long subtitle and compress five panels into three: diverge, marker, resolve. |
| `ch03-pandas-transform-method-choice-en.svg`, `.svg`, `-ja.svg` | `docs/ch03-data-analysis/ch03-pandas/05-data-transform.md`; zh/ja matching pages | Keep as a compact decision tree; remove prose and use short verbs plus function names. |
| `ch06-pytorch-debug-check-order-en.svg`, `.svg`, `-ja.svg` | `docs/ch06-deep-learning/ch02-pytorch/06-practical-tips.md`; zh/ja matching pages | Keep vector checklist, but reduce each card to one noun phrase and make the update-order strip the visual center. |
| `ch08-rag-evidence-pack-en.svg`, `.svg`, `-ja.svg` | `docs/ch08-rag/study-guide.md`; zh/ja matching pages | Keep as checklist because filenames are exact evidence artifacts; turn into a clean file-stack diagram with less explanation. |
| `ch09-agent-trace-pack-en.svg`, `.svg`, `-ja.svg` | `docs/ch09-agent/study-guide.md`; zh/ja matching pages | Keep exact artifact names; reduce to trace file stack + safety boundary + evaluation record. |
| `ch10-vision-evidence-pack-en.svg`, `.svg`, `-ja.svg` | `docs/ch10-computer-vision/study-guide.md`; `docs/ch10-computer-vision/ch06-projects/00-roadmap.md`; zh/ja matching pages | Keep exact deliverables; simplify to four evidence folders with icons. |
| `ch11-nlp-evidence-pack-en.svg`, `.svg`, `-ja.svg` | `docs/ch11-nlp/study-guide.md`; `docs/ch11-nlp/ch07-projects/00-roadmap.md`; zh/ja matching pages | Keep exact deliverables; simplify text into artifact labels and one-line purpose. |
| `ch12-multimodal-evidence-pack-en.svg`, `.svg`, `-ja.svg` | `docs/ch12-multimodal/study-guide.md`; zh/ja matching pages | Keep exact deliverables; simplify into sources, versions, review, export sections. |

## A. Retain SVG

| SVG group | References | Reason |
|---|---|---|
| `ch01-terminal-pipe-redirection-path-en.svg`, `.svg`, `-ja.svg` | `docs/ch01-tools/ch01-terminal/02-basic-operations.md`; zh/ja matching pages | Command flow, pipe/redirection/PATH behavior, and monospace snippets benefit from precise vector layout. |
| `ch02-mutable-default-trap-en.svg`, `.svg`, `-ja.svg` | `docs/ch02-python/ch01-basics/07-functions.md`; zh/ja matching pages | Exact Python behavior and call results; SVG is appropriate. |
| `ch02-short-circuit-safety-check-en.svg`, `.svg`, `-ja.svg` | `docs/ch02-python/ch01-basics/03-operators.md`; zh/ja matching pages | Exact boolean evaluation order; SVG is appropriate. |
| `ch02-string-index-slice-en.svg`, `.svg`, `-ja.svg` | `docs/ch02-python/ch01-basics/02-data-types.md`; zh/ja matching pages | Index positions and slice boundaries require precision. |
| `ch03-numpy-view-copy-trap-en.svg`, `.svg`, `-ja.svg` | `docs/ch03-data-analysis/ch02-numpy/03-indexing-slicing.md`; zh/ja matching pages | Memory sharing and array cells should remain crisp/editable. |
| `ch03-pandas-resample-rolling-timeline-en.svg`, `.svg`, `-ja.svg` | `docs/ch03-data-analysis/ch03-pandas/08-time-series.md`; zh/ja matching pages | Timeline buckets/windows are clearer as vector geometry. |
| `ch04-matrix-multiplication-shape-rule-en.svg`, `.svg`, `-ja.svg` | `docs/ch04-ai-math/ch01-linear-algebra/02-matrices.md`; zh/ja matching pages | Matrix dimensions need exact alignment. |
| `ch04-pvalue-null-distribution-en.svg`, `.svg`, `-ja.svg` | `docs/ch04-ai-math/ch02-probability/03-statistical-inference.md`; zh/ja matching pages | Distribution curve and tail area need precise visual geometry. |
| `ch04-vector-norm-unit-vector-en.svg`, `.svg`, `-ja.svg` | `docs/ch04-ai-math/ch01-linear-algebra/01-vectors.md`; zh/ja matching pages | Coordinate vector, norm, and unit vector geometry need crisp vector rendering. |

## Generation Decision

No image2 replacements were generated in this sidecar pass. The current C candidates are actively referenced by docs/i18n, so replacing them safely requires either:

1. generating same-name PNGs and then coordinating doc/i18n reference changes, or
2. overwriting SVGs with richer SVG art, which would not solve the underlying image2-vs-template issue.

Recommended first low-risk batch when API capacity is available:

```bash
python3 scripts/generate_course_images.py --only ch09-agent-execution-loop.png
python3 scripts/generate_localized_course_images.py --locale en --only ch09-agent-execution-loop-en.png --ignore-pending --limit 1
python3 scripts/generate_localized_course_images.py --locale ja --only ch09-agent-execution-loop-ja.png --ignore-pending --limit 1
```

Before running those commands, add matching `IMAGE_JOBS` entries or a small one-off generation job for the three `ch09-agent-execution-loop*.png` targets. Then coordinate a docs/i18n reference change from `.svg` to `.png` in the main thread.
