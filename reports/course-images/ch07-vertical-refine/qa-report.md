# CH07 Vertical Image QA Report

Updated: 2026-05-19 Asia/Taipei

## Scope
- Chapter: `ch07` official Markdown image references across zh/base, en, and ja docs.
- Candidate directory: `tmp/ch07-vertical-images/`
- Official output directory: `static/img/course/`
- Candidate targets: 45 PNGs, split into 15 zh/base, 15 en, and 15 ja files.

## QA Checks
- All 45 candidate PNGs exist.
- All 45 candidate PNGs are native `1024x1792` vertical images.
- Contact sheets reviewed:
  - `tmp/ch07-vertical-images/qa-contact-sheet-zh.jpg`
  - `tmp/ch07-vertical-images/qa-contact-sheet-en.jpg`
  - `tmp/ch07-vertical-images/qa-contact-sheet-ja.jpg`
- Visual QA confirmed no forced landscape stretch, squeezed text, wrong-language labels, watermarks, real brand logos, decorative-only posters, or unreadable gibberish text.
- Numeric QA confirmed no invented reward scores, percentages, dashboard readings, memory values, benchmark values, or random matrix cell numbers in accepted candidates. Token IDs and symbolic matrix shapes are treated as conceptual teaching notation, not performance metrics.

## Rejections And Regeneration
Initial visual QA rejected six candidates because they displayed concrete numeric values not present in the lesson content:
- `ch07-rlhf-reward-kl-loop-map.png`: concrete reward number.
- `ch07-rlhf-reward-kl-loop-map-en.png`: concrete reward/drift numbers.
- `ch07-rlhf-reward-kl-loop-map-ja.png`: regenerated for consistency with no-number RLHF prompt.
- `ch07-lora-qlora-low-rank-memory-map.png`: concrete memory percentages.
- `ch07-lora-qlora-low-rank-memory-map-en.png`: random matrix cell values.
- `ch07-lora-qlora-low-rank-memory-map-ja.png`: regenerated for consistency with no-number LoRA/QLoRA prompt.

`scripts/generate_course_images.py` now includes stricter vertical refinement instructions and dedicated no-number prompts for the two base concepts. The six rejected candidates were regenerated and visually rechecked. The regenerated files passed dimension, language, stretch, readability, and no-fake-metric checks.

## Accepted Candidate Set
Accepted zh/base candidates:
- `ch07-alignment-app-safety-map.png`
- `ch07-contextual-embedding-sense-map.png`
- `ch07-domain-finetune-evaluation-board-map.png`
- `ch07-embedding-onehot-dense-map.png`
- `ch07-finetune-decision-rag-prompt-peft-map.png`
- `ch07-lora-qlora-low-rank-memory-map.png`
- `ch07-nlp-crash-chapter-flow.png`
- `ch07-rlhf-reward-kl-loop-map.png`
- `ch07-study-guide-evolution-line.png`
- `ch07-tokenizer-granularity-tradeoff-map.png`
- `ch07-tokenizer-inputids-mask-length-map.png`
- `embedding-semantic-space.png`
- `lora-parameter-update.png`
- `rlhf-three-stage-loop.png`
- `tokenizer-subword-flow.png`

Accepted en candidates:
- `ch07-alignment-app-safety-map-en.png`
- `ch07-contextual-embedding-sense-map-en.png`
- `ch07-domain-finetune-evaluation-board-map-en.png`
- `ch07-embedding-onehot-dense-map-en.png`
- `ch07-finetune-decision-rag-prompt-peft-map-en.png`
- `ch07-lora-qlora-low-rank-memory-map-en.png`
- `ch07-nlp-crash-chapter-flow-en.png`
- `ch07-rlhf-reward-kl-loop-map-en.png`
- `ch07-study-guide-evolution-line-en.png`
- `ch07-tokenizer-granularity-tradeoff-map-en.png`
- `ch07-tokenizer-inputids-mask-length-map-en.png`
- `embedding-semantic-space-en.png`
- `lora-parameter-update-en.png`
- `rlhf-three-stage-loop-en.png`
- `tokenizer-subword-flow-en.png`

Accepted ja candidates:
- `ch07-alignment-app-safety-map-ja.png`
- `ch07-contextual-embedding-sense-map-ja.png`
- `ch07-domain-finetune-evaluation-board-map-ja.png`
- `ch07-embedding-onehot-dense-map-ja.png`
- `ch07-finetune-decision-rag-prompt-peft-map-ja.png`
- `ch07-lora-qlora-low-rank-memory-map-ja.png`
- `ch07-nlp-crash-chapter-flow-ja.png`
- `ch07-rlhf-reward-kl-loop-map-ja.png`
- `ch07-study-guide-evolution-line-ja.png`
- `ch07-tokenizer-granularity-tradeoff-map-ja.png`
- `ch07-tokenizer-inputids-mask-length-map-ja.png`
- `embedding-semantic-space-ja.png`
- `lora-parameter-update-ja.png`
- `rlhf-three-stage-loop-ja.png`
- `tokenizer-subword-flow-ja.png`

## Conversion Status
- QA status: passed.
- Official WebP conversion: completed for all 45 accepted candidates.
- Post-conversion dimension check: passed for all 45 official WebP files at `1024x1792`.
- Markdown image references remain `.webp`; no Markdown image links should be changed to PNG.
