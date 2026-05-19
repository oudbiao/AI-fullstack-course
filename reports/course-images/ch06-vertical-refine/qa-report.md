# ch06 Vertical Image QA Report

Updated: 2026-05-19, Asia/Taipei.

## Scope

Candidate source:

- `tmp/ch06-vertical-images/*.png`

Official target:

- `static/img/course/*.webp`

This QA covered 30 Chapter 6 candidate PNGs. Every candidate is `1024x1792` and has a same-name WebP target in `static/img/course/`. Existing Markdown references already point to WebP files and were not changed to PNG.

## QA Criteria

Each candidate was checked for:

- Real vertical recomposition, not a landscape graphic stretched into portrait format.
- Readable labels and diagrams without squeezed text, warped shapes, or distorted arrows.
- Correct locale and natural language for zh-Hans, en, and ja variants.
- No gibberish microtext, wrong-language filler, fake brands, watermarks, or pure decorative poster treatment.
- No fabricated run-specific metric values, outputs, losses, accuracies, or benchmark numbers.
- Fit with Chapter 6 teaching content and the linked article context.

## Decision

All 30 candidates are accepted for conversion to official WebP assets.

Notes:

- `ch06-study-guide-training-loop.png` was previously regenerated after an old candidate had broken numbering. The accepted candidate now uses continuous 1-8 steps.
- `ch06-study-guide-training-loop-ja.png` was previously regenerated after an old candidate displayed concrete accuracy/loss values not tied to real output. The accepted candidate no longer shows fake metric values.
- `ch06-causal-mask-no-peeking-map-en.png` uses a small simplification around "past/current", but the diagram includes current-token visibility and remains pedagogically correct.

## Accepted Files

- `ch06-attention-qkv-library-analogy-map.png`
- `ch06-attention-qkv-library-analogy-map-en.png`
- `ch06-attention-qkv-library-analogy-map-ja.png`
- `ch06-backprop-error-responsibility-map.png`
- `ch06-backprop-error-responsibility-map-en.png`
- `ch06-backprop-error-responsibility-map-ja.png`
- `ch06-causal-mask-no-peeking-map.png`
- `ch06-causal-mask-no-peeking-map-en.png`
- `ch06-causal-mask-no-peeking-map-ja.png`
- `ch06-deep-learning-project-cycle.png`
- `ch06-deep-learning-project-cycle-en.png`
- `ch06-deep-learning-project-cycle-ja.png`
- `ch06-neuron-linear-activation-gate.png`
- `ch06-neuron-linear-activation-gate-en.png`
- `ch06-neuron-linear-activation-gate-ja.png`
- `ch06-nn-module-parameter-flow.png`
- `ch06-nn-module-parameter-flow-en.png`
- `ch06-nn-module-parameter-flow-ja.png`
- `ch06-projects-portfolio-loop.png`
- `ch06-projects-portfolio-loop-en.png`
- `ch06-projects-portfolio-loop-ja.png`
- `ch06-pytorch-chapter-flow.png`
- `ch06-pytorch-chapter-flow-en.png`
- `ch06-pytorch-chapter-flow-ja.png`
- `ch06-study-guide-training-loop.png`
- `ch06-study-guide-training-loop-en.png`
- `ch06-study-guide-training-loop-ja.png`
- `ch06-transformer-chapter-flow.png`
- `ch06-transformer-chapter-flow-en.png`
- `ch06-transformer-chapter-flow-ja.png`

## Adoption Result

Accepted PNGs have been converted to same-name WebP files in `static/img/course/` with the existing Markdown references left unchanged.
