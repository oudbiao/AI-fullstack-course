# Chapter 13 Image Deduplication QA

Date: 2026-05-24

## Scope

Generated a new trilingual image2 set for Chapter 13 so each Ch13 page has a distinct teaching image:

- `ch13-open-source-llm-runtime-decision.png`
- `ch13-open-source-llm-runtime-decision-en.png`
- `ch13-open-source-llm-runtime-decision-ja.png`
- `ch13-open-source-llm-study-checklist.png`
- `ch13-open-source-llm-study-checklist-en.png`
- `ch13-open-source-llm-study-checklist-ja.png`

Accepted candidate PNGs are stored in `tmp/ch13-image-dedup/`. Published WebP assets were promoted to `public/img/course/`.

## QA Result

All six accepted candidates are native vertical teaching images at `1024x1792`.

Visual checks passed:

- Hand-drawn whiteboard / worksheet style, not SVG-style infographic or pasted UI-card layout.
- Runtime-decision images teach task, license, hardware, runtime, and fallback instead of repeating the hands-on runtime loop.
- Study-checklist images teach the exit checklist instead of repeating the deployment evidence-pack image.
- Text is sparse and readable on desktop and mobile.
- No real provider logos, real prices, GPU model names, screenshots, account screens, benchmark scores, secrets, or invented metrics.
- English, Simplified Chinese, and Japanese variants keep the same teaching purpose and visual rhythm.

Rejected or regenerated candidates:

- Initial parallel API attempts returned upstream `502` for five images. The failed outputs were not promoted.
- The failed images were retried as smaller single-image jobs and accepted after visual QA.

## Reference Check

The new runtime-decision images are referenced by the 13.3 model/runtime decision page. The new study-checklist images are referenced by the 13.0 study guide. The 13.2 hands-on page keeps the runtime-loop image, and 13.4 keeps the evidence-pack image.
