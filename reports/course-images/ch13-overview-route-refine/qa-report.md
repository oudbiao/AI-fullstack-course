# Chapter 13 Overview Route Image QA

Date: 2026-05-24

## Scope

Generated a separate trilingual image2 set for the Chapter 13 overview page so the overview page and hands-on lab no longer reuse the same runtime-loop image.

Accepted candidate PNGs:

- `tmp/ch13-open-source-llm-overview-route/ch13-open-source-llm-overview-route.png`
- `tmp/ch13-open-source-llm-overview-route/ch13-open-source-llm-overview-route-en.png`
- `tmp/ch13-open-source-llm-overview-route/ch13-open-source-llm-overview-route-ja.png`

Published WebP assets:

- `public/img/course/ch13-open-source-llm-overview-route.webp`
- `public/img/course/ch13-open-source-llm-overview-route-en.webp`
- `public/img/course/ch13-open-source-llm-overview-route-ja.webp`

## QA Result

All three accepted candidates are native vertical teaching images at `1024x1792`.

Visual checks passed:

- Whiteboard or lined-notebook teaching style, not SVG-style infographic layout.
- No pasted-text mockup, screenshot collage, white rounded-card stack, provider logo, or marketing-poster composition.
- Large readable labels and sparse text suitable for mobile reading.
- Chapter-level overview content is distinct from the hands-on runtime loop: route, Self-LLM reference, lab, evaluation, and adaptation decision.
- English, Simplified Chinese, and Japanese versions keep the same teaching purpose and visual rhythm.

Rejected or regenerated candidates:

- The first Japanese candidate misspelled `tiny` as `ting`, so it was rejected and regenerated with safer Japanese wording.
- Earlier API attempts that returned upstream `502` errors were not used.

## Reference Check

After the change:

- Chapter 13 overview pages use `ch13-open-source-llm-overview-route*.webp`.
- Chapter 13 hands-on pages use `ch13-open-source-llm-runtime-loop*.webp`.
- Chapter 13 study-guide pages use `ch13-open-source-llm-evidence-pack*.webp`.

This keeps the overview image, hands-on image, and checklist image visually related but instructionally distinct.
