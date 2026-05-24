# Chapter 13 Compute Route Image QA

Date: 2026-05-24

## Scope

Generated a new trilingual image2 set for the added Chapter 13 compute-route lesson:

- `ch13-open-source-llm-compute-routes.png`
- `ch13-open-source-llm-compute-routes-en.png`
- `ch13-open-source-llm-compute-routes-ja.png`

Accepted candidate PNGs are stored in `tmp/ch13-compute-routes/`. Published WebP assets were promoted to `public/img/course/`.

## QA Result

All three accepted candidates are native vertical teaching images at `1024x1792`.

Visual checks passed:

- Hand-drawn whiteboard / classroom teaching style, not SVG-style infographic or pasted UI card layout.
- The image teaches the three compute routes: local CPU, free Colab when available, and rented GPU with stop plan.
- No real provider logos, real prices, GPU model names, screenshots, account screens, benchmark scores, or invented metrics.
- Text is sparse and readable on mobile.
- English, Simplified Chinese, and Japanese variants keep the same teaching purpose and visual rhythm.

Rejected or regenerated candidates:

- The first English API attempt returned upstream `502`; no partial output was used.

## Reference Check

The new compute-route images are referenced by the new `compute-routes.md` page in English, Simplified Chinese, and Japanese.
