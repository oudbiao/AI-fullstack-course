# Intro Career Image QA Report

Date: 2026-05-21

Scope: refresh the `0.4 Plan The Main Route` image set so the homepage and intro route page explain one main engineering route, one project thread, and inspectable portfolio evidence.

## Accepted Candidates

| Candidate PNG | Official WebP target | Size | QA result |
|---|---|---:|---|
| `tmp/intro-career-images/intro-learning-path-selection-en.png` | `static/img/course/intro-learning-path-selection-en.webp` | 1024 x 1792 | Pass |
| `tmp/intro-career-images/intro-learning-path-selection.png` | `static/img/course/intro-learning-path-selection.webp` | 1024 x 1792 | Pass |
| `tmp/intro-career-images/intro-learning-path-selection-ja.png` | `static/img/course/intro-learning-path-selection-ja.webp` | 1024 x 1792 | Pass |

## QA Notes

- All accepted candidates are native vertical images, not stretched landscape images.
- Large labels are readable in the intended language, with no obvious garbled microtext in the teaching-critical areas.
- The visuals teach main-route pacing and project-thread evidence instead of presenting multiple competing learning paths.
- No fake dashboard metrics, invented numeric performance claims, brand logos, or purely decorative poster layouts were accepted.
- The first Japanese candidate was rejected because it paired LLM/RAG/Agent with incorrect chapter numbers. The prompt was hardened and the regenerated Japanese candidate uses capability nodes plus correct Ch10-12 specialization labels.

## Promotion

Accepted PNGs were converted to same-stem WebP assets in `static/img/course/` at quality 92. Markdown references remain WebP-only.
