---
title: "A.4 Course Visual Guidelines"
description: "A compact guide for deciding when to use image2 visuals, SVG diagrams, code-generated charts, screenshots, or no image at all."
head:
  - tag: meta
    attrs:
      name: keywords
      content: "course illustrations, instructional design, visual learning, AI course images, image2, SVG diagrams"
---

# A.4 Course Visual Guidelines

![Course image asset planning board](/img/course/appendix-visual-enhancement-kanban-en.webp)

![Flowchart from image gap detection to generation and release](/img/course/appendix-image-production-pipeline-map-en.webp)

Images are teaching content, not decoration. Add or keep an image only when it reduces the learner’s thinking cost.

## Review the learning depth first

Before changing a lesson, read the page as a learner and ask whether it supports three levels at the same time:

| Learner level | Page must provide | Warning sign |
|---|---|---|
| Beginner | A small runnable action, plain-language meaning, expected output, and a recovery path when it fails | The page defines terms but never shows what to run or inspect |
| Practitioner | Decision rules, common failure modes, evidence to keep, and a next project step | The page has a demo but no criteria for judging whether the result is good |
| Experienced learner | Trade-offs, edge cases, debugging signals, evaluation habits, or production constraints | The page feels like a glossary and offers no deeper question to think about |

If a page is too short, add depth around the learner bottleneck instead of adding more definitions. A good page lets a new learner finish one action and lets an experienced learner notice a design or evaluation issue.

## Choose the right visual type

| Learning need | Best visual type | Keep text short |
|---|---|---|
| New concept, chapter entry, story, comparison | image2 teaching illustration or comic | Yes |
| Exact formula, coordinate, matrix, code execution order | SVG or code-generated chart | Yes |
| Real project result | Screenshot or generated mock result | Yes |
| Training curve, metric chart, distribution | Code-generated chart | Yes |
| Reference checklist | Compact table or diagram | Yes |

## Default image strategy

Prefer vertical teaching images for core lessons. A 9:16 or near-vertical image works well for phone-first reading, step-by-step flows, and hard concepts that need stacked explanation. If the teaching goal becomes crowded, split it into two consecutive images instead of shrinking labels or forcing everything into one poster.

Use image style by learning need:

| Need | Preferred style |
|---|---|
| Abstract concept or hard intuition | Analogy image, hand-drawn classroom note, or vertical whiteboard flow |
| Exact model structure, formula, metric, or matrix shape | Professional textbook-style diagram, SVG, or code-generated chart |
| Execution sequence, API call, RAG chain, training loop, Agent trace | Vertical whiteboard process with inputs, actions, outputs, and debug checkpoints |
| Important historical turning point or algorithm evolution | Multi-panel hand-drawn comic with one memorable change per panel |
| Dense concept that has two separate learner bottlenecks | Split image pair: first intuition, then mechanism or debugging |

For analogies, keep the formal concept attached to the visual. For example, show RAG as open-book lookup, gradient descent as walking downhill on loss, Attention as choosing which earlier tokens to listen to, and Agent as a tool-using assistant with a trace log. The analogy should make the next code or formula easier to read, not replace the technical explanation.

## Mobile and three-language image rules

Choose the image direction from the teaching purpose. Use vertical 9:16 for step-by-step lessons, phone-first reading, and hard concepts that need stacked explanation. Use landscape for timelines, side-by-side comparisons, architecture maps, dashboards, or anything learners must scan horizontally. Use a compact square or near-square image only when the concept is a single object, pattern, or checklist. The goal is readable learning, not one fixed ratio.

Generate teaching images as a three-language group: Chinese, English, and Japanese. The group should share the same scene, layout, teaching goal, key values, and visual rhythm. Only the visible learner-facing labels, title, subtitle, and alt text should change by locale.

Use the filename pattern as one source group, then sync the published assets in the same pattern:

```text
topic-name.png
topic-name-en.png
topic-name-ja.png

topic-name.webp
topic-name-en.webp
topic-name-ja.webp
```

After generation, follow the project’s current publishing style: convert or sync the image to WebP before referencing it from course pages. Reduce file size, but do not over-compress. Titles, labels, formulas, code snippets, axes, metric values, and arrows must stay legible on mobile. If compression makes content fuzzy or hard to read, keep a larger WebP or simplify the image before publishing.

When restructuring lessons, it is acceptable to add a clear placeholder first and generate the final image later. The placeholder must still record the teaching goal, target page, image direction, and the three-language filename group so the final batch generation stays consistent.

Do not add a new single-language image to only one locale unless it is a temporary placeholder with a clear follow-up. For exact numeric outputs, formulas, code order, or metric values, prefer an SVG or a generated-result group that locks the same data across all three languages.

## Replace template-heavy SVGs

Some old SVGs are accurate but feel like repeated slide templates: colored boxes, arrows, and long text. For chapter entry pages and broad concept pages, prefer image2 PNGs with a clearer visual story.

Keep SVG when precision matters:

- matrix dimensions
- vector geometry
- terminal paths and commands
- Python evaluation order
- exact data/table shapes

Replace with image2 when the goal is intuition:

- Agent execution loop
- RAG application loop
- chapter-level roadmap
- project workflow
- historical turning points
- before/after comparison

## Image insertion rhythm

Use this order inside teaching pages:

1. Show the image.
2. Run the smallest code or operation.
3. Read the output or result.
4. Explain only the part the learner just saw.

## Minimal production loop

1. Find the learner bottleneck.
2. Define the visual intent in one sentence.
3. Choose the image direction and ratio from the teaching purpose.
4. Add a placeholder or generate/draw the right type of image as a three-language group.
5. For final images, convert or sync the published course assets to WebP, keeping text and key details readable.
6. Insert it in English, Chinese, and Japanese pages with matching alt text.
7. Verify images resolve, mobile readability holds, and the site builds.

Do not add a picture just because the page looks empty. Add it because the next step becomes easier to understand.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
page_target: which lesson or concept needs a teaching image
visual_role: analogy, workflow, history comic, output explanation, or failure debug map
language_set: English, Chinese, Japanese image references or text requirements
risk_check: decorative image, crowded text, unrelated metaphor, or missing nearby explanation
Expected_output: image brief or QA note tied to a specific course page
```
