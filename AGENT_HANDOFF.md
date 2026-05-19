# Agent Handoff Notes

This file is for future agents. It is not course content.

## Operating rules

- Do not push.
- Prefer small local commits that are easy to review.
- Before every commit, run a build check first.
- Do not overwrite user changes.
- Do not commit generated `__pycache__` files.

## Core workflow

1. Read the relevant chapter text line by line before changing any image.
2. Treat images as teaching material, not decoration.
3. Keep English docs, `zh-Hans`, and `ja` aligned in structure, meaning, and image references.
4. If a page or section has a broken image reference, fix the asset or the reference before continuing.
5. After edits, run validation, build, and browser QA.

## Image generation rules

- Use `scripts/generate_course_images.py` for image work.
- The script already reads `.env.local`.
- Available API keys are read from:
  - `OPENAI_API_KEY`
  - `OPENAI_API_KEY_2`
  - `OPENAI_API_KEY_3`
- Prefer parallel generation when the script supports it. Treat each API key as one worker lane: one active image request per key. When a key finishes its image, that same lane should immediately take the next queued image; do not wait for all keys in the batch to finish before starting the next work item.
- For long jobs, allow a long timeout budget (about 1200s).
- Prefer direct AI-generated teaching images.
- Do not fake the result by generating a background and then pasting text on top.
- Do not use SVG-style infographic aesthetics for replacement teaching images.
- Prefer hand-drawn classroom notes / lined notebook / whiteboard teaching styles when the topic fits.
- If a topic is too abstract, use an analogy, sequence, state change, comparison, or error-repair image.
- If one vertical image is too crowded, split the concept into two or more vertical images.
- Prefer vertical teaching images for core lessons, especially step-by-step flows, phone-first reading, and hard concepts that need stacked explanation.
- Vertical image work must be a real vertical recomposition, not a stretched landscape image. Reject images with tall/narrow/compressed lettering, warped circles, squeezed tables, distorted arrows, or text made unreadable to fit the page.
- Generated images must stay in a temporary review location until they pass QA. Only replace course assets after checking dimensions, readability, language, layout pressure, concept match, and absence of stretching/warping.
- Image generation is a candidate step, not acceptance. A generated file is not allowed into `static/img/course` or Markdown references until it passes QA.
- For every generated or regenerated course image, inspect it before adoption. At minimum check: correct aspect ratio, no forced stretch, readable large text, natural language for the locale, no tiny filler/gibberish, no fake brand/logo/watermark, no invented metrics or output values, and a clear match to the nearby lesson/code/output.
- If the image fails QA, regenerate or simplify. Do not “settle” for stretched text, crowded diagrams, decorative posters, wrong language, hallucinated numbers, or unrelated imagery.
- If a concept is too dense for one vertical page, split it into two images or reduce visible text. Never solve density by shrinking or distorting text.
- Result-map images must not invent values. They must match the real code/output shown nearby, or use nonnumeric schematic labels when the page does not provide exact numbers.
- Chapter/project route images should not look like marketing posters. Prefer reviewable teaching artifacts: route maps, evidence chains, debugging order, before/after comparison, output-reading notes, or project checklists.
- For abstract or hard-to-remember ideas, prefer analogy images that make the mental model visible before the formal terms.
- For important historical turning points, algorithm evolution, or "why this changed the field" moments, use multi-panel hand-drawn comics when they improve memory and pacing.
- Keep the image content tightly tied to the chapter context and the actual code/output shown nearby.
- Do not change outputs just to make the image easier to draw.
- If Chinese text is used, keep it natural Chinese. Allow English terms only when they are real technical terms.
- For multilingual image sets, keep the visual style consistent across EN, ZH, and JA.

## Course structure rules

- Keep the default learner route embedded in the course flow rather than relying on learners to read a separate recommended-route page first.
- Do not move major chapters unless explicitly requested. The current default route is tools -> Python -> data -> math -> ML -> DL/Transformer -> LLM -> RAG -> Agent -> specializations.
- Chapter index pages should act as navigation stations: what the learner already knows, what this chapter solves, the core path, optional extensions, depth challenge, evidence to keep, and the next chapter bridge.
- For model-foundation chapters 4-6, keep the learning practical: runnable code, visible output, metrics/curves/errors, and a bridge into Chapter 7 LLMs.
- Treat "beginner-friendly" as "easy first action, clear evidence, and recoverable failure," not shallow content.
- For Chapters 6-9, preserve the core/extension split in sidebar labels and chapter indexes:
  - Chapter 6 core path: 6.1 -> 6.2 -> 6.5 -> 6.8.
  - Chapter 7 core path: 7.1 -> 7.2 -> 7.5 -> 7.8.
  - Chapter 8 core path: 8.1 -> 8.3 -> 8.4 -> 8.5, with deployment/API serving as an extension.
  - Chapter 9 core single-Agent path: 9.1 -> 9.2 -> 9.3 -> 9.4 -> 9.8 -> 9.10; MCP, frameworks, multi-agent, and deployment operations are advanced/elective after the single-Agent loop is stable.

## Quality bar

- Text must be readable.
- No garbled small text.
- No random decorative boxes or pure poster layouts.
- No image that feels unrelated to the surrounding explanation.
- If a generated image feels too close to an infographic or text-only poster, regenerate it.
- If a chart, algorithm, or workflow can be clarified with an actual diagram or step-by-step visual, prefer that.

## Verification

- Run image reference validation after image changes.
- Run `npm run build`.
- If relevant, run the Docker build flow too.
- Open the local site in the browser and inspect the affected chapters directly.
- Check mobile layout, image height, text pressure, chapter pacing, and language switching.

## Current optimization status

- 2026-05-19: Intro, Chapters 1-12, electives, and appendix have been optimized across English, Simplified Chinese, and Japanese.
- 2026-05-19: Chapter 6 vertical image refinement is complete and committed as `f7b70382` (`Refine ch06 content and vertical course images`). QA report and handoff are in `reports/course-images/ch06-vertical-refine/`; 30 approved candidate PNGs were converted to official `static/img/course/*.webp` assets.
- 2026-05-19: Chapter 7 refinement is complete and committed as `13df18b2` (`Refine ch07 LLM principles content`). English, Simplified Chinese, and Japanese `ch07-llm-principles` each have 44 pages with image references, code blocks, evidence sections, and expected-output/result cues. Image references are WebP-only and all referenced course assets exist.
- 2026-05-19: Chapter 8 refinement is complete and committed as `b138a62c` (`Refine ch08 RAG application content`). English, Simplified Chinese, and Japanese `ch08-rag` each have 34 pages with image references, code blocks, evidence sections, and expected-output/result cues. Image references are WebP-only and all referenced course assets exist.
- 2026-05-19: Chapter 9 refinement is complete. English, Simplified Chinese, and Japanese `ch09-agent` each have 70 pages with image references, code blocks, evidence sections, and expected-output/result cues. Image references are WebP-only and all referenced course assets exist. The chapter index preserves the core single-Agent route `9.1 -> 9.2 -> 9.3 -> 9.4 -> 9.8 -> 9.10`, with MCP, frameworks, multi-agent, and deployment operations treated as advanced/elective after the single-Agent loop is stable. Local ch09 audit plus full repository validation/build passed.
- 2026-05-19: Chapter 7-9 vertical image refinement resumed after content commits. Inventories are written to `reports/course-images/ch07-vertical-refine/`, `reports/course-images/ch08-vertical-refine/`, and `reports/course-images/ch09-vertical-refine/`. Current candidate counts from official Markdown WebP refs: ch07 has 45 non-vertical assets, ch08 has 123, ch09 has 153; missing official assets are 0. Candidate PNGs must be generated under `tmp/ch07-vertical-images/`, `tmp/ch08-vertical-images/`, and `tmp/ch09-vertical-images/`, QA'd, then converted to same-stem WebP in `static/img/course/`.
- 2026-05-19: Chapter 7 vertical image refinement is complete. All 45 candidate PNGs in `tmp/ch07-vertical-images/` passed QA and were converted to official same-stem WebP files in `static/img/course/`; QA details are in `reports/course-images/ch07-vertical-refine/qa-report.md`. Validation and `npm run build` passed before commit. Continue with ch08 candidate generation and QA.
- 2026-05-19: ch08 vertical image generation was paused after 19 zh/base candidate PNGs to address localized sidebar/title English residue. Added zh-Hans and ja translations for the new ch06-ch09 sidebar labels with `(Core)`, `(Extension)`, `(Deep Dive)`, `(Core Evidence)`, and related path tags. Also localized visible zh/ja table rows such as "Advanced RAG", "Advanced planning", and "Electives" where they appeared as learner-facing list labels. Custom scans now report 0 missing zh-Hans/ja sidebar translations for those suffixed labels and 0 visible localized message/title hits for the English taxonomy residue. Resume ch08 image generation after validation/commit.
- Latest pass focused on Chapters 1-5. Every page in `ch01-tools`, `ch02-python`, `ch03-data-analysis`, `ch04-ai-math`, and `ch05-machine-learning` now has a localized evidence section, image reference, code block, and expected output/result cue.
- Chapter index pages for Chapters 1-5 now include output-reading guidance so beginners know how to interpret terminal output, files, metrics, and plots.
- Latest verification passed: markdown fences, internal links, sidebars, course structure, image refs, `git diff --check`, and `npm run build`. Image audit reported `course_image_refs=2957 missing=0 unused=0`.

## Practical notes

- When replacing old SVG assets, remove unused SVGs only after confirming nothing references them.
- Keep report and script changes separate from course content edits when possible.
- If a change needs a follow-up commit, prefer a second small commit over one large mixed commit.
