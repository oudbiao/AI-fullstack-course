# Agent Handoff Notes

This file is for future agents and maintainers. It is not course content. Keep it short; archive long historical notes in reports instead of turning this file into a timeline.

## Operating rules

- Do not push unless the user explicitly asks.
- Prefer small local commits that are easy to review.
- Before every commit, run validation and `npm run build`.
- Do not overwrite user changes.
- Do not commit generated `__pycache__`, `tmp/`, `.codex/`, or stale report noise unless the current task explicitly creates a report.
- After modifying course content, run `npm run clean` before the final build when possible, so stale Astro/Starlight cache does not hide issues.

## Course structure rules

- Keep the default learner route embedded in the course flow rather than relying on learners to read a separate route page first.
- Do not move major chapters unless explicitly requested. The current default route is tools -> Python -> data -> math -> ML -> DL/Transformer -> LLM -> RAG -> Agent -> specializations.
- Keep English, Simplified Chinese, and Japanese aligned in structure, meaning, image references, exercises, folded explanations, and evidence sections.
- Chapter index pages should act as navigation stations: what the learner already knows, what this chapter solves, core path, optional extensions, depth challenge, evidence to keep, and next-chapter bridge.
- Treat beginner-friendly as easy first action, clear evidence, and recoverable failure, not shallow content.
- For model-foundation chapters 4-6, keep the learning practical: runnable code, visible output, metrics/curves/errors, and a bridge into Chapter 7 LLMs.
- For Chapters 6-9, preserve the core/extension split in sidebar labels and chapter indexes. The intended core paths are: ch06 `6.1 -> 6.2 -> 6.5 -> 6.8`, ch07 `7.1 -> 7.2 -> 7.5 -> 7.8`, ch08 `8.1 -> 8.3 -> 8.4 -> 8.5`, and ch09 single-Agent path `9.1 -> 9.2 -> 9.3 -> 9.4 -> 9.8 -> 9.10`.

## Image rules

- Treat images as teaching material, not decoration.
- Read the relevant chapter text before changing any image.
- Use `scripts/generate_course_images.py` for image-generation work; it reads `.env.local` and supported API keys.
- Generated images must stay in temporary review locations until they pass QA. Only promote candidates after checking dimensions, readability, language, layout pressure, concept match, no stretching/warping, no fake brands/watermarks, and no invented metrics/output values.
- Prefer direct teaching images, hand-drawn notes, whiteboard/worksheet style, route maps, evidence chains, debugging order, before/after comparisons, or project checklists. Avoid decorative poster layouts.
- For multilingual image sets, keep the visual style consistent across EN, zh-Hans, and JA.
- Run `python3 scripts/validate_course_image_refs.py` after image changes.

## Current status

- Intro, Chapters 1-12, electives, and appendix have been optimized across English, Simplified Chinese, and Japanese with image references, evidence sections, exercises, and folded explanations.
- Chapter 13 has been expanded from a compact lab into a deeper open-source LLM deployment path: chapter entry, hands-on lab, model/runtime decision, serving/evaluation/release runbook, and study guide.
- The ch08 SOP document assistant intentionally keeps some old `courseware` slugs/image stems for link stability. Do not rename them without adding redirects and updating all references.
- A course quality dashboard is available as `npm run qa:course` / `python3 scripts/course_quality_dashboard.py`. It checks multilingual mirror coverage, evidence blocks, folded summaries, image refs, direct PNG/JPG refs, unbalanced details tags, duplicate summaries, and visible localized-text residue patterns.

## Validation commands

Run these before committing course changes:

```bash
npm run clean
python3 validate_markdown_fences.py
python3 validate_internal_links.py
python3 validate_sidebars.py
python3 validate_course_structure.py
python3 scripts/validate_course_image_refs.py
npm run qa:course
git diff --check
npm run build
```

For rendered diagram work, also run:

```bash
npm run qa:rendered-diagrams
```

For generated-site-only QA after a build, run:

```bash
npm run qa:dist
```

## Historical reference

Detailed historical progress from the large May 2026 cleanup/refinement series was previously stored in this file. If a future task needs old commit context, use git history and reports under `reports/course-images/`. Keep new handoff updates concise: current fact, why it matters, and next action only.
