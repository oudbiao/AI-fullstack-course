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
- Prefer parallel generation when the script supports it.
- For long jobs, allow a long timeout budget (about 1200s).
- Prefer direct AI-generated teaching images.
- Do not fake the result by generating a background and then pasting text on top.
- Do not use SVG-style infographic aesthetics for replacement teaching images.
- Prefer hand-drawn classroom notes / lined notebook / whiteboard teaching styles when the topic fits.
- If a topic is too abstract, use an analogy, sequence, state change, comparison, or error-repair image.
- If one vertical image is too crowded, split the concept into two or more vertical images.
- Keep the image content tightly tied to the chapter context and the actual code/output shown nearby.
- Do not change outputs just to make the image easier to draw.
- If Chinese text is used, keep it natural Chinese. Allow English terms only when they are real technical terms.
- For multilingual image sets, keep the visual style consistent across EN, ZH, and JA.

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

## Practical notes

- When replacing old SVG assets, remove unused SVGs only after confirming nothing references them.
- Keep report and script changes separate from course content edits when possible.
- If a change needs a follow-up commit, prefer a second small commit over one large mixed commit.
