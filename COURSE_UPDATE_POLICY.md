# Course Update Policy

This document defines how AI Roads should keep up with new AI technology while preserving the course standard: trilingual content, visual teaching, runnable practice, and reviewable evidence.

## Update Rhythm

Use three review loops:

| Rhythm | When | What to do |
|---|---|---|
| Monthly lightweight scan | Every 4 weeks | Check major AI model, tooling, protocol, safety, and deployment changes. Decide whether to update an existing page, add a new lesson, or defer. |
| Quarterly curriculum review | Every 3 months | Re-read chapter order, prerequisites, project flow, image quality, runnable examples, and study guides. Check whether the course still reflects current AI engineering practice. |
| Urgent update | Within 1 week of a major change | Update or add content when a major model/API/runtime/security change breaks examples, changes best practice, or introduces an important learner workflow. |

## Sources To Check

Prioritize primary and official sources:

- Official model/provider release notes and documentation.
- Official framework, runtime, protocol, and security documentation.
- Major open-source repository READMEs, model cards, and release notes.
- Research papers only when they clearly affect practical engineering.
- Security references such as OWASP LLM and agentic security guidance.
- Course user feedback, Search Console/Bing reports, broken-link reports, and QA reports.

If information is time-sensitive, verify it with current sources before editing the course.

## Add, Update, Or Defer

Use this decision table:

| Finding | Action |
|---|---|
| Existing lesson is still right but missing a current example | Update that lesson and its three language versions. |
| New technology changes a workflow learners should practice | Add a focused lesson in the closest existing chapter. |
| New model name is interesting but does not change workflow | Mention only if it clarifies a decision table; otherwise defer. |
| Existing code no longer runs | Fix code first, then update explanation and expected output. |
| New security risk affects agents, RAG, tools, or deployment | Add a safety note, runnable sandbox/check, and evidence requirement. |
| Trend is speculative or not reproducible | Defer and add it to a backlog, not to core lessons. |

## Lesson Standard

Every new or substantially updated lesson should include:

- Background and why the technology appeared.
- The concrete problem it solves and what it does not solve.
- A concept map or visual explanation.
- A decision table or checklist.
- Runnable code that works without network access unless the lesson explicitly requires cloud/API access.
- A line-by-line explanation of the code.
- A mini exercise.
- Evidence to keep.
- A small summary.
- A folded check/answer section.

## Trilingual Sync

For every new lesson or major update:

1. Update English, Simplified Chinese, and Japanese pages in the same path structure.
2. Keep titles, descriptions, sidebar order, links, image references, code behavior, and evidence requirements aligned.
3. Localize teaching language; do not leave one language as a partial summary.
4. Update index pages, roadmap pages, study guides, README files, and sidebar-related references when needed.

## Image Standard

Course images should be teaching assets, not decoration.

- Use `scripts/generate_course_images.py` and project image prompts.
- When secondary keys are unstable, use only the main key and generate single-threaded.
- Prefer whiteboard, hand-drawn, or classroom teaching style for conceptual lessons.
- Use richer teaching text, but keep labels readable.
- If one image becomes crowded, split it into two images.
- Avoid SVG-like infographics, pasted UI templates, dark poster style when a lesson needs explanation, real logos, fake metrics, dates, API keys, and tiny unreadable text.
- Generate PNG first, visually inspect, then convert accepted images to WebP for course references.
- Validate image references and check for repeated or mismatched locale assets.

## Code And QA Standard

Before committing a curriculum update:

1. Run or manually verify every new runnable code example.
2. Run Markdown and code audits.
3. Validate course image references.
4. Run course QA and readability/image teaching checks when practical.
5. Run a full build before release or push.
6. Keep the commit focused and describe what changed.

Recommended commands:

```bash
npm run qa:course
npm run qa:code
npm run qa:images
npm run qa:image-teaching
npm run qa:readability
npm run build
```

Use broader checks such as `npm run qa:all` and `npm run build:docker` before publishing larger updates.

## Review Template

Use this checklist during each monthly or quarterly review:

```text
review_date:
review_scope: monthly scan / quarterly review / urgent update
sources_checked:
technology_or_change:
affected_chapters:
decision: update existing / add lesson / defer
reason:
trilingual_pages:
image_needs:
runnable_code_needs:
qa_commands:
evidence_left:
commit:
```

## Commit Rule

Each completed curriculum update should be committed. If the update is large, split commits by coherent unit: content, images, QA/metadata, and release docs. Push only when the maintainer asks for push or when the release workflow requires it.
