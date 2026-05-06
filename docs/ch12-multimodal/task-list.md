---
title: "Stage Learning Task Sheet"
description: "Break the AIGC and multimodal stage into actionable learning tasks, practice deliverables, and passing criteria."
keywords: [AIGC, multimodal, image generation, video generation, multimodal applications, learning task sheet]
---

# Stage Learning Task Sheet: AIGC and Multimodal

The goal of this stage is to help you organize text, images, speech, video, and documents into a deliverable AI application. The focus is not on how amazing a single generation result looks, but on whether the input quality, version tracking, human review, copyright boundaries, and final delivery process are clear.

## Required tasks for this stage

| Task | Deliverable | Passing criteria |
| --- | --- | --- |
| Understand multimodal inputs and outputs | A multimodal pipeline diagram | Can explain how text, images, audio, video, and documents enter the system |
| Get image understanding or generation working | A minimal Demo | Can record the input, Prompt, output, and human screening results |
| Complete document or screenshot understanding | A multimodal parsing example | Can preserve source page numbers, regions, or screenshot evidence |
| Add a review and delivery workflow | A review checklist | Can check copyright, portrait rights, sensitive content, and factual risks |
| Complete a stage project | A multimodal work or creative workspace | Includes input/output, version tracking, review, export, and README |

## Recommended learning order

First understand the input/output boundaries of a multimodal system, then study image understanding, image generation, video/speech generation, and multimodal applications. Do not treat generation results as only an aesthetic issue; record the requirements, source materials, Prompt, versions, human screening, and delivery format.

Multimodal projects must pay special attention to source and permissions. External images, people’s portraits, music, video clips, and PDF content may all involve copyright, privacy, or safety boundaries. In your portfolio, explain the usage scope and review method.

## Relationship to the AI learning assistant project

This stage can correspond to AI Learning Assistant v1.0: understanding course slide screenshots, PDFs, voice notes, and text-image materials, and generating review cards, text-image summaries, or presentation materials. It can also serve as the demo layer for the capstone project.

The recommended minimum features are: input a course screenshot or PDF page, extract the key information, generate a structured summary, and preserve source citations and human confirmation records.

## Common sticking points

Common issues include blurry images causing understanding errors, messy PDF table parsing, image generation that does not fit the use case, video storyboards that are not coherent, unclear source materials, and generated results that cannot be exported or reused. When troubleshooting, first check the quality of the original input, Prompt versions, generation parameters, human screening, and review records.

For a guided first run, complete [Hands-on: Build a Reproducible Multimodal Creative Package](./ch05-projects/02-hands-on-multimodal-workshop.md). Use its `multimodal_workshop_run/prompts/`, `assets/`, `outputs/export_preview.html`, and `reports/safety_review.md` as the minimum evidence for this stage.


## Easy version / Standard version / Challenge version tasks

| Difficulty | What you need to complete | Suitable for |
|---|---|---|
| Easy version | Complete one material-to-output example | First-time learners, learners with little time, or beginners |
| Standard version | Record the source of materials, the generation process, and human review | Learners who want to include this stage in their portfolio |
| Challenge version | Compare successful, failed, and boundary cases, and write export limitations | Learners with a foundation who want stronger project evidence |

## Badges and boss fight for this stage

| Type | Content |
|---|---|
| Boss fight | Multimodal chaos entity |
| Unlockable badges | Multimodal reviewer, material curator |
| Minimum pass slogan | Get it working first, then explain it, then record the failures |
| Evidence storage suggestion | Save screenshots, logs, failure samples, or evaluation sheets to `reports/`, `evals/`, or `logs/` |

Once you complete the easy version, you can move on; only recommend putting it in your portfolio after completing the standard version; do the challenge version only when you have extra capacity.

## Portfolio deliverables for this stage

If you want to preserve the results of this stage in your portfolio, it is recommended to keep at least the following files or equivalent materials.

| Deliverable | Description |
| --- | --- |
| `multimodal_pipeline.md` | The system pipeline for input, parsing, generation, review, and export |
| `prompts/` | Multimodal Prompt, reference images, negative prompts, and version records |
| `outputs/` | Generated results, candidate versions, reasons for manual selection, and final deliverables |
| `safety_review.md` | Checks for copyright, portrait rights, sensitive content, factuality, and usage boundaries |
| `README.md` | Project goals, how to run it, source materials, sample outputs, and limitations |

These materials will upgrade a multimodal project from “generate a pretty result” to “a complete work that can explain requirements, sources, versions, review, and delivery boundaries.”

## Stage pass questions

After finishing this stage, you should be able to answer these questions: why multimodal inputs need source preservation, why image generation needs Prompt and version records, why PDF/screenshot understanding needs evidence checks, why content generation needs human review, and when multimodal capabilities are suitable for integration with RAG or an Agent.

## Completion status Checklist

- [ ] I can explain the input, processing, output, and review pipeline of a multimodal application.
- [ ] I can get an image understanding, image generation, or document understanding Demo working.
- [ ] I can record the Prompt, materials, candidate outputs, and selection reasons.
- [ ] I have completed copyright, portrait rights, sensitive content, and factual risk checks.
- [ ] I have integrated multimodal capabilities into a showcase project or capstone project.
