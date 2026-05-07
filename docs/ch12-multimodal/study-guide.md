---
title: "Study Guide and Task Sheet: How to Learn Multimodal and AIGC Without Getting Lost"
sidebar_position: 1
description: "A multimodal and AIGC study guide for AI full-stack beginners: multimodal basics, image generation, video and speech, digital humans, project roadmap, and acceptance criteria."
keywords: [AIGC study guide, how to learn multimodal, how to learn diffusion models, how to learn image generation, Stable Diffusion]
---

# Study Guide and Task Sheet: How to Learn Multimodal and AIGC Without Getting Lost

If you reach `Chapter 12: AIGC and Multimodal` and feel that images, speech, video, digital humans, and all kinds of new models are scattered all over the place, don’t rush to follow every Demo. On your first pass through multimodal learning, you should understand how different modalities enter the same system.

## Core principle for this stage

Multimodal and AIGC learning should follow one system-level thread: different modalities are encoded into representations, the model performs understanding or generation, and the result is connected to creative, editing, review, and delivery workflows.

![Multimodal study guide workflow diagram](/img/course/ch12-study-guide-modal-workflow-map-en.png)

## Tasks You Must Complete in This Stage

Use these tasks to keep multimodal work deliverable. The focus is not one impressive generated result; the focus is whether inputs, versions, review, rights, and export are clear.

| Task | Deliverable | Passing Criteria |
|---|---|---|
| Understand multimodal inputs and outputs | A multimodal pipeline diagram | Can explain how text, images, audio, video, and documents enter the system |
| Get image understanding or generation working | A minimal demo | Can record the input, prompt, output, and human screening results |
| Complete document or screenshot understanding | A multimodal parsing example | Can preserve source page numbers, regions, or screenshot evidence |
| Add a review and delivery workflow | A review checklist | Can check copyright, portrait rights, sensitive content, and factual risks |
| Run the reproducible creative package | `multimodal_workshop_run/` evidence folder | Can explain prompts, assets, export preview, safety review, and failure cases |
| Complete one stage project | A multimodal work or creative workspace | Includes input/output, version tracking, review, export, and README |

## Recommended learning order

In the first round, learn the multimodal basics first. You need to understand image-text alignment, vision-language models, multimodal input and output, and typical applications.

In the second round, learn image generation. Focus on understanding diffusion models, Stable Diffusion, prompts, ControlNet, LoRA, and common workflows.

In the third round, learn video generation and speech generation. Understand why temporal content is more complex, and how TTS, digital humans, and video generation can be combined.

In the fourth round, learn frontier trends and ethics. AIGC directly involves copyright, portrait rights, bias, misinformation, and regulatory boundaries, so you cannot look only at technical results.

In the fifth round, build a comprehensive project that organizes generation capabilities into a usable creative workflow.

## Suggested learning pace

| Content type | Suggested time | Learning goal |
|---|---|---|
| Multimodal basics | 4–8 hours | Understand how different modalities are aligned |
| Image generation | 8–16 hours | Get one image generation workflow running end to end |
| Video / speech / digital humans | 8–20 hours | Understand temporal generation and asset flow |
| Ethics and compliance | 3–6 hours | Build awareness of content safety and copyright |
| Comprehensive project | 16–32 hours | Complete a generative product prototype |

## Stage project roadmap

For the first project, it is recommended to build an image generation workflow, such as generating posters, covers, or course illustrations based on a theme.

For the second project, it is recommended to build an image-text multimodal Q&A system, such as uploading an image and having the model explain, classify, or generate a description.

For the third project, you can build a prototype of a creative content platform: input a topic, generate copy, images, speech, or video scripts, and add review and export flows.

Before choosing a larger capstone, run the [reproducible multimodal creative package](./ch05-projects/02-hands-on-multimodal-workshop.md). It is the baseline exercise for this stage: one script, a creative brief, prompt versions, generated SVG assets, storyboard, safety review, export preview, and failure cases.

## Common stumbling blocks

The most common stumbling block is chasing new models without understanding the workflow. A generative product is not just one model; it also needs prompts, materials, control conditions, post-processing, review, and delivery.

The second stumbling block is ignoring copyright and portrait-right risks. From the very beginning of an AIGC project, you need to consider material sources, authorization, portrait rights, and content safety.

The third stumbling block is thinking of multimodal as “images plus text.” A real multimodal system must consider how different modalities align, reference each other, be edited, and work together to complete tasks.

## Stage Portfolio Deliverables

![Multimodal review and export map](/img/course/ch12-workshop-review-export-map-en.png)

If you want this stage to become portfolio material, keep at least these files or equivalent evidence.

| Deliverable | Description |
|---|---|
| `multimodal_pipeline.md` | The system pipeline for input, parsing, generation, review, and export |
| `prompts/` | Multimodal prompt, reference images, negative prompts, and version records |
| `outputs/` | Generated results, candidate versions, reasons for manual selection, and final deliverables |
| `safety_review.md` | Checks for copyright, portrait rights, sensitive content, factuality, and usage boundaries |
| `README.md` | Project goals, how to run it, source materials, sample outputs, and limitations |

These materials upgrade a multimodal project from “generate a pretty result” to “a complete work that can explain requirements, sources, versions, review, and delivery boundaries.”

## Stage Completion Questions

After finishing this stage, you should be able to explain how a multimodal system receives text, images, speech, or video, and completes understanding or generation.

Before moving on, check that you can answer these questions:

- Why do multimodal inputs need source preservation?
- Why does image generation need prompt and version records?
- Why does PDF or screenshot understanding need evidence checks?
- Why does content generation need human review?
- When is multimodal capability suitable for integration with RAG or an Agent?

If you can build a small AIGC product prototype with input, generation, editing, review, and export steps, then you have reached the entry-level standard for this direction.
