---
title: "12 AIGC and Multimodal"
sidebar_position: 0
description: "Learn multimodal and AIGC through structured inputs, image/text/audio/video workflows, RAG and Agent integration, review, safety, and export."
keywords: [AIGC, multimodal, Stable Diffusion, image generation, video generation, speech synthesis, multimodal large models]
---

# 12 AIGC and Multimodal

![Main visual for AIGC and Multimodal](/img/course/ch12-multimodal-aigc-en.webp)

Chapter 12 is the final expansion: **AI is no longer only text.** Images, PDFs, audio, video, screenshots, charts, and generated assets can all enter the same product workflow.

Do not chase every new demo. First learn how to turn non-text inputs into structured records, connect them to RAG or Agents, generate or edit assets, review risks, and export something usable.

## See the Multimodal Workflow

![Multimodal workflow loop](/img/course/ch12-multimodal-workflow-loop-en.webp)

Use this workflow as the chapter map.

| Layer | What happens | Evidence to keep |
|---|---|---|
| Input | text, screenshot, image, PDF, audio, video | source file, owner, license, version |
| Parse / align | OCR, layout parsing, visual understanding, transcript | structured record, page/region/time reference |
| Understand / generate | answer, caption, image, voice, storyboard, video plan | prompt, model, output, candidate versions |
| Edit / review | human selection, factual check, copyright and portrait checks | review checklist, rejected versions, reason |
| Export / integrate | RAG index, Agent trace, creative package, demo | README, export file, limitations, next step |

## Learning Order And Task List

Make one small workflow traceable before trying video or full creative platforms.

| Step | Read | Do | Evidence to keep |
|---|---|---|---|
| 12.1 | Multimodal basics | turn one screenshot or image into a structured record | source, visible text, objects, uncertainty |
| 12.2 | Image generation | record prompts, references, negative requirements, selected output | prompt versions and review notes |
| 12.3 | Video, speech, digital humans | understand storyboard, voice, shot, subtitle, timing | storyboard and asset list |
| 12.4 | Ethics and compliance | check copyright, portrait rights, sensitive content, factual risk | safety review checklist |
| 12.5 | Stage project | run [12.5.3 Hands-on: Build a Reproducible Multimodal Creative Package](./ch05-projects/02-hands-on-multimodal-workshop.md) | brief, prompts, assets, storyboard, review, export preview |

## First Runnable Loop: Structure A Visual Input

This offline script simulates the first engineering step of a multimodal system: after a model or human reads an image, the result must become a structured, checkable record.

Create `ch12_visual_record.py` and run it with Python 3.10 or later.

```python
visual_record = {
    "source": "course-slide-01.png",
    "content_type": "course screenshot",
    "visible_text": ["RAGOps", "evaluation set", "Trace", "cost monitoring"],
    "objects": ["flowchart", "table"],
    "uncertainty": ["small text in the lower-right corner is unclear"],
    "next_step": "write into the multimodal RAG index for the course Q&A assistant to cite",
}

required_fields = {"source", "content_type", "visible_text", "objects", "uncertainty", "next_step"}
missing = required_fields - visual_record.keys()
rag_ready = not missing and bool(visual_record["visible_text"])

print("source:", visual_record["source"])
print("visible_text_count:", len(visual_record["visible_text"]))
print("uncertainty_count:", len(visual_record["uncertainty"]))
print("rag_ready:", rag_ready)
```

Expected output:

```text
source: course-slide-01.png
visible_text_count: 4
uncertainty_count: 1
rag_ready: True
```

![Visual record RAG-ready result map](/img/course/ch12-visual-record-rag-ready-result-map-en.webp)

Operation tip: add `page`, `region`, or `timestamp` fields. If the record can be cited later, it can enter multimodal RAG. If it cannot be checked or cited, it should stay in review.

## Depth Ladder

| Level | What you can prove |
|---|---|
| Minimum pass | You can turn one screenshot, image, PDF, audio, or video note into a structured record with source and uncertainty. |
| Project-ready | You can preserve source references, prompt versions, candidate outputs, review decisions, and export files. |
| Deeper check | You can connect multimodal records to RAG or Agent while enforcing copyright, portrait, sensitive-content, factual, latency, and cost boundaries. |

## Connect Multimodal To RAG, Agent, And Creative Work

![Multimodal RAG Agent capstone map](/img/course/ch12-multimodal-rag-agent-capstone-map-en.webp)

Multimodal is not separate from the main track.

| Main-track skill | Multimodal extension |
|---|---|
| RAG | retrieve PDF pages, screenshots, charts, image captions, and text chunks with citations |
| Agent | observe screenshots or documents, choose tools, and leave traceable actions |
| Prompt | create image, voice, storyboard, and review prompts with version records |
| Engineering | track assets, licenses, reviews, export files, latency, and cost |
| Capstone | build a multimodal learning assistant or creative workspace |

## Common Failures

- Treating AIGC as "one pretty output" instead of a workflow.
- Losing source references after OCR, PDF parsing, or screenshot understanding.
- Comparing generated results without prompt and version records.
- Skipping human review for copyright, portrait rights, sensitive content, or factual risk.
- Starting with video generation before the storyboard, assets, and review rules are clear.

## Pass Check

Before finishing the course, you should be able to:

- explain how text, images, PDFs, audio, and video enter one workflow;
- run the visual record script and add source references such as page, region, or timestamp;
- preserve prompts, assets, selected outputs, rejected outputs, and review reasons;
- connect a multimodal record to RAG, Agent, or a creative package;
- run the multimodal workshop and keep a README, review checklist, export preview, and failure cases.

For a printable checklist, use [12.0 Learning Checklist](./study-guide.md). For the guided final project, start with [12.5.3 Hands-on: Build a Reproducible Multimodal Creative Package](./ch05-projects/02-hands-on-multimodal-workshop.md).
