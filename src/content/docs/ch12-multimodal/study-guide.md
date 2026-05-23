---
title: "12.0 Learning Checklist: AIGC and Multimodal"
description: "A compact checklist for Chapter 12: multimodal inputs, structured records, generation versions, safety review, export, and portfolio evidence."
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AIGC checklist, multimodal checklist, image generation, multimodal RAG, creative workflow"
---
Use this page as a printable checklist. If you need the full explanation, return to the [Chapter 12 entry page](/ch12-multimodal/).

![Multimodal portfolio evidence pack](/img/course/ch12-multimodal-evidence-pack-en.webp)

## Two-Hour First Pass

| Time box | Do this | Stop when you can say |
|---|---|---|
| 20 min | Read the workflow loop on the entry page | "Multimodal work starts with source-preserved inputs." |
| 25 min | Run the visual record script | "I can turn visual content into a checkable structured record." |
| 25 min | Skim multimodal basics and image generation | "Understanding and generation need prompts, models, outputs, and review." |
| 25 min | Skim ethics and compliance | "External use needs copyright, portrait, sensitive, and factual checks." |
| 25 min | Read the RAG/Agent bridge | "Multimodal can extend RAG, Agent, and the final capstone." |

## Required Evidence

| Evidence | Minimum version |
|---|---|
| `multimodal_pipeline.md` | input, parsing, generation/understanding, review, export |
| `visual_records.jsonl` | source, page/region/time reference, visible text, objects, uncertainty |
| `prompts/` | prompt versions, reference assets, negative requirements, selection notes |
| `outputs/` | candidate outputs, selected output, rejected output, reason |
| `safety_review.md` | copyright, portrait rights, sensitive content, factuality, usage boundary |
| `README.md` | goal, run command, source materials, sample output, limitations |

## Quality Gates

| Gate | Pass condition |
|---|---|
| Source trace | Every input and output keeps source, owner or license, version, and page/region/time reference when relevant. |
| Prompt/version | Candidate outputs link back to prompt, model or tool, reference assets, and selection reason. |
| Review | Copyright, portrait or voice, sensitive content, factuality, accessibility, and export scope are checked. |
| Export | README, manifest, selected outputs, rejected outputs, limits, and next fix can be inspected by another person. |

## Exit Questions

- Can you preserve source references for screenshots, PDFs, images, audio, or video?
- Can you turn a non-text input into a structured record that RAG or an Agent can use?
- Can you compare generated outputs with prompt versions and review notes?
- Can you explain what must be checked before external release?
- Can you package the result as a final portfolio or capstone demo?

If the answer is yes, you have the multimodal delivery path. Move to Chapter 13 when the project needs open-source model hosting, runtime ownership, or fine-tuning decisions.

<details>
<summary>Check reasoning and explanation</summary>

- Yes means every non-text input has a source, owner, version, and review status, not just a final file.
- A good structured record contains extracted content, modality metadata, confidence or review notes, and a stable link back to the source artifact.
- Generated outputs should be tied to prompt versions, candidate ids, selected/rejected decisions, and reviewer notes so iteration is explainable.
- Before external release, check factual grounding, consent and rights, privacy, sensitive content, safety policy, and whether a human approved high-risk material.
- A portfolio-ready package should include the brief, manifest, prompts, selected assets, rejected cases, review checklist, final export, and a README that explains the workflow.

</details>
## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
brief: user goal, audience, assets, constraints, and export format
artifacts: source files, prompts, generated candidates, selected output, and rejected versions
review: factual check, copyright/portrait/sensitive-content check, and human decision
integration: RAG record, Agent trace, creative package, storyboard, or export preview
Expected_output: reproducible asset package with README, review checklist, and failure notes
```
