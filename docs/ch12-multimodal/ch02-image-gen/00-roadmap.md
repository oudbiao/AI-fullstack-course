---
title: "12.2.1 Image Generation Roadmap: Prompt, Control, Review"
sidebar_position: 0
description: "A concise hands-on roadmap for image generation: design a prompt package, log parameters, choose a generation mode, and review the output."
keywords: [image generation guide, diffusion model, Stable Diffusion, ControlNet, LoRA]
---

# 12.2.1 Image Generation Roadmap: Prompt, Control, Review

Image generation is a workflow, not a single prompt. A useful result needs intent, prompt records, parameters, optional controls, candidate comparison, and review.

## See the Pipeline First

![Image generation chapter learning flowchart](/img/course/ch12-image-gen-chapter-flow-en.webp)

![Stable Diffusion application mode selector](/img/course/ch12-sd-application-mode-selector-map-en.webp)

![Stable Diffusion fine-tuning route selector](/img/course/ch12-sd-finetuning-route-choice-map-en.webp)

The first habit is to log what you asked for, which mode you used, which seed or parameters shaped the result, and what must be reviewed before export.

## Build a Prompt Record

```python
import json

brief = {
    "topic": "RAG basics",
    "audience": "beginners",
    "style": "clean editorial cover",
}
prompt = f"{brief['style']} for {brief['topic']}, friendly visual metaphor for {brief['audience']}, clear layout"
record = {
    "mode": "text-to-image",
    "prompt": prompt,
    "negative_prompt": "blurry, watermark, unreadable text",
    "seed": 42,
    "review": ["legibility", "copyright", "brand safety"],
}

print(json.dumps(record, indent=2))
```

Expected output:

```text
{
  "mode": "text-to-image",
  "prompt": "clean editorial cover for RAG basics, friendly visual metaphor for beginners, clear layout",
  "negative_prompt": "blurry, watermark, unreadable text",
  "seed": 42,
  "review": [
    "legibility",
    "copyright",
    "brand safety"
  ]
}
```

![Image generation prompt record result map](/img/course/ch12-image-prompt-record-result-map-en.webp)

If you cannot reproduce the prompt record, you cannot reliably improve the image.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | Diffusion intuition | Explain noise, denoising, seed, and sampling |
| 2 | Stable Diffusion parts | Map text encoder, U-Net, VAE, and latent space |
| 3 | Applications and control | Compare text-to-image, image-to-image, inpainting, ControlNet, LoRA |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
prompt_record: prompt, negative requirements, reference, seed/model, and version number
candidate_outputs: generated or simulated results with selection reason
technical_note: diffusion step, latent, cross-attention, LoRA, or application mode
failure_check: prompt drift, style mismatch, artifact, copyright, portrait, or review failure
Expected_output: selected image/version record plus rejected-candidate notes
```

## Pass Check

You pass this chapter when you can write a prompt record, explain which generation mode you chose, save 3 candidate notes, and mark at least one review risk before export.

<details>
<summary>Reference answers and explanation</summary>

1. A passing answer names the modalities involved, the input-output contract, and how text, image, audio, or video evidence is aligned.
2. The evidence should include a real media artifact or trace, plus a note on quality, safety, and failure cases.
3. A good self-check explains whether the task needs generation, understanding, retrieval, tool orchestration, or human review rather than treating every multimodal problem as the same kind of demo.

</details>
