---
title: "12.5.1 Integrated Project Roadmap: Creative Package Workflow"
sidebar_position: 0
description: "A concise hands-on roadmap for the AIGC integrated project: turn a brief into copy, image prompts, video script, asset versions, review, and export."
keywords: [AIGC project guide, creative platform, multimodal project, content generation workflow]
---

# 12.5.1 Integrated Project Roadmap: Creative Package Workflow

The capstone is not about connecting many model APIs. It is about giving a user a workflow: input a brief, generate assets, compare versions, edit, review, and export a usable content package.

## 12.5.1.1 See the Product Loop First

![AIGC creative platform project delivery loop diagram](/img/course/ch12-projects-delivery-loop-en.png)

![Creative package pipeline](/img/course/ch12-workshop-creative-package-pipeline-map-en.png)

![Prompt, asset, and version map](/img/course/ch12-workshop-prompt-asset-version-map-en.png)

![Review and export map](/img/course/ch12-workshop-review-export-map-en.png)

The first habit is to save every generated result as an asset with source, prompt, version, review status, and export target.

## 12.5.1.2 Build the Minimum Package State

```python
brief = {
    "topic": "RAG mini course",
    "audience": "new learners",
}
package = {
    "brief_ready": True,
    "assets": ["title", "cover_prompt", "video_script", "review_checklist"],
    "has_versions": True,
    "has_review": True,
}

ready = package["brief_ready"] and package["has_versions"] and package["has_review"] and len(package["assets"]) >= 4

print("package_ready:", ready)
print("assets:", ", ".join(package["assets"]))
```

Expected output:

```text
package_ready: True
assets: title, cover_prompt, video_script, review_checklist
```

If this state is missing, the project will look like a demo instead of a product.

## 12.5.1.3 Start with the Workshop

Run [12.5.3 Hands-on: Build a Reproducible Multimodal Creative Package](./02-hands-on-multimodal-workshop.md) before expanding the larger creative platform. It gives you the smallest reproducible loop for brief intake, prompt records, asset versions, storyboard export, safety review, and failure analysis.

## 12.5.1.4 Project Deliverable Standards

| Deliverable | Minimum Requirement |
|---|---|
| README | Goal, run command, dependencies, material sources, examples |
| Sample package | One complete brief with generated assets and review notes |
| Version records | At least two candidate outputs or one edited revision |
| Safety review | Copyright, portrait, voice, sensitive content, export label |
| Failure note | One real failure case and the next fix |

## 12.5.1.5 Pass Check

You pass this chapter when your project can accept a brief, produce a structured creative package, keep versions, run review, and export Markdown or JSON that another person can inspect.
