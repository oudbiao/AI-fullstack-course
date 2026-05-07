---
title: "A.2 Course Numbering Convention"
description: "Explains the mapping between source directories like ch01-tools and ch02-python and the displayed chapters 1–12 on the website, so course maintenance stays consistent."
keywords: [course numbering, chapter directory, chapter numbering, course maintenance]
---

# A.2 Course Numbering Convention

![Map showing the correspondence between chapter numbers and source directories](/img/course/appendix-course-numbering-map-en.png)

![Naming consistency check diagram for course maintenance](/img/course/appendix-course-numbering-maintenance-check-en.png)

:::tip Reading tip
When maintaining the course, keep the page title, sidebar order, source directories, and image names aligned with one another. When reading the diagram, separate the “display number” from the “file path” to avoid mixing up chxx with Chinese chapter numbers.
:::

When the course pages are shown to learners, use the display numbering from Chapter 1 to Chapter 12 consistently. The source directories are also aligned with the displayed chapter numbers: `ch01-*` corresponds to Chapter 1, `ch02-*` corresponds to Chapter 2, and so on.

The second half of each directory name is used to describe the topic. For example, `ch05-machine-learning` means Chapter 5, Machine Learning, and `ch09-agent` means Chapter 9, AI Agent. The “Tracks 1–4” in the sidebar are only learning groups and do not represent the file directory hierarchy.

## Correspondence

| Source directory | Displayed chapter on website | Course name |
|---|---|---|
| `docs/ch01-tools` | Chapter 1 | Developer Tools Basics |
| `docs/ch02-python` | Chapter 2 | Python Programming Basics |
| `docs/ch03-data-analysis` | Chapter 3 | Data Analysis and Visualization |
| `docs/ch04-ai-math` | Chapter 4 | Minimum Essential AI Math Basics |
| `docs/ch05-machine-learning` | Chapter 5 | Introduction to and Practice of Machine Learning |
| `docs/ch06-deep-learning` | Chapter 6 | Deep Learning and Transformer Basics |
| `docs/ch07-llm-principles` | Chapter 7 | LLM Principles, Prompt, and Fine-Tuning |
| `docs/ch08-rag` | Chapter 8 | LLM Application Development and RAG |
| `docs/ch09-agent` | Chapter 9 | AI Agent and Intelligent Agent Systems |
| `docs/ch10-computer-vision` | Chapter 10 | Computer Vision |
| `docs/ch11-nlp` | Chapter 11 | Natural Language Processing |
| `docs/ch12-multimodal` | Chapter 12 | AIGC and Multimodality |

## Writing rules

In page titles, introductions, task sheets, appendix notes, and image progress records, prefer the displayed chapter number on the website, for example, “Chapter 5, Machine Learning.”

For nested lessons, use a three-level display number so readers always know where they are:

| Level | Format | Example |
|---|---|---|
| Start-here guide | `0.K` | `0.1 30-Minute AI Quick Experience` |
| Chapter index | `N` | `4 AI Math` |
| Chapter study guide | `N.0` | `4.0 Study Guide and Task Sheet` |
| Section category | `N.M` | `4.1 Linear Algebra` |
| Page inside a section | `N.M.K` | `4.1.2 Vectors` |
| Elective module | `E.X` | `E.A C++ and Model Deployment` |
| Elective lesson | `E.X.K` | `E.A.1 C++ Programming Basics` |
| Appendix page | `A.K` | `A.2 Course Numbering Convention` |

Do not use local-only numbers such as `1.2` inside Chapter 4, because learners who open that page directly cannot tell whether it belongs to `4.1`, `5.1`, or another chapter.

When referring to file paths, code scripts, image filenames, or internal links, use source directory names like `ch05-machine-learning`.

Do not add old-style stage directories or stage directories with letter suffixes. When adding new chapters, images, or script configurations, prioritize the numbering system from `ch01-*` to `ch12-*`.

If a sentence must include both, the recommended format is:

```text
Chapter 5, Machine Learning (directory docs/ch05-machine-learning)
```
