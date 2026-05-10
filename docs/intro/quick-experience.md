---
sidebar_position: 0
title: "0.1 30-Minute AI Quick Experience"
description: "A short first-run AI experience for beginners: see input, model, output, then continue to setup."
keywords: [AI quick experience, Google Colab, image recognition, text generation, image generation, AI introduction]
---

# 0.1 30-Minute AI Quick Experience

![30-minute AI quick experience loop](/img/course/intro-quick-experience-loop-en.webp)

**Just feel the loop:** input -> model -> output -> inspect. Do not memorize terms yet.

## Fastest No-Code Try

Open any AI chat or image tool you can access and try:

```text
Explain RAG to a beginner with one analogy.
```

Then change one word, such as `beginner` to `developer`, and compare the result. Your goal is not to decide whether AI is smart. Your goal is to notice that a small input change can change structure, vocabulary, examples, and confidence.

| What to change | What to inspect |
|---|---|
| Audience: `beginner` -> `developer` | Does the answer change examples and vocabulary? |
| Constraint: add `under 80 words` | Does the model follow length and focus? |
| Format: add `give 3 bullets` | Does the output become easier to scan? |
| Evidence: add `include one limitation` | Does it admit what the answer cannot guarantee? |

This tiny comparison is the first habit in the whole course: never look at one output only. Change one condition, compare, and keep the better result.

## Optional Colab Try

Open [Google Colab](https://colab.research.google.com), create a notebook, and run:

```python
!pip install transformers torch pillow requests -q

from transformers import pipeline
from PIL import Image
import io
import requests

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"
image = Image.open(io.BytesIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).content))

for row in classifier(image)[:3]:
    print(f"{row['label']:30s} {row['score']:.1%}")
```

Expected shape:

```text
Labrador retriever              95.6%
golden retriever                1.0%
kuvasz                          0.5%
```

Your numbers may differ. The important shape is a ranked list of labels and confidence scores.

## Read The Result

| Beginner question | Practical answer | Deeper signal |
|---|---|---|
| What is the input? | One image from a URL | Real systems must check file type, size, source, and privacy |
| What is the model? | A pretrained image classifier | It only knows labels from its training setup |
| What is the output? | Top labels with scores | A high score is not proof; it is model confidence |
| What can go wrong? | Download, install, or model loading may fail | Reliable AI work needs logs, fallback paths, and reproducible environments |

If Colab fails, do not spend the whole day fixing it. Save the error message, continue with the no-code try, and return after Chapter 1 when terminal, Python, and environments are clearer.

## Keep One Note

Create a short note with four lines:

```text
Input tried:
Output observed:
One change I made:
What changed:
```

AI is not magic here: you give input, a trained model processes it, and you inspect the output. Experienced learners should also notice the hidden engineering work: dependency install time, model download, input validation, model limits, and how evidence is recorded. Next, set up the minimum environment.
