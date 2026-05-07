---
sidebar_position: 0
title: "30-Minute AI Quick Experience"
description: "A short first-run AI experience: image recognition, text generation, and image generation before starting the full course."
keywords: [AI quick experience, Google Colab, image recognition, text generation, image generation, AI introduction]
---

# 30-Minute AI Quick Experience

![30-minute AI quick experience loop](/img/course/intro-quick-experience-loop-en.png)

**Goal:** run three tiny AI examples before learning the theory.

**Need:** Google Colab and a browser. No local setup.

## 1. What You Will See

| Try | You do | Later chapter |
|---|---|---|
| Image recognition | Give a picture, get labels | Deep learning and vision |
| Text generation | Give a sentence start, get continuation | Transformer and LLM |
| Image generation | Give a prompt, get an image | AIGC and multimodal AI |

## 2. Image Recognition

Open [Google Colab](https://colab.research.google.com), create a notebook, and run two cells.

Cell 1:

```python
!pip install transformers torch pillow requests -q
```

Cell 2:

```python
from transformers import pipeline
from PIL import Image
import io
import requests

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
resp.raise_for_status()
image = Image.open(io.BytesIO(resp.content))

for row in classifier(image)[:3]:
    print(f"{row['label']:30s} confidence: {row['score']:.1%}")
```

Expected shape:

```text
Labrador retriever              confidence: 95.6%
golden retriever                confidence: 1.0%
kuvasz                          confidence: 0.5%
```

Your numbers may differ. The important idea: a trained model can label an image it has never seen from you.

## 3. Text Generation

Add one more cell:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
result = generator(
    "The future of artificial intelligence is",
    max_length=60,
    num_return_sequences=1,
)
print(result[0]["generated_text"])
```

GPT-2 is old and small; it is used here because it runs quickly in a free notebook. The idea is still current: a language model predicts likely next tokens.

## 4. Image Generation

Open any image generation tool you can access and try:

```text
a small robot reading a book in a warm library, digital art
```

Change `library` to `spaceship` and generate again. You just tested prompt control.

## Finish

Keep only three notes:

| Signal | Meaning |
|---|---|
| Recognition | AI maps input to labels |
| Generation | AI continues or creates content |
| Prompt control | Wording changes the result |

Next, open the capability map and read it as a picture first.
