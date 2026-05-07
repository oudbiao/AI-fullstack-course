---
sidebar_position: 0
title: "30-Minute AI Quick Experience"
description: "A short first-run AI experience: image recognition, text generation, and image generation before starting the full course."
keywords: [AI quick experience, Google Colab, image recognition, text generation, image generation, AI introduction]
---

# 30-Minute AI Quick Experience

![30-minute AI quick experience loop](/img/course/intro-quick-experience-loop-en.png)

**Goal:** run three tiny AI examples before studying the theory.

**Need:** a Google account for Colab. No local install.

Do not memorize the terms yet. Copy, run, observe the output, and keep one question for later.

## What You Will Try

| Try | What happens | You will study later |
| --- | --- | --- |
| Image recognition | A model labels a picture | Deep learning and computer vision |
| Text generation | A model continues a sentence | Transformer and large language models |
| Image generation | A model draws from a prompt | AIGC and multimodal AI |

## 1. Image Recognition in Colab

Open [Google Colab](https://colab.research.google.com), create a new notebook, and run:

```python
!pip install transformers torch pillow requests -q
```

Then create a second code cell:

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

results = classifier(image)

print("AI thinks this image is:")
for row in results[:3]:
    print(f"{row['label']:30s} confidence: {row['score']:.1%}")
```

Expected shape of the output:

```text
AI thinks this image is:
Labrador retriever              confidence: 95.6%
golden retriever                confidence: 1.0%
kuvasz                          confidence: 0.5%
```

Your exact confidence values may differ. The point is simple: the model already learned visual patterns from many images, so it can label a new image you did not train it on.

## 2. Text Generation

Add a new cell:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "The future of artificial intelligence is"
result = generator(prompt, max_length=80, num_return_sequences=1)

print(result[0]["generated_text"])
```

GPT-2 is an old, small model. We use it here only because it runs quickly in a free notebook. The core idea is still useful: a language model keeps predicting the next likely token.

## 3. Image Generation Without Code

Open any image generation tool you can access and try this prompt:

```text
a small robot reading a book in a warm library, digital art
```

Change one detail, such as `library` to `spaceship`, and generate again. This is the first taste of prompt control: the words you provide become constraints for the model.

## Finish Here

You have seen the three signals that will appear throughout the course:

| Signal | Course meaning |
| --- | --- |
| Recognition | Models can map inputs to labels |
| Generation | Models can continue or create content |
| Prompt control | Your wording changes the output |

Next, read the capability map only as a picture first. The details will make sense after you build small projects.
