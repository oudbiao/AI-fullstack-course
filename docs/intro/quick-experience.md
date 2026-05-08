---
sidebar_position: 0
title: "0.1 30-Minute AI Quick Experience"
description: "A short first-run AI experience for beginners: see input, model, output, then continue to setup."
keywords: [AI quick experience, Google Colab, image recognition, text generation, image generation, AI introduction]
---

# 0.1 30-Minute AI Quick Experience

![30-minute AI quick experience loop](/img/course/intro-quick-experience-loop-en.webp)

**Just feel the loop:** input -> model -> output. Do not memorize terms yet.

## Fastest No-Code Try

Open any AI chat or image tool you can access and try:

```text
Explain RAG to a beginner with one analogy.
```

Then change one word, such as `beginner` to `developer`, and compare the result.

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

## Keep One Note

AI is not magic here: you give input, a trained model processes it, and you inspect the output. Next, set up the minimum environment.
