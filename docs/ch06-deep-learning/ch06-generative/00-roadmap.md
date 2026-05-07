---
title: "6.6.1 Generative Models Roadmap: Sample, Decode, Review"
sidebar_position: 0
description: "A compact generative models roadmap: latent vectors, GAN, VAE, generated outputs, and evaluation habits."
keywords: [generative model guide, GAN, VAE, latent vector, deep learning]
---

# 6.6.1 Generative Models Roadmap: Sample, Decode, Review

Generative models create new samples instead of only predicting labels. The practical loop is: sample a latent code, decode it, review the output, and compare versions.

## Look at the Generation Flow First

![Generative models chapter relationship diagram](/img/course/ch06-generative-chapter-flow-en.png)

![GAN adversarial balance map](/img/course/ch06-gan-adversarial-balance-map-en.png)

| Concept | First meaning |
|---|---|
| latent vector | compact hidden input used for generation |
| decoder / generator | turns latent code into an output |
| discriminator | judges real vs generated in GANs |
| VAE | learns a smoother latent space |
| review | generated output still needs human and metric checks |

## Run One Tiny Decoder

Create `generative_first_loop.py` and run it after installing `torch`.

```python
import torch

torch.manual_seed(0)
latent = torch.randn(2, 4)
decoder = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Tanh())
generated = decoder(latent)

print("latent_shape:", tuple(latent.shape))
print("generated_shape:", tuple(generated.shape))
print("value_range:", round(generated.min().item(), 3), round(generated.max().item(), 3))
```

Expected output:

```text
latent_shape: (2, 4)
generated_shape: (2, 6)
value_range: -0.863 0.695
```

This is not a real generator yet. It shows the core shape idea: small latent vectors can be decoded into larger outputs.

## Learn in This Order

| Order | Read | What to focus on |
|---|---|---|
| 1 | [6.6.2 GAN](./01-gan.md) | generator, discriminator, adversarial balance |
| 2 | [6.6.3 VAE](./02-vae.md) | encoder, decoder, latent space |

## Pass Check

You pass this roadmap when you can explain the difference between predicting a label and generating a sample, and describe why generated outputs need review rather than blind trust.
