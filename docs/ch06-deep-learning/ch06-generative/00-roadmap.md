---
title: "6.6.1 Generative Models Roadmap: Sample, Decode, Review"
sidebar_position: 0
description: "A compact generative models roadmap: latent vectors, GAN, VAE, generated outputs, and evaluation habits."
keywords: [generative model guide, GAN, VAE, latent vector, deep learning]
---

# 6.6.1 Generative Models Roadmap: Sample, Decode, Review

Generative models create new samples instead of only predicting labels. The practical loop is: sample a latent code, decode it, review the output, and compare versions.

## Look at the Generation Flow First

![Generative models chapter relationship diagram](/img/course/ch06-generative-chapter-flow-en.webp)

![GAN adversarial balance map](/img/course/ch06-gan-adversarial-balance-map-en.webp)

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

![Tiny decoder run result map](/img/course/ch06-generative-tiny-decoder-result-map-en.webp)

## Learn in This Order

| Order | Read | What to focus on |
|---|---|---|
| 1 | [6.6.2 GAN](./01-gan.md) | generator, discriminator, adversarial balance |
| 2 | [6.6.3 VAE](./02-vae.md) | encoder, decoder, latent space |

## Evidence to Keep

Keep one generation review note:

```text
latent_shape: what compact code enters the generator/decoder
output_shape: what sample-like object comes out
quality_check: does it look plausible or reconstruct well?
diversity_check: are outputs varied, or collapsing?
trust_rule: generated output always needs review
```

## Pass Check

You pass this roadmap when you can explain the difference between predicting a label and generating a sample, and describe why generated outputs need review rather than blind trust.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer connects tensors, model layers, loss, `backward()`, and optimizer updates into one training loop.
2. The evidence should include a runnable mini experiment, tensor-shape checks, and a loss or validation curve you can explain.
3. A good self-check names one failure mode such as shape mismatch, no loss decrease, overfitting, data leakage, or using Attention/Transformer words without explaining the data flow.

</details>
