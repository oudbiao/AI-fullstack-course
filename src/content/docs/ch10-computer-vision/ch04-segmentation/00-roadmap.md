---
title: "10.4.1 Segmentation Roadmap: Pixel-Level Regions"
description: "A concise hands-on roadmap for image segmentation: understand masks, semantic segmentation, instance segmentation, IoU, and boundary failures."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "image segmentation guide, semantic segmentation, instance segmentation, mask"
---
Segmentation is finer than detection. Instead of a box, it outputs a mask that says which pixels belong to a class or instance.

## See the Mask Workflow First

![Image segmentation chapter learning order diagram](/img/course/ch10-segmentation-chapter-flow-en.webp)

![Semantic segmentation mask example](/img/course/semantic-segmentation-mask-en.webp)

![Semantic segmentation IoU and boundary map](/img/course/ch10-semantic-segmentation-iou-boundary-map-en.webp)

The main object in this chapter is the mask. The main failure is often boundary quality, tiny objects, occlusion, or class confusion.

## Run a Mask IoU Check

This script compares two tiny binary masks.

```python
truth = [
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
]

pred = [
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 0],
]

intersection = 0
union = 0
for y in range(3):
    for x in range(3):
        intersection += truth[y][x] == 1 and pred[y][x] == 1
        union += truth[y][x] == 1 or pred[y][x] == 1

print("mask_iou:", round(intersection / union, 3))
```

Expected output:

```text
mask_iou: 0.5
```

Segmentation reports should show masks, metrics, and boundary errors, not only a colored overlay.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | Semantic segmentation | Predict one class for every pixel |
| 2 | Instance segmentation | Separate different objects of the same class |
| 3 | Segmentation practice | Compare masks, IoU/Dice, boundary errors, and failed samples |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
input_image: original image and target mask or class map
prediction: predicted mask, overlay visualization, and boundary examples
metric: IoU, Dice, per-class score, and boundary failure notes
failure_check: annotation quality, thin boundary, small region, or class confusion
Expected_output: mask overlay plus segmentation metric summary
```

## Pass Check

You pass this chapter when you can create or inspect a mask, compute a simple overlap metric, and explain one boundary or class-confusion failure.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer maps the task to the right visual output: class label, bounding box, mask, OCR text, embedding, or video event.
2. The evidence should include a rendered visual artifact and one metric or qualitative error note.
3. A good self-check names one visual failure mode such as class confusion, missed objects, bad masks, lighting shift, domain shift, or weak annotation quality.

</details>
