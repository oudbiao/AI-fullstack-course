---
title: "10.0 Learning Checklist: Computer Vision"
description: "A compact checklist for Chapter 10: pixels, classification, detection, segmentation, metrics, failure samples, and portfolio evidence."
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Computer vision checklist, image classification, object detection, image segmentation, vision metrics"
---
Use this page as a printable checklist. If you need the full explanation, return to the [Chapter 10 entry page](./index.md).

![Vision portfolio evidence pack](/img/course/ch10-vision-evidence-pack-en.webp)

## Two-Hour First Pass

| Time box | Do this | Stop when you can say |
|---|---|---|
| 20 min | Read the output-granularity ladder | "Classification, detection, and segmentation differ by output." |
| 25 min | Run the pixel lab | "I can inspect size, channels, RGB, and grayscale." |
| 25 min | Skim 10.1 image basics | "Preprocessing changes the data the model sees." |
| 25 min | Skim classification, detection, segmentation roadmaps | "I know which metric belongs to which task." |
| 25 min | Read the debugging loop | "I should inspect data and labels before blaming architecture." |

## Required Evidence

| Evidence | Minimum version |
|---|---|
| `opencv_demo.py` or `pixel_lab.py` | image load or generated image, preprocessing, saved output |
| `vision_dataset.md` | data source, classes, sample count, annotation method, limitations |
| `eval_results.md` | accuracy/F1, mAP, IoU/Dice, OCR hit rate, or chosen metric |
| `failure_cases.md` | failed images, possible cause, fix direction |
| `README.md` | task goal, run command, input/output examples, scenario boundary |

## Quality Gates

| Gate | Pass condition |
|---|---|
| Visual trace | Original, processed, prediction, and failure images are saved with matching filenames. |
| Annotation | Dataset notes define classes, boxes or masks, source, split, and known label uncertainty. |
| Metric fit | Accuracy/F1, mAP, IoU/Dice, or OCR hit rate matches the task output. |
| Real-world boundary | Report names lighting, angle, camera or source, latency, image size, and device limits. |

## Exit Questions

- Can you explain classification, detection, segmentation, and OCR by output shape?
- Can you show the original image, processed image, and prediction visualization?
- Can you explain why annotation quality affects metrics?
- Can you choose accuracy/F1, mAP, IoU, or Dice for the right task?
- Can you explain why a demo may fail on real images?

If the answer is yes, you can connect vision to multimodal work in Chapter 12.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer maps the task to the right visual output: class label, bounding box, mask, OCR text, embedding, or video event.
2. The evidence should include a rendered visual artifact and one metric or qualitative error note.
3. A good self-check names one visual failure mode such as class confusion, missed objects, bad masks, lighting shift, domain shift, or weak annotation quality.

</details>


## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
task_output: classification label, detection box, segmentation mask, OCR text, or video event
artifacts: original image, processed image, prediction overlay, metrics file, and failure samples
metric: accuracy/F1, mAP, IoU, Dice, latency, or scenario-specific review score
failure_check: data quality, label error, preprocessing mismatch, threshold, or deployment constraint
Expected_output: a reproducible run folder with visual outputs and a short failure report
```
