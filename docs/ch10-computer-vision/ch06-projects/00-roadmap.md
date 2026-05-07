---
title: "10.6.1 Project Roadmap: Build a Vision Evidence Pack"
sidebar_position: 0
description: "A concise hands-on roadmap for computer vision projects: connect data, annotation, model output, metrics, failure cases, and presentation."
keywords: [CV project guide, security inspection, medical imaging, image classification project, object detection project]
---

# 10.6.1 Project Roadmap: Build a Vision Evidence Pack

A computer vision project is not “I used a model.” It is a loop of data, annotation, model output, metrics, failure cases, and presentation.

## 10.6.1.1 See the Project Loop First

![Progression map of output granularity in vision tasks](/img/course/ch10-visual-task-progression-map-en.png)

![Closed-loop delivery diagram for vision projects](/img/course/ch10-projects-delivery-loop-en.png)

![Computer vision evidence pack diagram](/img/course/ch10-vision-evidence-pack-en.svg)

Start with classification if you need the fastest complete loop. Move to detection for boxes, segmentation for masks, and OCR/video/3D for specialized scenarios.

## 10.6.1.2 Run a Project Readiness Check

Use this before you call the project presentable.

```python
project = {
    "task": "helmet detection",
    "has_data_note": True,
    "has_metric": True,
    "has_failure_case": True,
    "has_annotation_rule": True,
}

ready = all(project[key] for key in ["has_data_note", "has_metric", "has_failure_case", "has_annotation_rule"])

print("task:", project["task"])
print("presentable:", ready)
```

Expected output:

```text
task: helmet detection
presentable: True
```

If a project has no annotation rule or failure case, it is still a demo, not a portfolio project.

## 10.6.1.3 Learn in This Order

| Step | Project Type | Evidence |
|---|---|---|
| 1 | Classification | Dataset split, accuracy/F1, confusion examples |
| 2 | Detection | Box annotations, IoU/mAP, false positives and missed detections |
| 3 | Segmentation | Masks, IoU/Dice, boundary failures |
| 4 | Industry scenario | Risk notes, user impact, deployment idea |
| 5 | Hands-on workshop | Reproducible mini pipeline before larger project pages |

Run [10.6.4 Hands-on: Build a Reproducible Vision Mini Pipeline](./03-hands-on-vision-workshop.md) before expanding the project.

## 10.6.1.4 Project Deliverable Standards

| Deliverable | Minimum Requirement | Stronger Portfolio Version |
|---|---|---|
| README | Goal, run command, dependencies, examples | Add task boundary, data source, deployment idea |
| Data and annotation | Image source, class list, annotation format | Add annotation examples, quality checks, bias notes |
| Results | At least 1 input image and prediction result | Add correct, false positive, false negative, boundary cases |
| Evaluation | Accuracy, F1, mAP, IoU, Dice, or OCR hit rate | Add error analysis by class, scenario, lighting, clarity |
| Failure analysis | At least 1 real failure | Add suspected cause, fix action, regression check |
| Presentation | Screenshot or short GIF proving it runs | Build a clear visual project page |

## 10.6.1.5 Pass Check

You pass this chapter when your vision project can be reproduced, has clear data and annotation rules, reports proper metrics, and shows where the model fails.
