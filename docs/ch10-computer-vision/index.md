---
title: "10 Computer Vision (Elective Track)"
sidebar_position: 0
description: "Learn the core tasks of computer vision, including image fundamentals, OpenCV, image classification, object detection, image segmentation, and hands-on visual projects."
keywords: [computer vision, OpenCV, image classification, object detection, YOLO, image segmentation]
---

# 10 Computer Vision (Elective Track)

![Computer Vision main visual](/img/course/ch10-computer-vision-en.png)

This stage is about “how to make models understand images.” It is an elective track: if your main goal is LLM applications and Agent development, you can come back to it later; if you want to work on vision, multimodal systems, industrial inspection, OCR, or medical imaging, it is recommended to study this systematically.

## Story-driven introduction: teaching the model to see the world

When humans look at an image, we naturally recognize objects, positions, boundaries, and actions; a model only sees a matrix of pixels. What computer vision does is help the model gradually learn from pixels: “what is this” and “where is it” and “where are the boundaries.” From classification to detection to segmentation, each step makes the model see more precisely.

## Interactive exercise: ask three levels of questions about the same image

Take an image with multiple objects and first ask, “What is the main category of this image?” Then ask, “Where is each object?” Finally ask, “What are the boundaries of each object?” These three questions correspond to classification, detection, and segmentation. You’ll find that the difficulty of vision tasks does not suddenly increase; rather, the outputs become more and more detailed.

## Project bonus content

A bonus project for this stage could be a “visual detection mini tool”: after uploading an image, the system performs preprocessing, recognizes objects, marks their locations, and outputs confidence scores and result explanations. It can later be upgraded into an OCR document assistant, an industrial defect detection tool, or a multimodal Q&A project.

## Stage positioning

| Information | Description |
|---|---|
| Suitable for | Learners who have completed the basics of deep learning and want to move into vision or multimodal directions |
| Estimated study time | 120–180 hours |
| Prerequisites | Complete the basics of deep learning and Transformer |
| Stage deliverables | Image classification, object detection, image segmentation, or a comprehensive vision project |

## Minimum path for beginners

Beginners should first understand image pixels, color spaces, OpenCV preprocessing, and the differences between classification, detection, and segmentation. There is no need to chase the newest model at the very beginning. As long as you can train or call an image classification model and clearly explain what detection and segmentation output beyond classification, you’ve completed the minimum path.

## Advanced path

Experienced learners can go deeper into data annotation, augmentation strategies, YOLO, segmentation models, mAP, deployment scenarios, and failure case analysis. Try further to connect a vision model to a small application that outputs bounding boxes, confidence scores, and explanations of error examples.

## How vision tasks become more detailed step by step

Computer vision is not a single task. It usually becomes more complex in stages by output granularity: first determine what the whole image is, then locate the objects, then determine which region each pixel belongs to.

![Progression map of vision task output granularity](/img/course/ch10-visual-task-progression-map-en.png)

## What beginners should do first, and what advanced learners should do later

When beginners study this stage for the first time, they should focus on the main line of image tasks: how images become tensors, how convolutions extract local features, and what problems classification, detection, and segmentation solve respectively.

Experienced learners can focus on data and evaluation: annotation quality, class imbalance, IoU, mAP, failure samples, and deployment speed. Your goal is to turn a vision demo into a project that can be explained, evaluated, and iterated on.

## Learning path for this stage

Chapter 1 covers CV basics and OpenCV, helping you understand image pixels, color spaces, filtering, edges, morphology, and basic image processing.

Chapter 2 covers advanced image classification, including data augmentation, modern classification architectures, and training techniques.

Chapter 3 covers object detection, helping you understand candidate boxes, classes, confidence, IoU, mAP, and the YOLO family.

Chapter 4 covers image segmentation, helping you understand semantic segmentation, instance segmentation, and pixel-level outputs.

Chapter 5 covers advanced topics, including face detection, video analysis, OCR, and 3D vision.

Chapter 6 completes a comprehensive project that connects data, models, metrics, and application scenarios.

## What you should be able to do after learning

- Explain the differences between classification, detection, and segmentation tasks
- Use OpenCV to complete basic image processing
- Train or fine-tune an image classification model
- Understand the inputs, outputs, and evaluation metrics of object detection and segmentation tasks
- Prepare data, train a model, and analyze results for a vision project

## Common misconceptions

Don’t just chase the newest vision model. The real difficulty in vision projects often lies in data collection, annotation quality, class imbalance, metric selection, and deployment scenarios.

Also don’t separate OpenCV from deep learning. OpenCV is suitable for traditional image processing and engineering preprocessing, while deep learning is suitable for complex recognition tasks; the two often appear together.

## Vision failure theater: model mistakes usually do not have only one cause

If classification results are unstable, first check whether the training images are clear, whether the classes are balanced, and whether augmentation is too aggressive. If detection misses small objects, check annotation quality, image resolution, and evaluation thresholds. If the demo works well on sample images but performs poorly on real images, suspect a mismatch in data distribution first.

## Minimal runnable experiment: read an image and output inspectable results

The minimum experiment for this stage can start with OpenCV or PIL: read an image, process its size, channels, cropping, or grayscale conversion, and save the processed result. Then replace it with a classification, detection, or OCR model.

```python
from PIL import Image

img = Image.open("sample.jpg")
print(img.size, img.mode)
small = img.resize((224, 224))
small.save("sample_224.jpg")
```

Vision projects must keep the input image, processed result, and prediction visualization. Otherwise, when the model makes a mistake, it is hard to tell whether the problem comes from the data, annotation, preprocessing, or the model itself.

## Vision failure case library: check input quality and annotation boundaries first

| Symptom | Common cause | How to locate it | Fix direction |
|---|---|---|---|
| Unstable classification | Blurry images, class imbalance, excessive augmentation | Check misclassified images and class distribution | Clean the data, adjust augmentation strategy |
| Detection misses small objects | Low resolution, inconsistent annotations, overly high threshold | Visualize boxes and confidence scores | Increase resolution, check annotations, tune thresholds |
| Rough segmentation boundaries | Inaccurate annotation edges or low model output resolution | Compare the mask with the original image | Improve annotation standards, use more suitable metrics |
| Demo looks good but real images look bad | Training data and real-world scene distribution differ | Compare lighting, angle, and background | Add real samples and scene descriptions |

## Stage acceptance rubric

| Level | Acceptance criteria | Portfolio evidence |
|---|---|---|
| Minimum pass | Can explain the inputs and outputs of classification, detection, segmentation, and OCR | Input image, prediction results |
| Recommended pass | Can train or call a vision model and calculate metrics | Data description, metrics, visualized results |
| Portfolio pass | Can analyze false positives, false negatives, and scenario risks | Error sample set, annotation notes, project report |

## Stage project

The basic version is to complete an image classification project, including data preparation, training, and basic evaluation. The standard version should add data augmentation, error sample analysis, and visualization of prediction results. The challenge version can be an object detection or segmentation project, with annotation formats, mAP/IoU metrics, inference demos, and scenario-based application descriptions.

If you want one guided baseline before choosing a larger direction, start with [Hands-on: Build a Reproducible Vision Mini Pipeline](./ch06-projects/03-hands-on-vision-workshop.md). It gives you a local script that generates images, masks, boxes, metrics, prediction visualizations, and a failure report.

If you want a more detailed learning rhythm, you can read [Study Guide: How to Learn Computer Vision Without Getting Confused](./study-guide.md).





## Stage deliverables

| Deliverable | Minimum version | Portfolio version |
|---|---|---|
| Image classification experiment | Can train or call a model to complete classification | Includes data split, augmentation strategy, metrics, and prediction visualization |
| Image processing script | Completes reading, cropping, grayscale, edge, and other processing | Explains how preprocessing affects model inputs and results |
| Error sample set | Saves several misclassified images | Analyzes clarity, class confusion, annotation quality, and data distribution |
| Vision project report | Clearly describes the task, data, and metrics | Shows mAP/IoU/accuracy, visualized results, and limitations |
| Application demo | Can infer on a single image | Includes input/output examples, run commands, and scenario boundaries |

## Relationship with the AI Learning Assistant capstone project

This stage can add vision capabilities to the AI Learning Assistant: recognizing lecture screenshots, extracting text with OCR, or analyzing image-based learning materials. If you are following the capstone project path, it is recommended that by the end of this stage you submit at least one version log: what new capabilities were added, how to run it, what the sample inputs and outputs are, what problems you encountered, and what you plan to improve next.


## Stage completion criteria

| Completion level | What you need to do |
|---|---|
| Minimum pass | Understand the inputs and outputs of vision tasks such as image classification, detection, segmentation, and OCR. |
| Recommended pass | Complete at least one runnable mini project in this stage, and record the run method, sample input/output, and problems encountered in the README. |
| Portfolio pass | Integrate the outputs of this stage into the “AI Learning Assistant” capstone project, and leave screenshots, logs, evaluation samples, and next-step plans. |

After finishing this stage, you do not need to memorize every detail. What matters more is being able to clearly explain: what problem this stage solves, how it relates to the previous stage, and how it supports later learning. The multimodal stage will continue to use vision understanding and generation capabilities.
