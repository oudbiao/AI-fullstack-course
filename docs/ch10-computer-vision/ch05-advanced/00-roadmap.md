---
title: "10.5.1 Advanced Vision Roadmap: OCR, Face, Video, 3D"
sidebar_position: 0
description: "A concise hands-on roadmap for advanced vision directions: choose OCR, face, video, or 3D based on input, output, risk, and project goals."
keywords: [Advanced Vision Guide, OCR, Video Analysis, Face Recognition, 3D Vision]
---

# 10.5.1 Advanced Vision Roadmap: OCR, Face, Video, 3D

Advanced vision is not a list of model names. It is a set of application directions built on the same visual foundation: more complex inputs, outputs, constraints, and risks.

## 10.5.1.1 See the Direction Map First

![Advanced vision direction selection map](/img/course/ch10-advanced-vision-route-map-en.png)

![OCR layout reading order map](/img/course/ch10-ocr-layout-reading-order-map-en.png)

![Video frame tracking temporal window map](/img/course/ch10-video-frame-tracking-temporal-window-map-en.png)

OCR fits documents, face recognition fits identity-sensitive scenarios, video fits time and motion, and 3D vision fits spatial structure.

## 10.5.1.2 Run a Direction Choice Check

Pick one direction instead of trying all four shallowly.

```python
requirement = {
    "input": "screenshot",
    "needs_text": True,
    "needs_identity": False,
    "needs_time": False,
    "needs_depth": False,
}

if requirement["needs_text"]:
    direction = "OCR"
elif requirement["needs_identity"]:
    direction = "Face"
elif requirement["needs_time"]:
    direction = "Video"
elif requirement["needs_depth"]:
    direction = "3D"
else:
    direction = "Classification or detection"

print("direction:", direction)
print("first_output:", "text with layout")
```

Expected output:

```text
direction: OCR
first_output: text with layout
```

For face, surveillance, medical, or identity projects, write privacy and usage boundaries before showing results.

## 10.5.1.3 Learn in This Order

| Step | Direction | Practice Output |
|---|---|---|
| 1 | OCR | Extract text, layout, fields, confidence, failure samples |
| 2 | Face | Detect faces, explain threshold, privacy, and bias risks |
| 3 | Video | Track events across frames and record temporal failures |
| 4 | 3D vision | Explain depth, point cloud, geometry, and sensor assumptions |

## 10.5.1.4 Pass Check

You pass this chapter when you choose one direction, define input/output, run a minimum project, and document failure cases plus usage boundaries.
