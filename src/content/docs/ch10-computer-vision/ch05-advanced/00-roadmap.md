---
title: "10.5.1 Advanced Vision Roadmap: OCR, Face, Video, 3D"
description: "A concise hands-on roadmap for advanced vision directions: choose OCR, face, video, or 3D based on input, output, risk, and project goals."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Advanced Vision Guide, OCR, Video Analysis, Face Recognition, 3D Vision"
---
Advanced vision is not a list of model names. It is a set of application directions built on the same visual foundation: more complex inputs, outputs, constraints, and risks.

## See the Direction Map First

![Advanced vision direction selection map](/img/course/ch10-advanced-vision-route-map-en.webp)

![OCR layout reading order map](/img/course/ch10-ocr-layout-reading-order-map-en.webp)

![Video frame tracking temporal window map](/img/course/ch10-video-frame-tracking-temporal-window-map-en.webp)

OCR fits documents, face recognition fits identity-sensitive scenarios, video fits time and motion, and 3D vision fits spatial structure.

## Run a Direction Choice Check

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

## Learn in This Order

| Step | Direction | Practice Output |
|---|---|---|
| 1 | OCR | Extract text, layout, fields, confidence, failure samples |
| 2 | Face | Detect faces, explain threshold, privacy, and bias risks |
| 3 | Video | Track events across frames and record temporal failures |
| 4 | 3D vision | Explain depth, point cloud, geometry, and sensor assumptions |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
scenario_boundary: face, video, OCR, 3D, medical, or another vision scenario
input_sample: source image/frame/document and the expected output type
result_artifact: extracted text, tracked event, depth clue, diagnosis flag, or review note
failure_check: privacy, lighting, temporal drift, layout, calibration, or domain risk
Expected_output: scenario-specific artifact with metric or human-review note
```

## Pass Check

You pass this chapter when you choose one direction, define input/output, run a minimum project, and document failure cases plus usage boundaries.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer maps the task to the right visual output: class label, bounding box, mask, OCR text, embedding, or video event.
2. The evidence should include a rendered visual artifact and one metric or qualitative error note.
3. A good self-check names one visual failure mode such as class confusion, missed objects, bad masks, lighting shift, domain shift, or weak annotation quality.

</details>
