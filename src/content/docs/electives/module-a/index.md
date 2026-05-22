---
title: "E.A C++ and Model Deployment Roadmap"
description: "A concise hands-on roadmap for the C++ and model deployment elective: move from runtime basics to optimization, inference engines, edge deployment, serving, and a delivery project."
sidebar:
  order: 0
---
Use this elective when a Python model already works, but latency, memory, packaging, or serving cost becomes the real problem.

## See the Deployment Path First

![C++ and Model Deployment module learning map](/img/course/elective-cpp-deployment-module-map-en.webp)

![C++ runtime memory map](/img/course/elective-cpp-runtime-memory-en.webp)

The core question is simple: can you turn model output into a fast, measurable, deployable inference path?

## Run the Smallest C++ Inference Step

Create `demo.cpp`:

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<float> logits = {1.2f, 0.3f, 2.1f};
    int best_index = 0;

    for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
        if (logits[i] > logits[best_index]) {
            best_index = i;
        }
    }

    std::cout << "best_class=" << best_index << "\n";
    std::cout << "score=" << logits[best_index] << "\n";
    return 0;
}
```

Run it:

```bash
c++ -std=c++17 demo.cpp -o demo
./demo
```

Expected output:

```text
best_class=2
score=2.1
```

This is the smallest deployment habit: input tensor-like values, compute a decision, print a reproducible result.

## Learn in This Order

| Step | Lesson | Practice Output |
|---|---|---|
| 1 | [E.A.1 C++ Basics](./01-cpp-basics.md) | Compile and run a tiny inference helper |
| 2 | [E.A.2 Advanced C++](./02-cpp-advanced.md) | Explain ownership, RAII, and safe resource release |
| 3 | [E.A.3 Optimization](./03-model-optimization.md) | Compare latency, memory, and accuracy trade-offs |
| 4 | [E.A.4 Inference Engines](./04-inference-engines.md) | Pick an engine based on hardware and model format |
| 5 | [E.A.5 Edge Deployment](./05-edge-deployment.md) | Name edge constraints and export a checklist |
| 6 | [E.A.6 Model Serving](./06-model-serving.md) | Design versioned serving with metrics |
| 7 | [E.A.7 Project](./07-projects.md) | Deliver a small deployment evidence pack |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
deployment_target: local inference, edge device, model server, or optimization experiment
artifact: C++ snippet, benchmark, model artifact, serving config, or deployment note
metric: latency, memory, throughput, model size, accuracy drop, or reliability
failure_check: ABI/build issue, hardware mismatch, quantization loss, or serving bottleneck
Expected_output: reproducible deployment or optimization evidence, not only theory notes
```

## Pass Check

You pass this module when you can compile one C++ example, explain the deployment trade-off, record latency or memory evidence, and connect the result to the [Elective Hands-on Workshop](../hands-on-elective-workshop.md).

<details>
<summary>Check reasoning and explanation</summary>

A passing evidence pack should include one successful compile/run output, one latency or memory note, and one sentence that explains the deployment trade-off. For example: “The C++ helper returns the same class as the Python prototype, the optimized variant reduces memory, and the remaining risk is an accuracy check on real cases.”

The answer is weak if it only says “the code runs.” Deployment readiness requires a reproducible artifact plus the reason it matters for an actual target.

</details>
