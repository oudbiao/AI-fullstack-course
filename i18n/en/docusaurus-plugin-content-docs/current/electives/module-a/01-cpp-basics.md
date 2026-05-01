---
title: "1.1 C++ Programming Basics"
sidebar_position: 1
description: "Build C++ intuition for model deployment scenarios, starting from variables, functions, references, vector, and simple classes."
keywords: [C++, basics, vector, reference, function, class, deployment]
---

# C++ Programming Basics

![C++ runtime and memory model diagram](/img/course/elective-cpp-runtime-memory-en.png)

:::tip Where this section fits
Many students working on AI applications are more familiar with Python.  
But once you move into model deployment, inference services, edge devices, or high-performance modules, you will often run into C++.

This lesson does not try to cover every syntax rule from a traditional language textbook.  
Instead, it starts from the perspective of “what you need for model deployment” and fills in the most common C++ basics first.
:::

## Learning Objectives

- Understand why C++ is often used in model deployment scenarios
- Master the basics of variables, functions, references, `std::vector`, and simple classes
- Build intuition for “how C++ organizes data and logic” through compilable examples
- Understand why later advanced lessons will continue with RAII, smart pointers, and abstract interfaces

---

## 1. Why do we encounter C++ in model deployment?

### 1.1 Because it is closer to the underlying execution environment

Common scenarios include:

- Inference engine SDKs
- Low-level interfaces for ONNX / TensorRT / OpenVINO
- High-performance post-processing modules
- Local inference on edge devices

### 1.2 It is not “just for showing off”

Most of the time, C++ is not chosen because it is cooler.  
It is chosen because deployment scenarios care more about:

- Performance
- Memory control
- Native library integration

### 1.3 A realistic goal

For many AI engineers,  
the first goal is usually not “master every C++ syntax rule,”  
but rather:

> **Understand basic code, write simple modules, and connect them to the deployment pipeline.**

---

## 2. First, build a few of the most common basic concepts

### 2.1 Variables and types

C++ is a statically typed language.  
You usually need to write the variable type explicitly:

- `int`
- `float`
- `bool`
- `std::string`

### 2.2 Functions

A function needs to declare:

- Return type
- Parameter types

This makes interfaces clearer and also helps the compiler check your code.

### 2.3 References

References are very common because they help avoid unnecessary copies.  
They are especially important when dealing with large vectors or tensors.

For example:

- `const std::vector<float>& logits`

means:

- Read-only reference
- Do not copy the entire dataset

### 2.4 `std::vector`

You will see it very often in deployment code.  
A simple way to think about it is:

- A dynamic array that is more type-specific than a Python list

---

## 3. First, run a C++ example that is close to a deployment scenario

The example below does something very typical:

- Takes a set of classification scores
- Finds the top-1 category

This is closer to model post-processing than just printing `hello world`.

```cpp
#include <iostream>
#include <vector>
#include <string>

int argmax(const std::vector<float>& logits) {
    int best_idx = 0;
    float best_score = logits[0];

    for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
        if (logits[i] > best_score) {
            best_score = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

int main() {
    std::vector<std::string> labels = {"cat", "dog", "bird"};
    std::vector<float> logits = {1.2f, 0.8f, 2.5f};

    int best_idx = argmax(logits);
    std::cout << "predicted label = " << labels[best_idx] << std::endl;
    return 0;
}
```

Compile command:

```bash
c++ -std=c++17 demo.cpp -o demo
./demo
```

### 3.1 What should you focus on first?

Look at these three parts first:

1. `std::vector<float>`  
   How the data is organized
2. `const std::vector<float>&`  
   Why function parameters often use references
3. `argmax`  
   What the most common deployment post-processing function looks like

### 3.2 Why emphasize references here?

Because if you pass by value directly:

- `std::vector<float> logits`

the entire data will be copied when the function is called.  
In deployment and inference paths, this kind of unnecessary copy is common, and it wastes resources.

---

## 4. How do classes usually appear in deployment code?

### 4.1 Classes are not just for OOP exam questions

In deployment scenarios, classes are often used to represent:

- A model runner
- A tokenizer
- A post-processor

### 4.2 A simple class example

```cpp
#include <iostream>
#include <vector>

class ThresholdFilter {
public:
    explicit ThresholdFilter(float threshold) : threshold_(threshold) {}

    std::vector<float> apply(const std::vector<float>& values) const {
        std::vector<float> kept;
        for (float v : values) {
            if (v >= threshold_) {
                kept.push_back(v);
            }
        }
        return kept;
    }

private:
    float threshold_;
};

int main() {
    ThresholdFilter filter(0.5f);
    std::vector<float> scores = {0.2f, 0.6f, 0.9f};
    std::vector<float> kept = filter.apply(scores);

    for (float v : kept) {
        std::cout << v << std::endl;
    }
}
```

What this example is trying to show is:

- A class can package configuration and behavior together

This is very common in deployment systems.

---

## 5. The most common sticking points for students with a Python background

### 5.1 Compilation

Python runs by interpretation,  
while C++ is usually compiled before execution.

### 5.2 Types need to be written clearly in advance

This may feel verbose at first,  
but it can also help you catch errors earlier.

### 5.3 Memory and copying deserve more attention

In Python, many copy details are not as visible.  
But in C++ and deployment performance paths, they become very important.

---

## 6. Common misunderstandings

### 6.1 Misunderstanding 1: If I can write Python, I should be able to understand C++ without effort

The logic can transfer,  
but concepts like types, references, and ownership still need separate adjustment.

### 6.2 Misunderstanding 2: Basic syntax is useless; just go straight to the engine

Without the basics like variables, functions, classes, and references,  
many deployment SDKs will feel very difficult to read.

### 6.3 Misunderstanding 3: A C++ basics course should be completely unrelated to AI

For this elective, a better approach is:

- Learn directly around the deployment and inference pipeline

---

## Summary

The most important thing in this lesson is not to turn C++ into a standalone specialty,  
but to first build a deployment-friendly foundation:

> **Being able to understand basic types, functions, references, vector, and simple classes is already enough to support your next step into model deployment and inference engines.**

Once these parts are solid, many pieces of code that look “very low-level” later on will feel much less intimidating.

---

## Exercises

1. Change `argmax` to return the top-2 indices, and practice `vector` operations.
2. Try making the `ThresholdFilter` threshold dynamically settable in addition to the constructor parameter.
3. Why is it important in deployment pipelines to “avoid unnecessary copies”?
4. Explain in your own words: why is `const std::vector<float>&` more suitable here than passing by value?
