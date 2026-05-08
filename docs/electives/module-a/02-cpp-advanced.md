---
title: "E.A.2 Advanced C++"
sidebar_position: 2
description: "Learn the advanced C++ ideas most visible in deployment code: ownership, RAII, smart pointers, and interfaces."
keywords: [C++, RAII, smart pointer, virtual, move semantics, threading, deployment]
---

# E.A.2 Advanced C++

![C++ RAII and ownership map](/img/course/elective-cpp-raii-ownership-map-en.webp)

Advanced C++ in deployment is mostly about one question: who owns the resource, and when is it released?

## Run an ownership and interface example

Create `advanced.cpp`:

```cpp
#include <iostream>
#include <memory>

struct Engine {
    virtual ~Engine() = default;
    virtual float run(float input) = 0;
};

struct CpuEngine : Engine {
    float run(float input) override {
        return input * 0.84f;
    }
};

class Session {
public:
    explicit Session(std::unique_ptr<Engine> engine)
        : engine_(std::move(engine)) {}

    void predict() {
        std::cout << "cpu_score=" << engine_->run(1.0f) << "\n";
        std::cout << "session_done\n";
    }

private:
    std::unique_ptr<Engine> engine_;
};

int main() {
    Session session(std::make_unique<CpuEngine>());
    session.predict();
    return 0;
}
```

Run it:

```bash
c++ -std=c++17 advanced.cpp -o advanced
./advanced
```

Expected output:

```text
cpu_score=0.84
session_done
```

## What to notice

| C++ idea | Deployment meaning |
|---|---|
| `Engine` interface | Business code can switch CPU/GPU/runtime backends |
| `std::unique_ptr` | Only one owner controls the engine resource |
| `std::move` | Ownership is transferred into `Session` |
| destructor through `virtual ~Engine()` | Cleanup is safe through the interface |
| RAII | Resource lifetime follows object lifetime |

## Practice change

Add a second engine:

```cpp
struct FastEngine : Engine {
    float run(float input) override {
        return input * 0.91f;
    }
};
```

Then replace `CpuEngine` with `FastEngine`. The rest of `Session` should not change.

## Pass check

You pass this lesson when you can explain why `Session` owns the engine, why `unique_ptr` is safer than a raw pointer here, and how an interface lets one deployment path swap runtime backends.
