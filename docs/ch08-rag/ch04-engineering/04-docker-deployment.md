---
title: "4.5 Containerization and Deployment"
sidebar_position: 20
description: "From why containerization matters, to the core structure of a Dockerfile, to how Compose starts services, understand how an LLM application evolves from a local script into a deployable service."
keywords: [Docker, containerization, deployment, Dockerfile, Compose, service deployment]
---

# Containerization and Deployment

:::tip Where This Section Fits
Many projects get stuck here:

- It runs locally
- It breaks on another machine
- Team members have inconsistent environments
- Dependency versions become a mess after going live

The core value of containerization is to take your application from:

> “It runs on my computer”

to:

> “It runs reliably and reproducibly in an agreed environment”.
:::

## Learning Objectives

- Understand why LLM applications are especially well-suited to containerization
- Read the key structure of a minimal Dockerfile
- Understand the core concepts of images, containers, ports, and environment variables
- Read a small Docker Compose startup example
- Understand that containerization is not the end of deployment, but the starting point

## Beginner terminology bridge

Docker becomes much less intimidating once the nouns are separated:

| Term | Beginner meaning | Why it matters |
|---|---|---|
| `image` | A packaged runtime template, like a recipe plus ingredients | You build it once and run containers from it |
| `container` | A running instance created from an image | This is the actual process serving requests |
| `Dockerfile` | The build recipe for an image | It records the base image, dependencies, files, and startup command |
| `port` | The doorway where a service listens for requests | `-p 8000:8000` maps the host port to the container port |
| `environment variable` | Configuration injected from outside the code | API keys, model names, and runtime modes should not be hardcoded |
| `Compose` | A tool for starting multiple related containers together | Useful when the app needs a vector database, Redis, or Postgres |

The core idea is not “learn Docker commands by heart,” but “make the runtime environment reproducible.”

---

## 1. Why containerize?

### 1.1 What is the biggest hidden risk of a local script?

When you can run a project locally, it often depends on many implicit conditions:

- Python version
- Package versions
- System dependencies
- Environment variables
- Startup command

Once you change the person, the machine, or the server, these conditions can easily cause problems.

### 1.2 What does containerization actually solve?

The core value of containerization is:

> **Package the application together with the runtime environment it depends on.**

This lets you reproduce more reliably:

- What was installed
- Which versions were used
- Which command was used to start it

This is especially important for LLM applications, because they often depend on:

- Web frameworks
- Model services
- Vector databases
- System tools

---

## 2. What are images and containers?

### 2.1 A very practical analogy

- **Image**: like a recipe + ingredient kit
- **Container**: the actual dish made from that recipe

In other words:

- An image is a static template
- A container is a running instance

### 2.2 Why is this distinction important?

Because during deployment, you usually:

1. Build the image first
2. Then start the container

If you do not clearly understand this order, Docker commands will feel confusing for a long time.

![Docker image, container, and Compose deployment diagram](/img/course/ch08-docker-image-container-compose-map-en.png)

:::tip Reading the Diagram
An image is a reproducible runtime template, a container is a running instance, and Compose is responsible for starting multiple services together. For LLM applications, you also need to include environment variables, health checks, vector databases, and logs in the deployment diagram.
:::

---

## 3. What does a minimal Dockerfile look like?

### 3.1 First, look at the complete example

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
```

### 3.2 What does each line do?

- `FROM`
  - Choose the base image

- `WORKDIR`
  - Set the working directory

- `COPY requirements.txt .`
  - Copy in the dependency file

- `RUN pip install ...`
  - Install dependencies

- `COPY . .`
  - Copy the project code in as well

- `EXPOSE 8000`
  - Indicate the port the service listens on

- `CMD`
  - The default command executed when the container starts

This is the core skeleton of a Dockerfile.

---

## 4. First prepare a small app that can actually run

### 4.1 Minimal Python service

To make the Docker deployment example more concrete, let's first write a very simple `app.py`.

```python
# app.py
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"message": "hello from llm app"}).encode())

server = HTTPServer(("0.0.0.0", 8000), Handler)
print("serving on 8000")
server.serve_forever()
```

### 4.2 Why start with this?

Because containerization is not about talking about Dockerfiles in the abstract,
but about understanding them around a real running application.

---

## 5. Then containerize it

### 5.1 Matching `requirements.txt`

This minimal service does not depend on any third-party packages, so `requirements.txt` can be empty, or you may even not need it.
But to stay close to a real project, we will keep the structure.

```text
# requirements.txt
```

### 5.2 Corresponding Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"]
```

### 5.3 Run commands

```bash
docker build -t mini-llm-app .
docker run -p 8000:8000 mini-llm-app
```

Then visit:

- `http://localhost:8000/`
- `http://localhost:8000/health`

and you will see the returned results.

This is the smallest containerization loop.

---

## 6. Why are environment variables important?

LLM applications often have configurations like these:

- API Key
- Model name
- Vector database address
- Runtime mode

These are usually not hardcoded in the code; environment variables are a better fit.

### 6.1 A minimal example

```python
import os

model_name = os.getenv("MODEL_NAME", "demo-model")
port = int(os.getenv("PORT", "8000"))

print("MODEL_NAME =", model_name)
print("PORT =", port)
```

### 6.2 How do you pass environment variables in Docker?

```bash
docker run -p 8000:8000 -e MODEL_NAME=qwen-demo mini-llm-app
```

This step is very important, because real deployment almost always relies on configuration injection.

---

## 7. Why is Compose so commonly used?

### 7.1 Because real projects usually have more than one service

An LLM application may also need to work with:

- Web service
- Vector database
- Redis
- Postgres

If you write `docker run` by hand for each one, things quickly become messy.

### 7.2 A minimal Compose example

```yaml
version: "3.9"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      MODEL_NAME: demo-model
```

Startup command:

```bash
docker compose up --build
```

This is why Compose is very useful for local development and small-scale deployments.

---

## 8. Containerization does not mean deployment is finished

This is a very common misunderstanding.

### 8.1 Containerization solves packaging and the runtime environment

But going live still requires considering:

- Logs
- Health checks
- Resource limits
- Automatic restarts
- Canary releases
- Reverse proxies

### 8.2 A very important health check idea

An endpoint like:

- `/health`

is very valuable.
Because deployment systems usually need to know:

> Is this container alive right now, and can it accept requests?

---

## 9. Common mistakes beginners often make

### 9.1 Putting everything into one huge image

The image becomes bloated.

### 9.2 No health check

You do not know when the service is broken.

### 9.3 Hardcoding configuration in the code

Things break easily when you switch environments.

### 9.4 Thinking containerization automatically makes things scalable

It does not.
Containerization is only the first step; orchestration, monitoring, and operations come next.

---

## Summary

The most important thing in this section is not memorizing Docker commands, but understanding:

> **The core value of containerization is standardizing “application + dependencies + startup method” together, so deployment becomes a reproducible process instead of personal machine experience.**

Once you make this step solid, service orchestration and production operations will have a foundation.

---

## Exercises

1. Use the `app.py` and Dockerfile from this section to actually build a minimal image locally.
2. Add another environment variable to the service, such as `APP_MODE=dev`.
3. Think about this: why is the `/health` endpoint important for deployment systems?
4. Explain in your own words: why is containerization the starting point of deployment, not the end?
