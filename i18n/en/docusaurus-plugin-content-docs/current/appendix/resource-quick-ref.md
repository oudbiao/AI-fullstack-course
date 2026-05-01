---
title: "Learning Resources Quick Reference"
sidebar_position: 6
description: "A quick reference page for common environment commands, model approaches, evaluation metrics, and key points for RAG and Agent design."
---

# Learning Resources Quick Reference

![AI Project Quick Reference Overview](/img/course/appendix-project-quick-reference-map.png)

![AI Project Quick Reference Troubleshooting Index](/img/course/appendix-quick-ref-debug-index-map.png)

:::tip Reading Guide
This quick reference is best for “locating things quickly while working on a project.” When reading it, first determine whether the problem is about the environment, data, training, evaluation, RAG, Agent, Prompt, or frontend, then jump to the relevant checklist. Don’t start from the beginning and reread the whole book.
:::

This page is not a complete tutorial. It is a “look it up quickly when needed” tool page. It’s a good idea to keep it in your browser bookmarks or nearby your project for quick access.

## 1. Quick Reference for Environment and Repository Commands

### 1.1 Python Environment

```bash
conda create -n ai-course python=3.11 -y
conda activate ai-course
pip install -r requirements-course-core.txt
```

If you want to run the AI examples in the later sections, also install:

```bash
pip install -r requirements-course-ai.txt
```

### 1.2 First Confirm Which Environment You Are In

```bash
python --version
which python
pip --version
pip list
```

### 1.3 Documentation Site Commands

```bash
npm install
npm run start
npm run build
```

## 2. First Set of Troubleshooting Commands

For most environment issues, run the following first:

```bash
pwd
ls
python --version
which python
pip --version
pip list
```

If you are using a GPU:

```bash
nvidia-smi
```

## 3. Which Baseline to Choose First for Common Tasks

| Task | Recommended baseline to try first | Why |
|---|---|---|
| Tabular classification / regression | Linear models, tree-based models | Simple, fast, easy to explain |
| Text classification | `TF-IDF + LogisticRegression` | A strong baseline and easy to understand |
| Image classification | Transfer learning | More realistic for small datasets |
| Named entity recognition | Rules / dictionary baseline, then sequence models | First confirm the task definition |
| Document Q&A | Keyword retrieval or BM25, then RAG | First see whether retrieval is worth it |
| Enterprise knowledge base | Single-turn Q&A + cited sources | Start with a verifiable closed loop |
| Agent tool calling | Single Agent + a small number of tools | Start stable, then add complexity |

Many projects get stuck not because the model is too weak, but because the baseline was never established.

## 4. Quick Reference for Common Evaluation Metrics

### 4.1 Classification Tasks

| Metric | When to use | Reminder |
|---|---|---|
| Accuracy | When classes are relatively balanced | Can be misleading on imbalanced data |
| Precision | When false positives are costly | Focus on “of the predicted positives, how many are correct” |
| Recall | When false negatives are costly | Very important in medical and safety-related tasks |
| F1 | When you need to balance Precision / Recall | A commonly used overall metric |

### 4.2 Regression Tasks

| Metric | When to use | Reminder |
|---|---|---|
| MAE | When you want to see average absolute error | Relatively robust to outliers |
| MSE / RMSE | When large errors should be penalized more | More sensitive to outliers |
| R² | When you want to see overall fit/explained variance | Do not use it alone |

### 4.3 Retrieval and Question Answering

| Metric | Purpose |
|---|---|
| Hit@K | Whether the correct document appears in the top K results |
| MRR | Whether the correct answer ranks near the top |
| Citation accuracy | Whether the answer is truly based on the retrieved content |
| Human evaluation | Whether the final usability meets the bar |

## 5. Signals to Watch During Training

| Phenomenon | What you should suspect first |
|---|---|
| Training loss does not decrease | Learning rate, labels, loss function, input format |
| Training is good but validation is poor | Overfitting, data leakage, distribution mismatch |
| Accuracy does not change | Features are too weak, labels are wrong, or the model is not learning at all |
| Out of memory on GPU | Batch too large, inputs too long, model too big |
| Results are extremely unstable | Dataset too small, too much randomness, experiments without fixed seeds |

## 6. What to Check First When Doing RAG

### 6.1 Minimal Closed Loop

1. Can the documents be split correctly?
2. Can retrieval bring back the right chunks?
3. Is the source included during generation?
4. Does the answer truly use the retrieved content?

### 6.2 Common Pitfalls

| Problem | First thing to check |
|---|---|
| The answer sounds made up | Retrieval did not bring back relevant chunks |
| The answer is vague | The chunk size is too large and the information is too scattered |
| It often answers restricted content incorrectly | Missing permission filtering |
| It is too slow | Measure the retrieval layer, model layer, and network calls separately |
| Cost is too high | Cache, shorten context, reduce unnecessary calls |

## 7. What to Check First When Doing Agent Work

### 7.1 Start with a Simple Structure

A more stable progression is usually:

1. Single-turn Q&A
2. Single Agent + single tool
3. Single Agent + multiple tools
4. Stateful workflow
5. Multi-Agent collaboration

### 7.2 Five Things to Check in the Tool Layer

1. Are the tool descriptions clear?
2. Is the parameter schema strict?
3. Can errors be returned properly?
4. Are high-risk operations restricted?
5. Are there logs for tracing?

## 8. Quick Reference for Prompt Design

A more stable Prompt usually includes at least:

- Role: what you want the model to act as
- Task: what it specifically needs to do
- Input: what it will receive
- Output: what format it must use
- Constraints: what it should not do

A simple template:

```text
You are a ____.
Your task is ____.
The input will include ____.
Please output in the following format ____.
If the information is insufficient, clearly say so. Do not make things up.
```

## 9. Common Python Debugging Actions

### 9.1 Inspect the Input

```python
print(type(x))
print(len(data))
print(data[:2])
```

### 9.2 Inspect Intermediate Results

```python
print("shape:", tensor.shape)
print("mean:", tensor.mean().item())
print("min/max:", tensor.min().item(), tensor.max().item())
```

### 9.3 Inspect Predictions

```python
for text, pred in list(zip(texts, preds))[:5]:
    print(text, "->", pred)
```

## 10. A Minimal Training Loop Skeleton

```python
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
```

The 6 lines most worth understanding here are:

- Where the data comes from
- What the forward pass does
- How the loss is defined
- How gradients are backpropagated
- When parameters are updated

## 11. When the Project Gets Stuck, Come Back to This Map

| Your current problem | What to add first |
|---|---|
| You can’t understand the code | Python and debugging |
| The results are unstable | Data and evaluation |
| You don’t know which model to choose | Baselines and task definition |
| You can’t build a complete system | Minimal closed loop, interface, and state design |
| You’ve learned a lot but still can’t remember it | Small projects, review, and output |

## 12. When to Check This Quick Reference and When to Go Back to the Main Text

Good times to check the quick reference:

- You forgot a command
- You forgot the difference between evaluation metrics
- You forgot the first step in a project
- You want to quickly locate where a problem roughly is

Good times to go back to the main text:

- You do not even know what a concept is
- You need systematic understanding of a complete main thread
- You need to write a project report or give an explanation
- You need to know why, not just how

This page works best together with the main text: first use the main text to build understanding, then use the quick reference to move faster.
