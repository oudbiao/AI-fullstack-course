---
title: "6.8.3 Project: Text Sentiment Analysis"
sidebar_position: 2
description: "Walk through a complete closed loop around a real, showcaseable sentiment analysis project, from label boundaries and baseline to error analysis and delivery format."
keywords: [sentiment analysis project, text classification, baseline, negation, sarcasm, NLP]
---

# 6.8.3 Project: Text Sentiment Analysis

:::tip Section focus
A sentiment analysis project is great for a portfolio not because it is the fanciest, but because it is very good for training “project judgment”:

- How should labels be defined?
- How should the baseline be built?
- How should errors be explained?
- How should results be presented?

The goal of this section is not to pile on complex models, but to truly complete a small project end to end.
:::

## Learning Objectives

- Learn how to design stable label boundaries for sentiment analysis tasks
- Learn how to build an interpretable baseline and understand its results
- Learn how to turn error analysis into a project highlight, not an after-the-fact patch
- Learn how to package a small NLP project as a deliverable

---

## First, Narrow the Project Topic

### The safest starting point is binary classification

Start with:

- positive
- negative

Instead of beginning with:

- positive / neutral / negative / irony / mixed

### Why is binary classification good for practice?

Because:

- The labels are clearer
- The data is easier to prepare
- The errors are easier to analyze

### A portfolio-friendly topic

For example:

> **Build a “course review sentiment analyzer” that determines whether a comment is positive or negative.**

This topic is especially suitable because the user text, label boundaries, and business meaning are all relatively clear.

---

## What Does the Minimum Closed Loop Look Like?

1. Define label boundaries
2. Prepare a small labeled dataset
3. Build a baseline
4. Perform error analysis
5. Design a minimal inference API or demo page

If these 5 steps are all clear, your project is usually already much closer to a portfolio-grade project than “just training a model.”

![Sentiment analysis project closed loop](/img/course/ch06-project-sentiment-analysis-loop-en.png)

:::tip How to read this diagram
Read it as an error-driven loop: define the label boundary first, build a small baseline, run predictions, group wrong cases by error type, and only then decide whether to add rules, data, or a stronger model.
:::

## Recommended Progression Order

For beginners, the safer order is usually:

1. Write down the label definitions first
2. Then build the simplest baseline
3. Then add a traditional ML baseline
4. Finally, consider a stronger deep learning model

This way, you won’t get pulled away by model complexity at the very beginning.

---

## Start with a Minimal Baseline Project

To keep the logic very clear, we’ll start with a keyword-counting baseline.
It is certainly not strong, but it is very suitable for explaining the project loop.

```python
from collections import Counter


def tokenize(text):
    cleaned = text.lower()
    for char in ",.!?":
        cleaned = cleaned.replace(char, "")
    return cleaned.split()


train_data = [
    ("This course explains things very clearly", "positive"),
    ("There are lots of examples, so it is easy to learn", "positive"),
    ("The content is too messy", "negative"),
    ("It is explained too quickly, I can't understand it", "negative"),
]

test_data = [
    ("This course is really clear", "positive"),
    ("The content is a bit messy", "negative"),
    ("Lots of examples but explained too quickly", "negative"),
]


positive_words = Counter()
negative_words = Counter()

for text, label in train_data:
    tokens = tokenize(text)
    if label == "positive":
        positive_words.update(tokens)
    else:
        negative_words.update(tokens)

# Add a few transparent seed words so the negation examples have sentiment words to flip.
positive_words.update(["clear"] * 3 + ["recommended"] * 4 + ["systematic"] * 6)
negative_words.update(["messy"] * 2 + ["confusing"] * 2)


def predict(text):
    score = 0
    for token in tokenize(text):
        score += positive_words[token]
        score -= negative_words[token]
    return "positive" if score >= 0 else "negative", score


results = []
for text, gold in test_data:
    pred, score = predict(text)
    results.append({"text": text, "gold": gold, "pred": pred, "score": score})
    print(results[-1])
```

### Why is this baseline educationally valuable?

Because it is easy to explain:

- Why it was judged positive
- Why it was judged negative
- What `tokenize` does: it turns raw text into word units that the baseline can count
- Why a tiny seed lexicon is useful: it keeps the demo small while giving the rule clear sentiment words to operate on

This lets you do real error analysis, instead of staring at just one number.

### Add a minimal “negation flip” upgrade

One of the most typical error patterns in sentiment analysis is:

- There is a positive word
- But a negation word in front flips it

For example:

- “not recommended”
- “not clear”
- “not worth it”

The tiny version below is not an industrial solution,
but it is very good for beginners to experience for the first time:

- **Why a rule-based patch can directly change the distribution of errors**

```python
negation_words = {"not", "no", "never"}


def predict_with_negation(text):
    score = 0
    negate_next_sentiment_word = False

    for token in tokenize(text):
        if token in negation_words:
            negate_next_sentiment_word = True
            continue

        token_score = positive_words[token] - negative_words[token]

        # If a negation word appeared, flip the next sentiment-bearing token.
        if negate_next_sentiment_word and token_score != 0:
            token_score *= -1
            negate_next_sentiment_word = False

        score += token_score

    return "positive" if score >= 0 else "negative", score


extra_cases = [
    ("Not recommended for this course", "negative"),
    ("It is not clear", "negative"),
    ("There are quite a few examples, but it is not systematic", "negative"),
]

for text, gold in extra_cases:
    pred, score = predict_with_negation(text)
    print({"text": text, "gold": gold, "pred": pred, "score": score})
```

The teaching value of this code is not that the rule is strong,
but that it helps you clearly see for the first time:

- Why the baseline makes mistakes
- What kind of mistakes a specific patch can fix
- How error analysis can genuinely drive the solution upgrade

The important engineering detail is that the rule and the tokenizer must use the same granularity. If the baseline counts words, then `not` should also be detected as a word. If the baseline counts characters, a word-level negation rule will silently stop working.

---

## What Really Makes the Project Stronger Is Error Analysis

### First, pull out the wrong cases

```python
errors = [row for row in results if row["gold"] != row["pred"]]
print(errors)
```

### Common error types

For sentiment analysis, the ones most worth looking at separately are:

- Negation
  For example, “not bad” and “not recommended”
- Sarcasm
  For example, “Great, it broke again”
- Mixed sentiment
  For example, “The content is great, but it is too difficult”

### Why is error analysis so valuable?

Because it can directly tell you what to do next:

- Add more data
- Adjust the label standard
- Upgrade the model

### Create a minimal error bucket table for yourself

When beginners do sentiment analysis projects, it is easy to only say:

- “The model got a few examples wrong”

But it is more valuable to bucket them first:

```python
error_buckets = {
    "negation": [],
    "sarcasm": [],
    "mixed_sentiment": [],
    "other": [],
}

examples = [
    ("Not recommended for this course", "negative", "positive"),
    ("Great, it got stuck again", "negative", "positive"),
    ("The content is great, but the pace is too fast", "negative", "positive"),
]

for text, gold, pred in examples:
    if "not" in text.lower():
        error_buckets["negation"].append(text)
    elif "great" in text.lower() and "again" in text.lower():
        error_buckets["sarcasm"].append(text)
    elif "but" in text.lower():
        error_buckets["mixed_sentiment"].append(text)
    else:
        error_buckets["other"].append(text)

for k, v in error_buckets.items():
    print(k, len(v), v)
```

This table is excellent for showing in a project,
because it immediately tells others:

- You are not just reporting scores
- You understand that errors have types
- Your next step is based on evidence

---

## How Can This Project Be Pushed One Step Further Toward Portfolio Quality?

### Add a traditional strong baseline

For example:

- TF-IDF + LogisticRegression

Then your project will have at least:

- A rule-based baseline
- A traditional ML baseline

### Add a deep learning baseline

For example:

- embedding + pooling
- BERT classification

### Don’t show only the final score

It is highly recommended to show:

- Label definitions
- Baseline comparison
- Typical error cases
- Hardest negative examples

This will make the project feel much more complete.

### A presentation order that feels more like a real project

If you turn this into a portfolio page,
the following order is recommended:

1. Task definition and label boundaries
2. Baseline methods
3. Baseline comparison table
4. Error buckets
5. What you upgraded for which type of error
6. The final retained solution and why

Then what others see is not just “I built a sentiment classifier,”
but rather:

- You really know how a small NLP project should move from baseline to portfolio quality

---

## The Most Common Pitfalls

### Inconsistent label standards

This is the first major pitfall in many sentiment projects.

### Only looking at accuracy

Without looking at where the errors are, it is hard to truly improve.

### Chasing the most complex model from the start

Without a baseline, it is hard to explain what the complex model improved.

---

## What Should Be Added at Delivery Time

- A label definition table
- A baseline comparison table
- A set of typical error cases
- A short judgment on the next upgrade path

---

## Summary

The most important thing in this section is to build a project habit:

> **The most valuable part of a sentiment analysis project is not how complex the model is, but whether you can explain the label boundaries, baseline, error analysis, and upgrade path as a complete closed loop.**

As long as you do that well, even if the topic is small, it will feel very much like a portfolio-grade course project.

## What You Should Take Away from This Section

- What matters most in sentiment analysis is not a complex model, but clear label boundaries and error analysis
- A simple baseline is very educational as long as it is interpretable enough
- What really separates project quality is often whether you can turn error cases into next actions



## Suggested Version Roadmap

| Version | Goal | Delivery Focus |
|---|---|---|
| Basic | Run through the minimal closed loop | Can accept input, process it, and output results, while keeping a set of examples |
| Standard | Form a project that can be shown | Add configuration, logging, error handling, README, and screenshots |
| Challenge | Approach portfolio quality | Add evaluation, comparison experiments, failure-case analysis, and a next-step roadmap |

It is recommended to finish the basic version first. Do not try to make everything big and complete from the start. Each time you improve a version, write into the README “what capability was added, how it was verified, and what problems remain.”

## Exercises

1. Design 12 course reviews on your own and assign positive/negative labels.
2. On the baseline, manually add a “negation flip rule” and see whether it can fix a certain type of error.
3. Think about why sentiment analysis is especially suitable for showing error analysis.
4. If you wanted to expand this project into a three-class task, would you change the label standard first or switch the model first? Why?
