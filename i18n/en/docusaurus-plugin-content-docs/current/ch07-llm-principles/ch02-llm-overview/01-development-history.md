---
title: "2.2 Development History of Large Models"
sidebar_position: 5
description: "Trace the main evolution of language models from rule-based systems, statistical language models, and RNNs to Transformer and large language models."
keywords: [LLM history, language model, n-gram, RNN, Transformer, GPT]
---

# Development History of Large Models

![Large Model Development Timeline](/img/course/llm-history-timeline-en.png)

## Learning Objectives

After completing this section, you will be able to:

- Understand that large models did not appear out of nowhere, but are the result of the long evolution of NLP
- Explain the key turning points from rule-based systems to Transformer
- Understand why Transformer truly propelled the era of large models
- Experience how early language models worked through a small example

---

## 1. First, Remember the Big Picture

The development of language models can be roughly divided into these 5 stages:

1. The rule-based system era
2. The statistical language model era
3. The neural network language model era
4. The Transformer era
5. The large model and instruction alignment era

You can think of it as an “autocomplete” path that keeps upgrading:

- First, it relied on manual rules
- Then, it relied on statistical frequency
- Then, neural networks learned representations
- Then, Transformer learned long-range context
- Finally, large-scale training and alignment techniques made it more practical

### 1.1 If You Want to Be Drawn Into This History First, Remember Three Scenes

For beginners, the most memorable parts of this history are usually not the concepts, but these three scenes:

1. Early systems were like a thick rule manual  
   They could do the job, but they depended heavily on people writing rules.

2. When Transformer appeared, it felt like a public announcement  
   The title `Attention Is All You Need` itself seemed to say:  
   “Maybe sequence modeling no longer needs to be tied to RNN.”

3. After GPT-3, many people intuitively felt for the first time  
   that doing only next token prediction, when scaled up enough, could really give rise to what looks like general capability.

So the most fascinating part of this history is:

> **Many abilities that look obvious today were not added gradually back then, but emerged through several shifts in direction.**

---

## 2. Rule-Based Systems: The Earliest “Artificial Language Intelligence”

In early NLP, people often wrote rules by hand:

- If the sentence contains “book a ticket,” classify it as travel
- If it contains “weather,” classify it as weather inquiry

The advantages are:

- Simple
- Interpretable
- Quick to start for small tasks

The disadvantages are:

- Tedious to write
- Easy to break when the scenario changes
- Hard to cover complex expressions

It is like training a new customer service agent using only a thick “conversation rules manual.”  
It works, but the ceiling is not high.

### 2.1 Why Did Many People Later Feel This Path Was “Doomed to Be Exhausting”?

Because it depends too much on humans first breaking the world down for the machine.  
Once the scenario becomes complex, the maintenance cost rises quickly:

- If the expression changes, the rule may miss it
- If the scenario expands, the rulebook gets thicker and thicker

So the most typical feeling around early rule-based systems was:

- They really did work
- But people increasingly realized that this path was hard to scale to the open world

---

## 3. Statistical Language Models: Starting to Predict the Next Word by “Frequency of Occurrence”

The core idea of statistical language models is:

> **What word comes next depends on what words appeared before it.**

For example:

- “Today the weather is very” is likely followed by “good”
- “I like to eat” may be followed by “noodles” or “rice”

This is the classic `n-gram` idea.

### A Minimal Runnable Example: Bigram Language Model

```python
from collections import defaultdict, Counter

corpus = [
    "I like learning AI",
    "I like learning Python",
    "You like learning NLP",
    "I like doing projects"
]

next_word_counter = defaultdict(Counter)

for sentence in corpus:
    tokens = sentence.split()
    for current_word, next_word in zip(tokens[:-1], tokens[1:]):
        next_word_counter[current_word][next_word] += 1

def suggest_next(word):
    candidates = next_word_counter[word]
    if not candidates:
        return []
    return candidates.most_common()

print("Common words after “I”:", suggest_next("I"))
print("Common words after “like”:", suggest_next("like"))
print("Common words after “learning”:", suggest_next("learning"))
```

This idea is already very close to “predict the next word.”  
But its limitations are also obvious:

- It can only look at very short context
- As word combinations increase, the statistics become sparse
- It cannot truly understand semantics

### 3.1 Why Is This Generation of Methods Still Very Important?

Because it was the first time NLP truly moved from:

- Humans writing rules

to:

- Letting data tell the system what is more common and more reasonable

In other words, although it does not look powerful by today’s standards,  
it was very much a stepping stone:

> **It first turned language problems from “rule problems” into “statistical problems.”**

---

## 4. Neural Network Language Models: Starting to Learn Representations

Later, people were no longer satisfied with “just counting word frequencies,” and began letting neural networks learn representations of words and context.

This brought several important changes:

- Words were no longer just discrete IDs; they had vector representations
- Models could start learning semantic similarity
- Language modeling no longer relied only on hard statistics

Important directions in this stage included:

- Word2Vec / GloVe
- RNN
- LSTM / GRU

### What Did They Solve?

For example:

- `king` and `queen` became closer in representation
- Sentences like “It rained today, I didn’t bring an umbrella, so I got wet” could use longer context

But the problems were still there:

- RNNs struggled with very long sequences
- Training efficiency was not high
- Long-range dependencies remained difficult

### 4.1 Why Did This Generation Make Many People Excited Again?

Because starting from here, language models were no longer just:

- Counting which words commonly follow which words

Instead, they began to truly learn:

- Relationships between words
- Semantic structure in context

So the appeal of this generation was:

> **Language was no longer just a string of symbols; for the first time, it really entered a representation space.**

---

## 5. Transformer: The Key Turning Point That Truly Changed the Game

The core breakthrough of Transformer is the attention mechanism.

You can think of it like this:

> When reading a sentence, instead of passing memory forward step by step in order, you can directly look at the relationships between the current word and all other words.

This is like:

- RNNs reading a book sentence by sentence, remembering as they go
- Transformer spreading the whole page out and circling the key points all at once

Its advantages are crucial:

- More suitable for parallel training
- Better at handling long context
- Easier to scale to very large sizes

This is exactly why models like GPT and BERT were able to rise so quickly later on.

### 5.1.1 Why Did This Paper Stick in People’s Minds So Quickly?

Part of the reason is, of course, the technique itself,  
but another part is that the title really gives off a strong “era shift” feeling:

- `Attention Is All You Need`

It is not as conservative as traditional papers,  
but carries a very strong judgment:

- Maybe the whole main line of sequence modeling needs to change

History later proved that this was not just a slogan,  
but something that genuinely rewrote the foundation of the large model era that followed.

---

## 6. The Pretraining Model Era: First Read Massive Amounts of Text, Then Do Specific Tasks

The core idea of this stage is:

1. First perform general pretraining on massive corpora
2. Then transfer to specific tasks

This is much more efficient than training a model from scratch for every task.

Typical representatives include:

- BERT: more focused on understanding
- GPT: more focused on generation
- T5: unified text-to-text framework

This brought a very important change:

> Models began to have increasingly strong “general language abilities.”

---

## 7. The Large Language Model Era: Scale Brings a Leap in Capability

When model parameters, data volume, and compute resources increase greatly, we get what we now call large language models (LLMs).

Key words in the large model era:

- Larger parameter scale
- Longer context
- Stronger generation ability
- Instruction-following ability
- Tool-using ability

In this stage, the model is no longer just “a classifier,” but more and more like a general-purpose text interface:

- Write code
- Summarize documents
- Translate
- Answer questions
- Reason
- Call tools

### 7.1 Why Did GPT-3 Become a “Turning Point in the Atmosphere” for Many People?

Because GPT-3 made many people feel not just that:

- The model got bigger

but that:

- Some abilities started to feel less like isolated features and more like a more general language interface

It made many people strongly realize for the first time that:

- Perhaps model scale, data scale, and training paradigm, when combined, can push “language models” into something more like platform-level capability

So for many people, GPT-3 was not just a paper milestone,  
but more like:

> **The emotional milestone that said “the large model era has really begun.”**

---

## 8. Why “Bigger” Alone Is Not Enough

After models became larger, they were stronger, but problems also appeared:

- They may not understand user intent
- They may not answer as requested
- They may produce harmful or unstable content

So later, an entire set of alignment techniques was developed:

- Instruction tuning
- Preference learning
- RLHF
- Safety alignment

It is like this:

> The model first becomes smarter by reading massive amounts of text, then learns to be more like a cooperative assistant through human feedback.

### 8.1 Why Did This Step Make Many People Suddenly Realize That “Training Objectives Are Not the Same as Product Objectives”?

Because after GPT-3, everyone had already seen that:

- Models can be very powerful
- But being “powerful” does not automatically mean “useful”

It may:

- Not follow instructions well enough
- Not be stable enough
- Not match human expectations well enough

So what the alignment stage really brought was not just another term,  
but a very important industry consensus:

> **Model capability and model behavior are not the same thing.**

---

## 9. Why Do Today’s Large Models Belong to More Than Just NLP?

Because Transformer and large model methods later expanded to more modalities:

- Images
- Speech
- Video
- Multimodal data

Further on, we got:

- RAG
- Agent
- Tool calling
- Multimodal Q&A

So the “development history of large models” is not just a small branch of NLP, but the prehistory of the entire AI application engineering landscape that followed.

---

## 10. A Quick Timeline Table

| Stage | Core Idea | Limitation |
|---|---|---|
| Rule-based systems | Humans write rules | Poor generalization, hard to maintain |
| Statistical language models | Predict with frequency | Sparse data, short context |
| RNN/LSTM | Model sequences with neural networks | Long-range dependencies are still hard |
| Transformer | Use attention to capture global relationships | Training cost is high |
| Large language models | Scaled pretraining + alignment | Cost, safety, hallucination issues |

---

## Summary

The most important thing in this section is not memorizing dates, but understanding one main line:

> **Language models have always been solving the same problem: how to better use context to predict and understand language.**

Rules were not flexible enough, statistics were not deep enough, RNNs could not look far enough, and Transformer was what truly pushed large-scale language modeling into today’s large model era.

---

## Exercises

1. Modify the `corpus` above by adding more sentences, and observe how the `n-gram` prediction results change.
2. Think about it: why is it hard to understand a long article using only statistics of “current word -> next word”?
3. Explain in your own words: what exactly makes Transformer stronger than RNN?
