---
title: "1.2 Tokenization and Tokenizer"
sidebar_position: 1
description: "Start with “why models can’t read text directly,” and understand the trade-offs between word-level, character-level, and subword-level tokenization, as well as why padding, truncation, and special tokens matter in practice."
keywords: [tokenizer, tokenization, subword, BPE, wordpiece, padding, truncation]
---

# Tokenization and Tokenizer

![Tokenizer Subword Splitting Flowchart](/img/course/tokenizer-subword-flow.png)

:::tip Where This Section Fits
When many people first learn about large models, they put all their attention on model architecture.  
But before text can actually be sent into a model, there is another unavoidable step:

> **What units should text be split into so the model can process it?**

That is what a tokenizer does.

If this step is not clear, later on these terms will all feel like a pile of jargon:

- `input_ids`
- `attention_mask`
- context length
- token cost

The goal of this lesson is to bring the tokenizer back from a “black-box tool” to its most fundamental role.
:::

## Learning Objectives

- Understand why models cannot directly consume raw strings
- Distinguish the key differences between character-level, word-level, and subword-level tokenization
- Understand the role of special tokens, padding, and truncation in practice
- See through a runnable example how a tokenizer turns text into `input_ids`

---

## 1. Why Can’t a Model Read Text Directly?

### 1.1 What a model ultimately processes is numbers, not characters themselves

Neural networks fundamentally can only process numerical tensors.  
But what humans feed into a model is usually:

- Chinese sentences
- English paragraphs
- code
- mixed punctuation and emojis

A model does not recognize the visual shape of words like “refund”, “password”, or “hello”.  
It first needs two steps:

1. Split text into tokens
2. Map tokens to integer IDs

So what a tokenizer does is not just “simple word splitting”;  
it is:

> **the first interface that turns human language into a discrete symbol sequence the model can process.**

### 1.2 An analogy: translating an article into numbered building blocks for a machine

You can think of a tokenizer as a warehouse manager.

Raw text is like a large pile of unsorted goods.  
The tokenizer must first decide:

- what size each building block should be
- what number each block should have

After that, the model no longer sees “an article,” but something like:

- `[101, 2057, 2024, 2172, 102]`

If the blocks are split too finely, the sequence becomes very long;  
if they are split too coarsely, many words will be unknown.

---

## 2. The Three Most Common Splitting Methods

### 2.1 Character-level: safest, but sequences become long

The simplest idea is:

- treat one character as one token

The advantages are:

- almost no OOV problem
- even unseen words can still be represented by splitting them

The disadvantages are:

- sequences become long
- semantic granularity is too fine
- the model must spend more layers combining meanings

For example, in English:

- “refund policy” -> `r / e / f / u / n / d / p / o / l / i / c / y`

### 2.2 Word-level: intuitive meaning, but serious OOV issues

Another approach is:

- treat one complete word as one token

The advantages are:

- natural granularity
- intuitive word meaning

The disadvantages are:

- many new words, spelling variants, and proper nouns
- the vocabulary can become very large

For example, in English:

- `refund` is common
- but `refundability` or `refund-processing` may easily become unknown words

### 2.3 Subword-level: the most common practical compromise

What modern large models most commonly use is:

- subword tokenizer

That is, words are broken into “frequent fragments.”

For example:

- `transformers` -> `transform` + `ers`
- `tokenization` -> `token` + `ization`

The benefits of this approach are:

- the vocabulary does not need to grow without bound
- new words can be composed from existing subwords
- it balances sequence length and OOV issues

This is also why methods like BPE, WordPiece, and SentencePiece are so important.

![Tokenizer Granularity Trade-off Diagram](/img/course/ch07-tokenizer-granularity-tradeoff-map.png)

:::tip Reading Tip
It is best to read this diagram from left to right: character-level is the safest but produces the longest sequences, word-level has intuitive meaning but high OOV risk, and subword-level strikes a balance among vocabulary size, sequence length, and coverage of new words. A tokenizer is not about “what looks neat”; it is about balancing cost and expressiveness.
:::

---

## 3. Let’s Run a Real Tokenizer Example That Shows the Problem Clearly

The code below does not reproduce a full industrial tokenizer,  
but it clearly demonstrates three things:

1. Word-level splitting
2. Subword-level splitting
3. Mapping tokens to IDs, padding, and truncation

```python
import re

vocab = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "refund": 4,
    "policy": 5,
    "reset": 6,
    "password": 7,
    "transform": 8,
    "##er": 9,
    "##s": 10,
    "token": 11,
    "##ization": 12,
    "please": 13,
    "help": 14,
}


def word_tokenize(text):
    return re.findall(r"[A-Za-z]+", text.lower())


def subword_tokenize(word, vocab):
    if word in vocab:
        return [word]

    tokens = []
    start = 0
    while start < len(word):
        matched = None
        for end in range(len(word), start, -1):
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                matched = piece
                tokens.append(piece)
                start = end
                break
        if matched is None:
            return ["[UNK]"]
    return tokens


def encode(text, vocab, max_length=8):
    words = word_tokenize(text)
    tokens = ["[CLS]"]
    for word in words:
        tokens.extend(subword_tokenize(word, vocab))
    tokens.append("[SEP]")

    token_ids = [vocab.get(token, vocab["[UNK]"]) for token in tokens]
    token_ids = token_ids[:max_length]
    attention_mask = [1] * len(token_ids)

    if len(token_ids) < max_length:
        pad_count = max_length - len(token_ids)
        token_ids += [vocab["[PAD]"]] * pad_count
        attention_mask += [0] * pad_count

    return tokens, token_ids, attention_mask


examples = [
    "Please help reset password",
    "Transformers policy",
    "Tokenization refund",
]

for text in examples:
    tokens, token_ids, attention_mask = encode(text, vocab, max_length=10)
    print("-" * 60)
    print("text          :", text)
    print("tokens        :", tokens)
    print("input_ids     :", token_ids)
    print("attention_mask:", attention_mask)
```

### 3.1 Which lines in this code matter most?

Focus on three parts:

1. `word_tokenize`  
   Shows how the raw string is first split into words
2. `subword_tokenize`  
   Shows how a word is greedily split into subwords when it is not in the vocabulary
3. `encode`  
   Shows how special tokens, padding, and truncation are added

### 3.2 Why is `Transformers` split into multiple subwords?

Because the vocabulary does not contain the full word `transformers`,  
but it does contain:

- `transform`
- `##er`
- `##s`

So it can still be represented.

This is the key advantage of a subword tokenizer:

- a new word does not have to be fully present in the vocabulary

### 3.3 What is `attention_mask` for?

Because sentences in a batch usually have different lengths.  
To make them into a single tensor, we pad shorter sentences with `[PAD]`.

But the model should not treat those pad positions as real content,  
so `attention_mask` tells it:

- `1` means a real token
- `0` means padding

![Tokenizer to input_ids and attention_mask Diagram](/img/course/ch07-tokenizer-inputids-mask-length-map.png)

:::tip Reading Tip
When reading this diagram, break the process into four steps: the original text is first split into tokens, then mapped to `input_ids`, shorter sequences are padded with `[PAD]`, and finally `attention_mask` tells the model which positions are real content. Many batch errors and strange results come from not understanding this chain clearly.
:::

---

## 4. Why Does a Tokenizer Directly Affect Cost and Performance?

### 4.1 The more finely a sentence is split, the more tokens it has

More tokens means:

- context is used up more quickly
- inference cost is higher
- API billing is more expensive

So a tokenizer is not just a theoretical issue;  
it also directly affects engineering cost.

### 4.2 Neither a vocabulary that is too small nor too large is good

If the vocabulary is too small:

- many words will be split too finely

If the vocabulary is too large:

- the embedding table becomes larger
- there will be more rare words
- data utilization during training may not actually improve

In practice, tokenizer design is about finding balance among these factors.

### 4.3 Different languages bring different challenges

For example:

- English naturally has spaces, so tokenization is relatively easy
- Chinese has no spaces, so segmentation granularity is more sensitive
- Code also mixes camelCase names, underscores, and symbols

So tokenizers are often adapted to the language characteristics of the training corpus.

---

## 5. Why Do Special Tokens Keep Appearing?

### 5.1 `[CLS]`, `[SEP]`, and `[PAD]` are not just decoration

These special tokens usually serve clear functions:

- `[CLS]`: the starting point for sentence-level representation
- `[SEP]`: separates multiple segments
- `[PAD]`: aligns batch lengths

Different models may use different exact symbols,  
but the idea is very similar.

### 5.2 In chat models, system / user / assistant are also based on the same idea

In the era of chat models, you will see more special markers, such as:

- `<|system|>`
- `<|user|>`
- `<|assistant|>`

They are also fundamentally special tokens that tell the model:

- who is speaking in this part
- how the dialogue structure is separated

So a chat template is actually part of the tokenizer ecosystem as well.

---

## 6. The Easiest Pitfalls to Fall Into

### 6.1 Mistake 1: Thinking the tokenizer is just a preprocessing detail

It is not.  
It directly affects:

- token count
- vocabulary size
- OOV handling
- downstream template format

### 6.2 Mistake 2: Thinking that as long as it can be split, that is enough

What really matters is:

- whether the splitting is stable
- whether it fits the corpus
- whether it balances length and semantic granularity

### 6.3 Mistake 3: Thinking Chinese should always be segmented by “words” for the best result

Not necessarily.  
Many modern models still use:

- character-level
- subword-level
- unified tokenization methods like SentencePiece

The key is still the training objective and data distribution.

---

## Summary

The most important thing in this lesson is not memorizing the names BPE or WordPiece,  
but understanding one main idea:

> **The essence of a tokenizer is to split raw text into discrete units the model can process, while making engineering trade-offs among vocabulary size, unknown-word handling, and sequence length.**

Once this main idea is clear,  
when you later see:

- input ids
- attention mask
- context length
- prompt templates

you will no longer treat them as disconnected concepts.

---

## Exercises

1. Remove `transform` or `##ization` from the vocabulary in the example and see which words degrade into `[UNK]`.
2. Why is a subword tokenizer a compromise between word-level and character-level tokenization?
3. Change `max_length` to a smaller value and observe how truncation affects the output.
4. Think about this: if your corpus contains a lot of code, what problem would a tokenizer design most likely run into first?
