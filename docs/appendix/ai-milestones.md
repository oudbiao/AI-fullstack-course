---
title: "AI Development History: 15 Stages and Key Papers"
sidebar_position: 2
description: "A beginner-friendly 15-stage map of AI development, followed by key papers and algorithms from the perceptron, backpropagation, AlexNet, Transformer, GPT, RAG, Agent, and diffusion models."
keywords: [AI history, AI development stages, key papers, Transformer paper, GPT paper, RAG, Agent, diffusion model paper]
---

# AI Development History: 15 Stages and Key Papers

![AI 15-stage development history map](/img/course/appendix-ai-15-stage-history-map-en.png)

This is an optional background page. If you are new, read only the 15-stage map first, then return to the main course. The long paper tables are for later lookup.

## Quick View: 15 Stages of AI Development

| Stage | Beginner meaning | Course anchor |
|---|---|---|
| 1. The AI question | Can a machine show intelligent behavior? | Course overview |
| 2. Symbolic AI | Humans write rules; machines reason with rules | History background |
| 3. Expert systems | Domain knowledge becomes rule-based software | System design background |
| 4. Probability and statistics | Use uncertainty and evidence instead of only fixed rules | Chapter 4 |
| 5. Classic machine learning | Learn patterns from data and features | Chapter 5 |
| 6. Early neural networks | A model begins to learn simple decision boundaries | Chapters 5-6 |
| 7. Backpropagation | Multi-layer networks become trainable | Chapter 6 |
| 8. Kernel and ensemble era | SVM, trees, forests, and boosting make ML practical | Chapter 5 |
| 9. Deep learning breakthrough | Data + GPUs + deep networks unlock vision and speech | Chapters 6 and 10 |
| 10. Embeddings and sequence models | Text becomes vectors and sequences become learnable | Chapter 11 |
| 11. Transformer and pretraining | Attention makes large-scale language models practical | Chapters 6-7 |
| 12. LLM and alignment | Models become instruction-following assistants | Chapter 7 |
| 13. RAG | Models connect to external knowledge and citations | Chapter 8 |
| 14. Agent and tool use | Models plan, call tools, and leave traces | Chapter 9 |
| 15. Multimodal and AIGC | AI works across text, image, speech, video, and generation | Chapter 12 |

The most important pattern is simple: each stage solves a bottleneck from the previous stage, then creates new engineering problems.

This page is not meant to make you memorize paper names. It is here to help you build a more stable, more vivid sense of history:

- Where did this concept come from?
- In which year did it become a turning point?
- What problem did it actually solve?
- How is it related to the project you are working on?
- Which stage of the course should you go back to in order to really learn it?

You can think of this page as an “AI tech adventure map”: each node is not just an isolated name, but a plot point where the previous generation’s problem was solved and the next generation’s capability was unlocked.

## 1. First, look at the big map: the AI main line is like a relay race

If we compress AI development into a single picture, you can think of it like this:

![AI Main Line Relay Map](/img/course/appendix-ai-main-relay-map-en.png)

The first half can be understood this way: AI started with “how to make judgments,” and gradually moved toward “can machines learn from data on their own?” Classic machine learning stabilized engineering habits such as training, evaluation, and generalization, while deep learning surged again after data, GPUs, and multi-layer networks matured.

What makes this main storyline most interesting is that AI history is not a queue of paper names, but a relay race. The previous runner exposes a problem, and the next one solves it; sometimes the solution is so effective that the whole industry changes direction; sometimes the problem is not fully solved, so the hype cools down and waits for data, compute, and engineering conditions to mature.

## 2. If you want to get interested first, start with these 6 “story-like moments”

Some historical milestones are worth remembering not only because they are important, but because they really feel like turning points in a story:

### When the perceptron first appeared, people briefly thought the “electronic brain” was almost here

After Rosenblatt proposed the perceptron in 1958, the outside world was extremely excited.
The reason was simple:

- It was one of the rare clear models where a machine could learn rules from data
- For people at the time, it already felt like “machines are starting to learn”

So what the early perceptron brought was not just an algorithm,
but the first strong sense that:

> **Maybe learning really can be done by machines.**

### The XOR problem brutally shut down the first wave of neural network hype

Later, Minsky and Papert clearly pointed out:

- A single-layer perceptron cannot even handle XOR, one of the most basic non-linearly separable problems

This was a very heavy blow.
Because it was not a “small issue”; it was a reminder to the whole field that:

- The expressive power of the approach was far weaker than people had imagined

So when people look back, they often treat this moment as:

> **The marker where the first neural network wave moved from frenzy to cooling off.**

### AlexNet was shocking not just because of the score, but because it proved this path could really break through

When AlexNet achieved a clear lead on ImageNet in 2012,
many people began to take deep learning seriously again.

What made it so compelling was not just “another model,” but the sudden realization that:

- big data
- GPUs
- deeper networks
- more stable training tricks

once combined, could really produce a qualitative leap.

So for many people, AlexNet feels more like:

> **The moment when the deep learning era truly fired the starting gun.**

### The title of the Transformer paper itself sounds like a declaration of war

Why is `Attention Is All You Need` so easy to remember?
Because it is not a conservative title. It boldly says:

- The mainstream view used to be that sequence models needed RNNs
- But this paper says attention alone might be enough

Everyone knows what happened next:
this was not just a catchy slogan, but a real rewrite of the main storyline that followed.

So beginners often remember this paper not only because it is technically strong,
but because it has a very classic “era shift” feeling.

### AlphaGo was the first time ordinary people truly felt that “the AI era has arrived”

Many papers are important in academia, but ordinary people may not feel it.
AlphaGo is different.

When it played against Lee Sedol, many people realized very directly:

- machines are not just calculators
- they can really show extremely strong capabilities in complex games

Especially the widely discussed “Move 37” later made many people feel that:

- the way machines make decisions was starting to become less transparent than traditional programs

### The AutoGPT boom was like a collective “Agent excitement experiment”

Once large models started showing that they could:

- plan
- use tools
- loop on their own

many people immediately imagined a very beautiful picture of “fully autonomous Agent.”
But practice later calmed everyone down:

- too much freedom does not mean stability
- multi-step execution easily accumulates errors
- truly deliverable systems still need constraints, logs, evaluation, and guardrails

So the AutoGPT boom is also a very good historical reminder:

> **When AI capabilities first explode, the industry often gets excited first, and only then engineers the system.**

### 2.1 Read these turning points like a comic strip

![AI History Turning Points Comic Strip](/img/course/appendix-ai-history-comic-turning-points-en.png)

This image is best read as a six-panel comic:

- The first panel is “hope”: the perceptron lets people see machines learning rules from data for the first time
- The second panel is “cold water”: the XOR problem reminds everyone that single-layer models have limited expressive power
- The third panel is “revival”: backpropagation makes multi-layer networks trainable again
- The fourth panel is “ignition”: ImageNet, GPUs, and AlexNet push deep learning onto the main stage
- The fifth panel is “changing tracks”: Transformer moves sequence modeling from recurrence to attention
- The sixth panel is “deployment”: RAG, tool use, and Agent connect models to real systems

Looking at history this way feels much easier: not as a pile of years, but as a sequence of “expectation, setback, correction, explosion, and engineering.”

## 3. How to use this page most effectively

The best way to use it is:

1. First learn the main course material
2. When you reach a key concept, come back here and look at its historical position
3. Remember only “what problem it solved” first; do not rush to memorize every paper name

If you can answer these three questions, this page has already helped you:

- Why did this method appear?
- What bottleneck from the previous generation did it replace?
- Where does it sit in this course?

### 3.1 First, one important note: some nodes are not “one paper”

Some items in this page are:

- A very clear landmark paper
  For example, `Attention Is All You Need`

Some are more like:

- An entire field
  For example, Bayes’ theorem, maximum likelihood estimation

And some are more like:

- A method family or a historical turning point
  For example, “the setback of the perceptron era” or “the revival of deep learning”

So the more stable way to understand them is not to force them all into the same shape,
but to first ask:

- Is this node historically very important?
- What problem did it actually solve?
- What main line did it lead to next?

### 3.2 If you do not want to memorize history, first remember three “mood shifts”

For beginners, the easiest thing to remember about AI history is not always the year,
but often the “mood shift” of each generation of researchers:

### First mood shift: maybe machines really can learn after all

When the perceptron first appeared, many people seriously felt for the first time that:

- maybe machines are not just executing rules
- maybe machines can really learn decision boundaries from data

That is why early neural networks could easily generate huge expectations.

### Second mood shift: many things are actually not that easy

XOR limitations, long-dependency problems, training difficulties, and insufficient compute
kept reminding the field again and again that:

- the idea looks beautiful
- but in reality optimization, expressive power, and data conditions are not so easy to keep up with

So AI history is often not a straight upward climb,
but rather:

- excitement first
- setbacks next
- then a new method and moving forward again

### Third mood shift: scale and engineering can really rewrite the main storyline

The truly shocking thing about nodes like AlexNet, Transformer, GPT-3, CLIP, and diffusion models
is often not that “the concept was proposed for the first time,” but that:

- once compute, data, engineering, and training methods finally came together
- something that originally looked like just a research direction
- suddenly began to rewrite the default path of the industry

So when you read this timeline, it is best to see it as:

> **An evolution history of “problems constantly exposed, methods constantly pushing back, and engineering finally making things real.”**

### 3.3 Beginners can read the timeline using just four tags

If this is your first time looking at the page, you do not need to memorize all authors and years at once. A simpler way is to attach four tags to each node:

| Tag | What you should ask | Example |
|---|---|---|
| Old problem | Before it appeared, where were people stuck? | RNN long dependencies and poor parallel training |
| New method | What new solution did it propose? | Transformer replaced the recurrent main line with self-attention |
| New capability | What capability space did it unlock? | Large language models can do few-shot learning and general text generation |
| Project connection | What kind of project will it affect? | RAG, Agent, knowledge-base assistants, AIGC applications |

This way of reading feels more like a technology evolution story, not like memorizing a list of names.

### 3.4 Go one step further: read papers as “old problem -> new method -> new capability”

![AI Paper Problem-Solution-Impact Chain](/img/course/appendix-ai-paper-problem-solution-impact-chain-en.png)

Many paper titles look intimidating, but behind them there is usually a simple chain:

- Old bottleneck: what was hard at the time
- New method: what new architecture, objective, dataset, or training approach the paper proposed
- New capability: what the model could reliably do for the first time
- New project: which course projects and product forms it later influenced

For example, `Attention Is All You Need` can be read like this:

| Old bottleneck | New method | New capability | Projects affected |
|---|---|---|---|
| RNNs were hard to parallelize, and long-distance dependency paths were too long | self-attention, multi-head attention, position encoding | long-text modeling became easier to scale in parallel | LLM, RAG, Agent, multimodal models |

This reading style is very friendly for beginners because it does not require you to derive formulas first. Instead, it helps you understand: which historical problem did this paper actually solve?

### 3.5 From the perspective of course projects, the timeline can be read like this

![AI Timeline Map from the Project Perspective](/img/course/appendix-ai-project-lens-map-en.png)

In other words, this page is not an isolated appendix. It can help you understand why machine learning projects care about evaluation, why deep learning moved toward Transformer, why RAG and Agent are not suddenly invented new words, and why multimodal AI and AIGC naturally follow large models.

---

## 2. Early foundations and probabilistic inference nodes

| Year | Paper / Algorithm | Key author(s) | What it most importantly solved | Where it maps in the course |
|---|---|---|---|---|
| 1763 | Bayes' Theorem (posterior updating idea) | Thomas Bayes (later organized and published by Price) | Established the main probabilistic inference idea of updating judgments when new evidence arrives | 4 Minimum necessary AI math foundations |
| 1948 | A Mathematical Theory of Communication | Claude Shannon | Established the main line of information, entropy, and modern information theory | 4 Minimum necessary AI math foundations |
| 1943 | McCulloch-Pitts Neuron | Warren McCulloch, Walter Pitts | Proposed the earliest abstraction of an artificial neuron and clearly explained that “neurons can perform logical computation” | 6 Deep Learning and Transformer Foundations |
| 1950 | Turing Test | Alan Turing | Turned “whether a machine shows intelligence” from a philosophical question into a testable framework | 9 AI Agent and agent systems / 12 AIGC and multimodal frontiers |
| 1956 | Logic Theorist | Newell, Shaw, Simon | Made it concrete for the first time that “programs can do symbolic reasoning and theorem proving” | AI history overview / Agent background |
| 1922 | Maximum Likelihood Estimation | Ronald Fisher | Systematized the idea of “the parameters that best explain the data” and became a foundational idea for statistical learning and loss functions | 4 Minimum necessary AI math foundations |
| 1977 | EM Algorithm | Dempster, Laird, Rubin | Provided a stable iterative estimation framework for scenarios with latent variables and missing information | 4 Minimum necessary AI math foundations / later probabilistic model background |
| 1990 | MCMC in Bayesian Inference | Gelfand, Smith | Made complex Bayesian inference more feasible in practical computation | Probabilistic inference background / elective extension |
| 2000 | Causality | Judea Pearl | Separated causality from mere correlation and became an important background for modern AI inference and decision-making | Probability and inference extension / Agent / safety |

What makes this line especially attractive is that:

- it is not as flashy as neural networks
- but it quietly determines how we make judgments from incomplete information

So many later model training, evaluation, and decision systems
have always stood on this older probabilistic inference main line.

---

## 3. Key nodes in the three waves of neural networks

![Timeline of Three Neural Network Waves and Two Valleys](/img/course/appendix-neural-network-waves-timeline-en.png)

When reading this figure, think of neural network history as a process of “believing in it again” three times:

- First belief: the perceptron showed that machines can learn boundaries from data
- First valley: XOR exposed the expressive power limit of single-layer models
- Second belief: backpropagation made multi-layer networks trainable
- Second valley: vanishing gradients, limited compute, and limited data made long sequences and deep networks hard to train stably
- Third belief: RBM/DBN, ImageNet, GPUs, ReLU, Dropout, residual connections, and Transformer together pushed deep learning into the mainstream

That is also why the course does not only say “neural networks are powerful,” but repeatedly discusses the training loop, gradients, initialization, regularization, data scale, and engineering diagnostics. What truly changed history was often not one isolated formula, but a whole set of conditions finally coming together.

| Year | Paper / Algorithm | Key author(s) | What it most importantly solved | Where it maps in the course |
|---|---|---|---|---|
| 1958 | Perceptron | Frank Rosenblatt | Proposed one of the earliest trainable linear classifiers and made “machines can learn rules from data” real for the first time | 5 Machine Learning from Basics to Practice: linear models, 6 Deep Learning and Transformer Foundations: basic neurons |
| 1969 | Perceptrons | Marvin Minsky, Seymour Papert | Systematically revealed the limitations of single-layer perceptrons on non-linearly separable problems such as XOR, dealing a blow to the first neural network wave | 6 Deep Learning and Transformer Foundations: perceptron / XOR |
| 1980 | Neocognitron | Kunihiko Fukushima | Anticipated the core ideas of convolution, local receptive fields, and hierarchical feature extraction | 6 Deep Learning and Transformer Foundations: CNN history background |
| 1986 | Backpropagation | Rumelhart, Hinton, Williams | Made multi-layer neural networks trainable effectively and became the foundational key to the second neural network revival | 4 Minimum necessary AI math foundations: chain rule, 6 Deep Learning and Transformer Foundations: forward and backward propagation |
| 1989 | Universal Approximation Theorem | George Cybenko | Theoretically showed that feedforward networks with nonlinearity have very strong function approximation ability | 6 Deep Learning and Transformer Foundations: MLP background |
| 1994 | Learning Long-Term Dependencies is Difficult | Bengio, Simard, Frasconi | Systematically revealed the vanishing gradient problem in long-sequence training and pushed the LSTM route forward | 6 Deep Learning and Transformer Foundations: RNN / LSTM |
| 1997 | LSTM | Hochreiter, Schmidhuber | Used gating mechanisms to alleviate long-term dependency and gradient problems | 6 Deep Learning and Transformer Foundations: LSTM |
| 2006 | A Fast Learning Algorithm for Deep Belief Nets | Hinton, Osindero, Teh | Used RBM / DBN pretraining to make deep networks trainable again, often seen as one of the starting points of the deep learning revival | 6 Deep Learning and Transformer Foundations: historical background |

This part of history feels like a story with clear ups and downs:

- The perceptron gave people the first spark of hope
- The XOR limitation quickly suppressed that hope
- Backpropagation brought multi-layer networks back
- Vanishing gradients reminded everyone that long sequences are not easy
- LSTM and later the engineering of deep learning pushed the whole road back onto the main stage

---

## 4. The classic machine learning main line

This line is not as flashy as large models, but it is very much like the process of turning modeling into reliable engineering. Decision trees made rules learnable, SVM pursued a more stable classification boundary, random forests and boosting combined multiple weak models, and XGBoost made this route a powerful baseline for tabular data projects.

For beginners, the key idea to remember is: what classic machine learning is most worth learning is not the algorithm names, but the whole modeling habit of training, validation, evaluation, tuning, and error analysis.

![Classic Machine Learning Branch Map](/img/course/appendix-classic-ml-branch-map-en.png)

| Year | Paper / Algorithm | Key author(s) | What it most importantly solved | Where it maps in the course |
|---|---|---|---|---|
| 1984 | CART | Breiman, Friedman, Olshen, Stone | Systematically established the framework for decision tree classification and regression | 5 Machine Learning from Basics to Practice: decision trees |
| 1995 | Support-Vector Networks | Cortes, Vapnik | Built a very strong classification route using the maximum-margin idea | Elective / classic ML background |
| 1997 | AdaBoost | Freund, Schapire | Clearly turned “combining weak learners into a strong learner” into a main line | 5 Machine Learning from Basics to Practice: ensemble learning |
| 2001 | Random Forests | Leo Breiman | Used Bagging + random feature selection to make tree models more stable and less prone to overfitting | 5 Machine Learning from Basics to Practice: random forests |
| 2016 | XGBoost: A Scalable Tree Boosting System | Chen, Guestrin | Turned GBDT into an industrial-grade, high-performance implementation and became one of the strong baselines in the tabular-data era | 5 Machine Learning from Basics to Practice: ensemble learning |

---

## 5. The deep learning vision main line

| Year | Paper / Algorithm | Key author(s) | What it most importantly solved | Where it maps in the course |
|---|---|---|---|---|
| 1998 | LeNet-5 | Yann LeCun et al. | Applied convolutional networks to image recognition and established the CNN main line | 6 Deep Learning and Transformer Foundations: CNN |
| 2009 | ImageNet | Deng, Dong, Socher, Li, Li, Fei-Fei | Established the standard large-scale vision dataset and pushed later deep visual models to explode | 10 Computer Vision main-line background |
| 2012 | AlexNet | Krizhevsky, Sutskever, Hinton | Made deep CNNs truly take off on ImageNet and helped trigger the deep learning revival | 6 Deep Learning and Transformer Foundations: classic CNNs, 10 Computer Vision classification |
| 2015 | ResNet | He et al. | Used residual connections to make much deeper networks trainable and solved deep network training difficulties | 6 Deep Learning and Transformer Foundations: classic CNNs, 10 Computer Vision modern architectures |
| 2015 | U-Net | Ronneberger et al. | Made medical image segmentation and pixel-level prediction practical | 10 Computer Vision: segmentation |
| 2015 | Faster R-CNN | Ren et al. | Made object detection move from slow to usable and became a representative of the two-stage detection line | 10 Computer Vision: object detection |
| 2016 | YOLO | Redmon et al. | Turned detection into an end-to-end real-time route | 10 Computer Vision: YOLO series |

---

## 6. The reinforcement learning and game-playing main line

| Year | Paper / Algorithm | Key author(s) | What it most importantly solved | Where it maps in the course |
|---|---|---|---|---|
| 1992 | TD-Gammon | Gerald Tesauro | Used temporal-difference learning to reach human expert-level play in complex games and proved the potential of the reinforcement learning route | RL / Agent historical background |
| 2015 | Human-level control through deep reinforcement learning | Mnih et al. | Achieved human-level performance on Atari games using deep reinforcement learning, and made DQN a key node in the deep RL main line | Agent / RL historical background |
| 2016 | AlphaGo | Silver et al. | Combined deep learning, search, and reinforcement learning to make a historic breakthrough in Go | Agent / planning / frontier history |

This line is especially easy for people outside the field to remember,
because it is often not “the score improved a little in a paper,” but rather:

- games
- competition
- adversarial settings

scenarios that are very intuitive and easy to relate to.

So the historical value of nodes like AlphaGo is not only academic breakthrough,
but also that they first made many ordinary people truly feel:

> **AI has started to enter domains that were once thought to be very hard for machines to touch.**

---

## 7. The modern NLP and large model main line

This line is very much like a story of an evolving language interface: at first we wanted machines to understand words and sentences, then we wanted translation and generation, and later we realized that the same text interface can summarize, write code, reason, call tools, and organize tasks.

![NLP to LLM Lineage Map](/img/course/appendix-nlp-llm-lineage-map-en.png)

When beginners look at this line, they can first grasp two changes: first, words moved from discrete symbols to points in a vector space; second, language models moved from “doing one NLP task” to “accomplishing many tasks through natural language.” This is also the foundation that later made Prompt, RAG, and Agent possible.

| Year | Paper / Algorithm | Key author(s) | What it most importantly solved | Where it maps in the course |
|---|---|---|---|---|
| 1980s~1990s | HMM / statistical sequence modeling route | Rabiner et al. (classic surveys and main-line summaries) | Made sequence tasks such as POS tagging, word segmentation, and speech recognition form a mature statistical main line | 11 NLP sequence labeling background |
| 2006 | CTC (Connectionist Temporal Classification) | Alex Graves et al. | Solved the training problem for sequence tasks such as speech when input and output are not aligned | 11 NLP / speech modeling background |
| 2013 | Word2Vec | Mikolov et al. | Brought word vectors into the mainstream, making words have distributed semantic representations | 11 NLP word embeddings |
| 2013 | AMR (Abstract Meaning Representation) | Banarescu et al. | Brought semantic graph representations into the mainstream and showed that “a sentence is not just a string of words, but can also be a structured semantic graph” | NLP semantic representation extension background |
| 2014 | Sequence to Sequence Learning | Sutskever, Vinyals, Le | Turned the encoder-decoder structure into the translation main line | 11 NLP Seq2Seq |
| 2014 | Deep Speech | Hannun et al. / Baidu | Pushed end-to-end speech recognition into the deep learning era | Speech recognition / CTC background |
| 2014 | GAN | Goodfellow et al. | Proposed generative adversarial learning and opened an important branch of the generative model route | 6 Deep Learning and Transformer Foundations: GAN |
| 2017 | Attention Is All You Need | Vaswani et al. | Replaced the RNN main line with self-attention and solved long-dependency and parallel training bottlenecks | 6 Deep Learning and Transformer Foundations: Transformer |
| 2018 | BERT | Devlin et al. | Turned bidirectional Transformer pretraining + fine-tuning into the NLP understanding main line | 11 NLP BERT, 7 LLM principles, Prompt and fine-tuning |
| 2018 | GPT-1 | Radford et al. | Brought the decoder-only pretraining route to the forefront | 11 NLP GPT |
| 2019 | GPT-2 | Radford et al. | Demonstrated strong generation capabilities with large-scale autoregressive language models | 11 NLP GPT |
| 2020 | GPT-3 | Brown et al. | Used larger-scale models to push few-shot / in-context learning into the mainstream | 11 NLP GPT, 7 LLM principles, Prompt and fine-tuning |

What makes this part of history so exciting is that it feels like a route where capabilities suddenly become more and more general:

- At first it was just about better word representations
- Later it became better sentence generation
- Then suddenly people felt that perhaps the text interface itself was becoming a general-purpose capability base

---

## 8. Alignment, Agent, and systems-oriented main line

If the previous main line solves “can the model generate,” this line cares more about “can the model do things according to human intent, safely, and in a traceable way?” This is also a necessary step when moving from model capability to real products.

![Alignment, Agent, and Systems Main Line Map](/img/course/appendix-agent-system-lineage-map-en.png)

![LLM to Agent Engineering Evolution Timeline](/img/course/appendix-llm-to-agent-evolution-timeline-en.png)

The relationship between this line and projects is very direct: if you are building a knowledge-base assistant, you need to care about citations and no-answer handling; if you are building an Agent, you need to care about tool schemas, call logs, failure recovery, permission boundaries, and stop conditions.

If we tell this line as a story, it actually went through four major shifts in focus:

- Language modeling stage: the model first learns to “continue writing,” with pretraining and next token prediction as the focus
- Instruction-following stage: the model starts to feel more like an assistant, with instruction tuning and RLHF as the focus
- External-world connection stage: the model connects to knowledge bases, databases, and business systems through RAG, function calling, and tool use
- Deliverable system stage: an Agent is no longer just “able to loop autonomously,” but must have planning, state, evaluation, permissions, logs, rollback, and cost control

So when beginners learn Agent, do not only ask whether it can do things on its own. Also ask whether it can be constrained, observed, reviewed, and safely stopped. That is the watershed between a demo and a real system.

| Year | Paper / Algorithm | Key author(s) | What it most importantly solved | Where it maps in the course |
|---|---|---|---|---|
| 2017 | Deep RL from Human Preferences | Christiano et al. | Formally introduced “human preferences” into the reinforcement learning feedback main line | 7 LLM principles, Prompt and fine-tuning: RLHF background |
| 2021 | LoRA | Hu et al. | Turned large model fine-tuning from “change all parameters” into “low-rank incremental adaptation” | 7 LLM principles, Prompt and fine-tuning |
| 2022 | InstructGPT | Ouyang et al. | Pushed RLHF into the mainstream for large language models and solved the problem of “can continue writing, but may not follow instructions well” | 7 LLM principles, Prompt and fine-tuning: RLHF |
| 2022 | Chain-of-Thought Prompting | Wei et al. | Made “think step by step before answering” an explicit prompting strategy | 7 LLM principles, Prompt and fine-tuning / 9 AI Agent and agent systems: reasoning |
| 2022 | ReAct | Yao et al. | Combined reasoning and action alternation and became a key node in the Agent main line | 9 AI Agent and agent systems: reasoning and Agent |
| 2023 | Toolformer | Schick et al. | Taught the model when to call external tools | 9 AI Agent and agent systems: tool use |

---

## 9. Multimodal and AIGC main line

The story of multimodal AI and AIGC can be understood as AI moving from “reading text” to “understanding and generating multiple media types.” CLIP aligns images and text, diffusion models make high-quality generation more stable, Latent Diffusion reduces generation cost, Whisper makes speech recognition more general-purpose, and SAM pushes visual segmentation toward a foundation-model form.

![Multimodal and AIGC Lineage Map](/img/course/appendix-multimodal-aigc-lineage-map-en.png)

Beginners can first remember this: this line is not only about “generating nice pictures,” but about turning text, images, speech, and video into objects that models can understand, generate, and combine. When you later build lecture material generation, creative platforms, digital humans, or multimodal assistants, this line will show up again.

| Year | Paper / Algorithm | Key author(s) | What it most importantly solved | Where it maps in the course |
|---|---|---|---|---|
| 2021 | CLIP | Radford et al. | Aligned images and text into the same semantic space and became one of the key foundations of the multimodal era | 12 AIGC and multimodal foundations |
| 2020 | DDPM | Ho et al. | Turned diffusion models into a practical route and significantly improved generation quality and stability | 12 AIGC and multimodal diffusion models |
| 2022 | Latent Diffusion Models | Rombach et al. | Moved diffusion from pixel space to latent space, greatly reducing cost and becoming the main line behind Stable Diffusion | 12 AIGC and multimodal image generation |
| 2022 | Whisper | Radford et al. | Turned large-scale weakly supervised speech recognition into a generally usable route | 12 AIGC and multimodal speech generation / speech understanding background |
| 2023 | Segment Anything | Kirillov et al. | Pushed large-scale vision foundation models into the segmentation task | 12 AIGC and multimodal / 10 Computer Vision extension |

What makes this line especially attractive to beginners is that:

- it is not just “judging” like traditional classification
- it is entering the world of generating content, editing content, and organizing assets

So many people’s first strong interest in AI
often begins with image generation, speech generation, and multimodal demos.

---

## 10. Review articles, textbooks, and brain-inspired supplementary nodes

The following nodes are not all “a single algorithm paper that directly rewrote the main line,”
but they are very useful for adding background and broadening your worldview, especially when you want to see AI history more completely:

| Year | Node | Author(s) | What it is best for supplementing |
|---|---|---|---|
| 1998 | *Artificial Intelligence: A New Synthesis* | Nils J. Nilsson | Good for supplementing the big picture from classic AI to modern AI |
| 2004 | Hierarchical Temporal Memory (HTM) | Jeff Hawkins, Dileep George et al. | Good for supplementing the brain-inspired route and hierarchical temporal memory ideas |
| 2015 | *Deep Learning* (Nature review) | LeCun, Bengio, Hinton | Good for supplementing the overview of why deep learning became mainstream again |

If you are learning the main course, these three are better suited as:

- extended reading
- background supplementation
- worldview calibration

rather than nodes you must memorize on the first pass.

---

## 11. The 17 nodes most worth remembering first in this course

If you do not want to memorize too many things at once, you can start with just these 17. A lighter way to remember them is not as a paper list, but as 17 plot twists.

| Node | One-sentence beginner understanding | What it helps with in projects |
|---|---|---|
| 1763 Bayes | When new evidence arrives, judgments should be updated | Understanding probabilistic inference, recommendation, diagnosis, and uncertainty |
| 1922 Fisher / MLE | Find the parameters that best explain the data | Understanding loss functions and model training |
| 1948 Shannon | Information and entropy can be computed | Understanding cross-entropy, information theory, and model evaluation |
| 1958 Perceptron | Machines first seemed to learn rules from data | Understanding linear classification and the neuron prototype |
| 1969 Minsky & Papert | Single-layer models have limited capability | Understanding why multi-layer networks and nonlinearity are needed |
| 1986 Backpropagation | Multi-layer networks finally became trainable effectively | Understanding the deep learning training loop |
| 1989 Cybenko | Neural networks have strong function approximation ability | Understanding why MLPs have expressive power |
| 1994 Gradient vanishing problem | Long sequences are very easy to learn poorly | Understanding the motivation for RNN, LSTM, and Transformer |
| 1997 LSTM | Gated memory alleviates long dependencies | Understanding early sequence modeling routes |
| 2006 RBM / DBN | Deep networks became trainable again | Understanding the background of the deep learning revival |
| 2009 ImageNet | Large datasets changed vision research | Understanding how data scale affects model capability |
| 2012 AlexNet | Deep learning broke through mainstream vision tasks | Understanding the combined power of CNNs, GPUs, and big data |
| 2013 Word2Vec | Words gained a computable semantic space | Understanding embeddings and vector retrieval |
| 2017 Transformer | Attention became the main line of sequence modeling | Understanding the base of large models, RAG, and Agent |
| 2018 BERT / GPT-1 | The understanding and generation main lines became clearly separated | Understanding the source of pretraining, fine-tuning, and Prompt |
| 2022 InstructGPT / RLHF | The model moved from “can continue writing” to “follows instructions better” | Understanding alignment and controllable output |
| CLIP / DDPM / Latent Diffusion | Multimodal and generative content entered the main line | Understanding AIGC, multimodal assistants, and creative workflows |

---

## 12. How course stages map to these historical nodes

| Learning station | Historical nodes most worth remembering |
|---|---|
| 4 Minimum necessary AI math foundations | Bayes, Shannon, MLE, EM |
| 5 Machine Learning from Basics to Practice | CART, SVM, Random Forest, XGBoost |
| 6 Deep Learning and Transformer Foundations | Perceptron, XOR limitations, Backprop, Cybenko, LSTM, RBM, AlexNet, ResNet, Transformer |
| 7 LLM principles, Prompt, and fine-tuning | Transformer, BERT, GPT-3, LoRA, InstructGPT |
| 9 AI Agent and agent systems | DQN, AlphaGo, Chain-of-Thought, ReAct, Toolformer |
| 10 Computer Vision (elective track) | AlexNet, ResNet, U-Net, Faster R-CNN, YOLO |
| 11 Natural Language Processing (elective track) | HMM, Word2Vec, AMR, Seq2Seq, BERT, GPT, CTC / Deep Speech |
| 12 AIGC and multimodal | CLIP, DDPM, Latent Diffusion |

## 12.1 Find by chapter: where should you learn these nodes?

If you are looking up a course chapter from a paper or algorithm name, you can use this table to locate it first:

| Node you want to understand | Which chapter to study | Why it belongs there |
|---|---|---|
| Bayes’ theorem | 4.2 Probability and statistics in practice | It is the probability foundation of “updating judgments when new evidence arrives” |
| Maximum likelihood estimation, MLE | 4.2 Statistical inference, 5.2 Supervised learning | It explains “inferring parameters from data” and where many losses come from |
| EM algorithm | 4.2 Probabilistic statistics main line, 5.3 Unsupervised learning | It is good for explaining hidden variables and iterative estimation behind clustering |
| Shannon information theory | 4.2 Information theory basics, 6.2 PyTorch loss | Entropy, cross-entropy, and KL divergence directly enter model training |
| Perceptron, XOR, backpropagation | 6.1 Neural network basics | They explain why neural networks need multiple layers, activations, and backprop |
| Neocognitron, LeNet, AlexNet, ResNet | 6.3 CNN, 10 Computer Vision | They are the main line of visual deep learning from local features to deep networks |
| Vanishing gradients, LSTM, GRU | 6.4 RNNs and sequence models | They explain why long sequences are hard and why gated memory appeared |
| RBM / DBN | 6.1 Deep learning history main line, generative models elective | They are better understood as the background to the deep learning revival |
| SVM | 5.2 Supervised learning, classic ML elective | It complements the classic route of maximum margin, support vectors, and kernels |
| Random Forest, Boosting, XGBoost | 5.2 Ensemble learning | They are strong baselines for tabular data and the main line of engineering modeling |
| HMM, CRF, BiLSTM-CRF | 11.4 Sequence labeling | They explain POS tagging, NER, and token-level label constraints |
| Word2Vec | 11.2 Word embeddings and language models | It turns “similar meaning” into a computable relation in vector space |
| AMR, semantic graphs | 11.7 NLP integrated projects | They connect information extraction, knowledge graphs, RAG, and lecture material generation |
| CTC, Deep Speech | 11.5 Seq2Seq extensions, 12 multimodal speech background | They explain alignment problems in speech recognition |
| Transformer, BERT, GPT | 6.5 Transformer, 7 LLM principles, 11.6 Pretrained models | They are the shared foundation of modern NLP and large models |
| TD-Gammon, DQN, AlphaGo | 9.1 Historical background for Agent basics | They explain how action, feedback, search, and planning connect to Agent |
| RLHF, ReAct, Toolformer | 7 Alignment, 9 reasoning and tool use | They explain how large models move from generating to acting toward goals |
| CLIP, diffusion models, Whisper, SAM | 12 AIGC and multimodal | They explain how text, images, speech, and video enter unified AI systems |

## 13. The most practical way to memorize

If you are worried that there are too many names, you can remember them using just this template:

- `Which year`
- `Which paper / which algorithm`
- `What old problem it solved`
- `What new capability it opened up`
- `What kind of project it will affect`
- `Which stage of the course it will appear in again`

For example:

- `2017, Attention Is All You Need, solved RNN parallelization and long-dependency path problems, opened the Transformer and large-model main line, affects Prompt, RAG, Agent, and multimodal applications, corresponds to Transformer in 6 Deep Learning and Transformer Foundations`

Remembering it this way is far more useful than simply memorizing paper names.

## 14. A small exercise: rewrite paper nodes in project language

After reading this page, choose any 3 nodes and rewrite them using the following format:

```text
Node I chose: Attention Is All You Need
The problem before it: RNNs were not ideal for long sequences or parallel training
The change it brought: self-attention became the main line of sequence modeling
Projects it affects: large models, RAG, Agent, multimodal models
What course material I should revisit: 6 Deep Learning and Transformer Foundations, 7 LLM principles, 8 RAG, 9 Agent
```

The purpose of this exercise is not to write a paper survey, but to train yourself to connect historical nodes with your own learning path and project path. The truly useful history of AI is not “knowing what happened,” but knowing: why it appeared, what problem it solved, and which systems you are building today it has influenced.
