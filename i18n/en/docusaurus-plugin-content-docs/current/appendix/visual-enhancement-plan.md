---
title: "Course Visual Enhancement Plan"
description: "Organize, by presentation chapter, which pages are suitable for adding more images, diagrams, flowcharts, and code-based visualizations to help newcomers build intuition faster."
keywords: [course illustrations, instructional design, visual learning, AI course images, Mermaid, image generation]
---

# Course Visual Enhancement Plan

![Course image asset planning board](/img/course/appendix-visual-enhancement-kanban.png)

![Flowchart from image gap detection to generation and release](/img/course/appendix-image-production-pipeline-map.png)

:::tip Reading guide
More images are not always better. The real goal is to reduce understanding cost. When reviewing images, follow the closed loop of "find the gap -> define the image intent -> generate the image -> insert it into the page -> record manifest/progress -> verify the build" to avoid losing control of image assets.
:::

The course now already has stage homepage hero visuals, as well as many Mermaid flowcharts. The next step is not to put images on every page, but to place them where they most effectively reduce understanding cost.

:::info Chapter numbering convention
The course source directory is already aligned with the web chapter numbers: `ch01-tools` corresponds to Chapter 1, and `ch05-machine-learning` corresponds to Chapter 5. The main tracks 1–4 in the sidebar are learning groups, not directory hierarchy levels.
:::

A simple principle is:

| Content type | Better visual format | Notes |
|---|---|---|
| Abstract concepts | Analogical illustrations, structural diagrams | Help newcomers first build a mental picture |
| Multi-step processes | Mermaid or flowcharts | Help newcomers understand the sequence |
| Math and data | Code-generated charts | More accurate than AI-generated images |
| Model architecture | Module structure diagrams | Help newcomers understand inputs, outputs, and intermediate layers |
| Project practice | Architecture diagrams, UI sketches, result example images | Help newcomers understand the final target |
| Historical context | Timeline charts, person-and-paper cards | Help newcomers place algorithms back into their era |

## Priority rules

| Priority | Images to make first | Reason |
|---|---|---|
| P0 | Hero visual for each chapter homepage | Already done; sets the chapter tone |
| P1 | Concept diagram for the first main lesson in each chapter | Most affects a newcomer’s first impression |
| P1 | System architecture diagrams and result examples for project pages | Most helpful for getting started |
| P2 | Code visualizations for math, training, and evaluation | Turn abstract processes into observable results |
| P2 | Historical timelines and paper story diagrams | Improve interest and memorability |
| P3 | Decorative illustrations | Add only when a page feels too plain or tiring to read |

## Chapter 1 (Directory ch01-tools): Developer Tools Fundamentals

| Section | Recommended image | Priority |
|---|---|---|
| Terminal command line | Diagram showing the relationship between the terminal, directory tree, and command execution results | P1 |
| Git basics | Four-panel diagram of working directory, staging area, local repository, and remote repository | P1 |
| Development environment | Diagram of the relationship between the Python environment, VS Code, Jupyter, and dependency files | P2 |

Images suitable for generation: developer workspace, Git archive system, environment isolation lab.

More suitable for code or Mermaid: Git state flow, branch merge process, virtual environment path relationships.

## Chapter 2 (Directory ch02-python): Python Programming Basics

| Section | Recommended image | Priority |
|---|---|---|
| Python basic syntax | Diagram showing how variables, branches, loops, and functions combine into program building blocks | P1 |
| Advanced Python | Runtime relationship diagram for objects, exceptions, files, and generators | P2 |
| Project practice | Input/output UI sketches for CLI, Web API, and AI API | P1 |

Images suitable for generation: Python mini-tool workshop, API request and response workstation.

More suitable for code-generated visuals: function call stack, before-and-after comparison of JSON file reading and writing.

## Chapter 3 (Directory ch03-data-analysis): Data Analysis and Visualization

| Section | Recommended image | Priority |
|---|---|---|
| Pure Python data warm-up | Diagram showing how raw lists/dictionaries become tables | P2 |
| NumPy | Diagrams for array shape, slicing, broadcasting, and matrix multiplication | P1 |
| Pandas | Diagram with a table-animation feel for DataFrame, Index, column selection, groupby, and merge | P1 |
| Data visualization | Decision tree for choosing charts and comparison of "the same data with different charts" | P1 |
| Database elective | Table, primary key, foreign key, and SQL query path diagram | P2 |
| Project practice | Sample EDA report images and multi-data-source merge flowchart | P1 |

Images suitable for generation: data detective workstation, analysis report cover, database archive system.

More suitable for code-generated visuals: histogram, box plot, scatter plot, line chart, heatmap.

## Chapter 4 (Directory ch04-ai-math): AI Math Fundamentals

| Section | Recommended image | Priority |
|---|---|---|
| Linear algebra | Vector arrows, matrix transformations, eigenvector directions, SVD decomposition diagram | P1 |
| Probability and statistics | Probability trees, distribution curves, sampling error, and MLE/EM detective-style analogies | P1 |
| Calculus and optimization | Function curves, tangent lines, gradient arrows, downhill paths, and chain rule propagation diagram | P1 |

Images suitable for generation: math goggles, gradient descent map, probability detective.

More suitable for code-generated visuals: 2D vectors, normal distribution, cross-entropy curves, gradient descent trajectory.

## Chapter 5 (Directory ch05-machine-learning): Machine Learning

| Section | Recommended image | Priority |
|---|---|---|
| Machine learning basics | Closed loop diagram of task definition, data splitting, baseline, evaluation, and review | P1 |
| Supervised learning | Linear regression fit line, logistic regression decision boundary, tree model split diagram | P1 |
| Unsupervised learning | Clustering results, PCA projection, anomaly detection diagram | P1 |
| Model evaluation | Confusion matrix, bias-variance curves, cross-validation fold diagram | P1 |
| Feature engineering | Processing pipeline diagram from raw fields to feature tables | P2 |
| Project practice | Report layouts and error analysis boards for house price, churn, and segmentation projects | P1 |

Images suitable for generation: modeling detective report, model training pipeline.

More suitable for code-generated visuals: decision boundary, residual plot, ROC/PR curves, PCA visualization.

## Chapter 6 (Directory ch06-deep-learning): Deep Learning and Transformer

| Section | Recommended image | Priority |
|---|---|---|
| Neural network basics | Neuron, MLP, activation function, and backpropagation responsibility flow diagram | P1 |
| PyTorch | Relationship diagram of Tensor, Dataset, DataLoader, Module, and Training Loop | P1 |
| CNN | Convolution kernel sliding, feature maps, pooling, and evolution of classic architectures | P1 |
| RNN | Time unrolling, hidden-state passing, and LSTM gating diagram | P1 |
| Transformer | QKV, Self-Attention, and Encoder/Decoder block structure diagram | P1 |
| Generative models | GAN adversarial relationship, VAE latent space, and generated sample evolution | P2 |
| Training tips | Training curve diagnosis chart, compression decision tree, tuning log board | P2 |
| Project practice | Result presentation templates for image classification, sentiment analysis, and generation projects | P1 |

Images suitable for generation: model engine room, neural network training lab.

More suitable for code-generated visuals: loss curves, activation function curves, attention heatmaps, latent space scatter plots.

## Chapter 10 (Directory ch10-computer-vision): Computer Vision

| Section | Recommended image | Priority |
|---|---|---|
| CV basics | Comparison of pixel grids, RGB channels, filtering, and edge detection before and after | P1 |
| Image classification | Data augmentation before/after comparison, architecture evolution, error sample wall | P1 |
| Object detection | bounding box, IoU, NMS, and YOLO grid diagram | P1 |
| Image segmentation | Semantic segmentation mask, instance segmentation mask, and boundary error diagram | P1 |
| Advanced vision | Face recognition pipeline, OCR layout parsing, video frame analysis, 3D point cloud | P1 |
| Vision projects | Security alert closed loop and medical imaging review dashboard | P1 |

Images suitable for generation: vision task map, image understanding workstation.

More suitable for real/code visuals: before-and-after preprocessing images, detection boxes, segmentation masks, OCR bounding results.

## Chapter 11 (Directory ch11-nlp): Natural Language Processing

| Section | Recommended image | Priority |
|---|---|---|
| Text basics | Comparison of text cleaning, tokenization, and BoW/TF-IDF representations | P1 |
| Representation learning | Evolution diagram from one-hot to Word2Vec to contextual representations | P1 |
| Text classification | Comparison of input paths for traditional classification and deep classification | P2 |
| Sequence labeling | Color-coded BIO tag to entity reconstruction diagram | P1 |
| Seq2Seq | Encoder-Decoder, Attention alignment matrix, and translation flowchart | P1 |
| Pretrained models | Diagram showing the task-organization differences among BERT/GPT/T5 | P1 |
| NLP projects | Sample input/output images for question answering, summarization, and information extraction | P1 |

Images suitable for generation: text understanding assistant, language model evolution path.

More suitable for code-generated visuals: TF-IDF table, word vector neighbors, attention alignment heatmap.

## Chapter 7 (Directory ch07-llm-principles): LLM Principles, Prompt, and Fine-Tuning

| Section | Recommended image | Priority |
|---|---|---|
| NLP crash course | Runtime diagram for Tokenizer, Embedding, and HuggingFace pipeline | P2 |
| LLM overview | Timeline of GPT/BERT/Transformer development | P1 |
| Transformer deep dive | Attention complexity, model variants, and scaled computation structure | P1 |
| Pretraining | Data cleaning, training objectives, and training cluster pipeline | P1 |
| Prompt | Comparison cards from bad Prompt to good Prompt | P1 |
| Fine-tuning | Comparison of parameter updates for Full fine-tune, LoRA, and QLoRA | P1 |
| Alignment | Flowchart of SFT, Reward Model, and RLHF | P1 |
| Project | Domain fine-tuning plan document and evaluation dashboard | P2 |

Images suitable for generation: large model capability factory, Prompt lab.

More suitable for Mermaid/structural diagrams: structured output schema, LoRA plugin structure, RLHF workflow.

## Chapter 8 (Directory ch08-rag): LLM Application Development and RAG

| Section | Recommended image | Priority |
|---|---|---|
| RAG | Closed loop diagram for document parsing, chunking, vector database, retrieval, reranking, and citation | P1 |
| Deployment | System topology diagram for local models, inference services, and unified API | P1 |
| Application development | Workflow diagram for Function Calling, conversation state, document parsing, and template export | P1 |
| Engineering | Async concurrency, API, log monitoring, and Docker deployment architecture diagram | P2 |
| Comprehensive project | Product UI sketches for enterprise knowledge bases, intelligent assistants, and courseware generation assistants | P1 |

Images suitable for generation: knowledge base engine room, courseware generation assistant workstation.

More suitable for Mermaid/structural diagrams: RAG trace, tool call chain, API request/response, log field flow.

## Chapter 9 (Directory ch09-agent): AI Agent

| Section | Recommended image | Priority |
|---|---|---|
| Agent basics | Boundary comparison diagram of Agent, chatbot, workflow, and function calling | P1 |
| Reasoning and planning | ReAct, Plan-and-Execute, and reflection loop diagrams | P1 |
| Tools | Tool schema, tool selection, and tool safety boundary diagram | P1 |
| Memory | Layered diagram of short-term memory, long-term memory, episodic memory, and procedural memory | P1 |
| MCP | Protocol structure diagram for Host, Client, Server, Resource, and Tool | P1 |
| Frameworks | Selection matrix for LangGraph, LlamaIndex, AutoGen, and similar frameworks | P2 |
| Multi-Agent | Role collaboration, message passing, and task assignment diagram | P1 |
| Evaluation and safety | Trace replay, guardrail layers, and permission confirmation diagram | P1 |
| Deployment and operations | Agent runtime, queue, logs, recovery, and cost monitoring architecture diagram | P1 |
| Project | Execution trace diagrams for research assistants, data analysis Agents, and multi-Agent teams | P1 |

Images suitable for generation: Agent command center, multi-tool collaboration control room.

More suitable for Mermaid/structural diagrams: execution trace, state machine, tool allowlist, human confirmation process.

## Chapter 12 (Directory ch12-multimodal): AIGC and Multimodal

| Section | Recommended image | Priority |
|---|---|---|
| Multimodal basics | Alignment and fusion diagram for text, images, speech, and video | P1 |
| Image generation | Diffusion model noise-addition/noise-removal diagram, Stable Diffusion component diagram, before-and-after LoRA fine-tuning comparison | P1 |
| Video and speech | Copywriting, storyboarding, TTS, video generation, and digital human pipeline diagram | P1 |
| Frontiers and compliance | Risk classification, copyright review, and human review workflow diagram | P2 |
| Comprehensive project | UI sketches for an AI creative platform, asset version flow, and export package structure | P1 |

Images suitable for generation: AI creative workstation, multimodal content factory.

More suitable for real/process diagrams: generation workflow, asset version tree, review checklist, output bundle.

## Appendix (Directory appendix): Reference, Troubleshooting, and Learning Support

The appendix is not a place for too many decorative images. It is better suited to reference-style images that help you "locate problems quickly." When newcomers open the appendix, they usually are not there to study a whole chapter systematically, but to quickly figure out: where am I stuck, which page should I check, what resources should I add, whether I should buy hardware, and how to prepare for projects and job hunting.

| Page | Recommended image | Priority |
|---|---|---|
| Recommended learning resources | Resource selection funnel: main course, targeted supplements, project validation, and review/refinement | P2 |
| Course numbering convention | Map showing the correspondence between chapter numbers and source directories | P3 |
| AI important papers and algorithm timeline | Long AI history timeline poster or "paper milestone card wall" | P1 |
| Course visual enhancement plan | Course image asset planning board: how P0/P1/P2/P3 are rolled out in batches | P3 |
| Hardware and cloud resource guide | Decision tree for hardware purchase and cloud GPU selection | P1 |
| Common questions | Newcomer learning decision tree: how to route questions about math, GPU, projects, papers, jobs, and more | P2 |
| Continuous learning methodology | Three-layer flywheel diagram for fundamentals, projects, and frontier tracking | P2 |
| Learning bottleneck rescue | Troubleshooting flowchart for environment, code, training, VRAM, projects, and anxiety | P1 |
| Learning resource quick reference | AI project quick-reference overview: environment, baseline, evaluation, RAG, Agent, Prompt | P2 |
| Job preparation checklist | Job-hunting funnel diagram: role targeting, project polishing, resume wording, and interview review | P1 |

Images suitable for generation: troubleshooting rescue map, hardware decision tree, learning resource navigation board, job-prep battle board, AI history timeline poster.

More suitable for Mermaid/structural diagrams: course-number correspondence, FAQ routing tree, continuous-learning flywheel, batch planning for image asset generation.

## The 20 most valuable images to generate first

If we only make one batch first, I recommend generating or producing them in the order below:

| Order | Image | Suggested insertion location | Type |
|---|---|---|---|
| 1 | Four-panel diagram of Git working directory, staging area, local repository, and remote repository | `ch01-tools/ch02-git/01-git-basics.md` | Flow diagram |
| 2 | Pandas DataFrame structure diagram | `ch03-data-analysis/ch03-pandas/01-core-structures.md` | Concept diagram |
| 3 | Chart selection decision tree | `ch03-data-analysis/ch04-visualization/04-best-practices.md` | Decision diagram |
| 4 | Gradient descent downhill path diagram | `ch04-ai-math/ch03-calculus/03-gradient-descent.md` | Code visualization |
| 5 | Machine learning modeling closed loop diagram | `ch05-machine-learning/ch01-ml-basics/01-what-is-ml.md` | Flowchart |
| 6 | Confusion matrix and error cost diagram | `ch05-machine-learning/ch04-evaluation/01-metrics.md` | Teaching diagram |
| 7 | PyTorch training loop diagram | `ch06-deep-learning/ch02-pytorch/05-training-loop.md` | Flowchart |
| 8 | CNN convolution kernel sliding diagram | `ch06-deep-learning/ch03-cnn/01-convolution-basics.md` | Concept diagram |
| 9 | Self-Attention QKV diagram | `ch06-deep-learning/ch05-transformer/01-attention-mechanism.md` | Structural diagram |
| 10 | Object detection output breakdown diagram | `ch10-computer-vision/ch03-detection/01-detection-overview.md` | Task diagram |
| 11 | Semantic segmentation mask comparison diagram | `ch10-computer-vision/ch04-segmentation/01-semantic-segmentation.md` | Result diagram |
| 12 | BIO tag to entity reconstruction diagram | `ch11-nlp/ch04-sequence-labeling/01-ner-overview.md` | Annotation diagram |
| 13 | BERT/GPT/T5 comparison diagram | `ch11-nlp/ch06-pretrained/00-roadmap.md` | Comparison diagram |
| 14 | Before-and-after Prompt rewrite comparison card | `ch07-llm-principles/ch05-prompt/01-prompt-basics.md` | Comparison diagram |
| 15 | LoRA parameter update comparison diagram | `ch07-llm-principles/ch06-finetuning/02-lora-qlora.md` | Structural diagram |
| 16 | RAG document-to-answer closed loop diagram | `ch08-rag/ch01-rag/01-rag-basics.md` | System diagram |
| 17 | Courseware generation assistant workflow diagram | `ch08-rag/ch05-projects/04-courseware-assistant.md` | Project architecture diagram |
| 18 | Agent tool-calling trace diagram | `ch09-agent/ch03-tools/08-multi-tool-practice.md` | Execution trace diagram |
| 19 | Agent guardrail layering diagram | `ch09-agent/ch08-eval-safety/04-guardrails.md` | Safety diagram |
| 20 | Diffusion model noise-addition/noise-removal diagram | `ch12-multimodal/ch02-image-gen/01-diffusion-models.md` | Model process diagram |

## Second batch: 25 recommended images

The first batch covers the entry points and the most core concepts. The second batch continues to fill in pages where newcomers often get stuck, but which also form the foundation of later chapters. Prioritize conceptual structure diagrams, algorithm intuition diagrams, and project workflow diagrams.

| Order | Image | Suggested insertion location | Type |
|---|---|---|---|
| 1 | Matplotlib Figure and Axes structure diagram | `ch03-data-analysis/ch04-visualization/01-matplotlib.md` | Object model diagram |
| 2 | Seaborn statistical chart selection diagram | `ch03-data-analysis/ch04-visualization/02-seaborn.md` | Chart map |
| 3 | SQL table join relationship diagram | `ch03-data-analysis/ch05-database/02-sql-basics.md` | Data relationship diagram |
| 4 | EDA exploratory data analysis flowchart | `ch03-data-analysis/ch06-projects/01-eda-project.md` | Project flowchart |
| 5 | Vector dot product and cosine similarity geometric diagram | `ch04-ai-math/ch01-linear-algebra/01-vectors.md` | Geometric intuition diagram |
| 6 | Matrix linear transformation grid diagram | `ch04-ai-math/ch01-linear-algebra/02-matrices.md` | Math visualization |
| 7 | Probability distribution and Bayesian update diagram | `ch04-ai-math/ch02-probability/01-probability-basics.md` | Concept diagram |
| 8 | Information entropy and uncertainty diagram | `ch04-ai-math/ch02-probability/04-information-theory.md` | Concept diagram |
| 9 | Scikit-learn Estimator and Pipeline diagram | `ch05-machine-learning/ch01-ml-basics/02-sklearn-intro.md` | Engineering flow diagram |
| 10 | Linear regression fit and loss surface diagram | `ch05-machine-learning/ch02-supervised/01-linear-regression.md` | Algorithm intuition diagram |
| 11 | Logistic regression decision boundary diagram | `ch05-machine-learning/ch02-supervised/02-logistic-regression.md` | Classification boundary diagram |
| 12 | Decision tree split path diagram | `ch05-machine-learning/ch02-supervised/03-decision-trees.md` | Algorithm structure diagram |
| 13 | Ensemble learning voting and forest diagram | `ch05-machine-learning/ch02-supervised/04-ensemble-learning.md` | Model ensemble diagram |
| 14 | K-Means clustering center iteration diagram | `ch05-machine-learning/ch03-unsupervised/01-clustering.md` | Algorithm process diagram |
| 15 | PCA dimensionality reduction projection diagram | `ch05-machine-learning/ch03-unsupervised/02-dimensionality-reduction.md` | Spatial projection diagram |
| 16 | Neural network forward and backward propagation diagram | `ch06-deep-learning/ch01-nn-basics/02-forward-backward.md` | Training mechanism diagram |
| 17 | Classic CNN architecture evolution diagram | `ch06-deep-learning/ch03-cnn/03-classic-architectures.md` | Historical evolution diagram |
| 18 | LSTM gating memory flow diagram | `ch06-deep-learning/ch04-rnn/02-lstm-gru.md` | Structural mechanism diagram |
| 19 | Transformer Block architecture diagram | `ch06-deep-learning/ch05-transformer/02-transformer-architecture.md` | Module structure diagram |
| 20 | Word vector semantic neighborhood diagram | `ch11-nlp/ch02-embeddings/01-word-embedding.md` | Semantic space diagram |
| 21 | BERT Masked Language Model diagram | `ch11-nlp/ch06-pretrained/02-bert.md` | Pretraining objective diagram |
| 22 | GPT autoregressive generation diagram | `ch11-nlp/ch06-pretrained/03-gpt-series.md` | Generation mechanism diagram |
| 23 | RLHF three-stage flowchart | `ch07-llm-principles/ch07-alignment/02-rlhf.md` | Alignment flowchart |
| 24 | RAG evaluation triangle diagram | `ch08-rag/ch01-rag/07-rag-evaluation.md` | Evaluation framework diagram |
| 25 | Agent memory system layering diagram | `ch09-agent/ch04-memory/01-memory-overview.md` | System structure diagram |

## Ten recommended images for the appendix

For the appendix, these images should be primarily reference-style and decision-style, helping learners quickly return to the correct page when they get stuck.

| Order | Image | Suggested insertion location | Type |
|---|---|---|---|
| 1 | AI historical relay timeline diagram | `appendix/ai-milestones.md` | Main historical timeline |
| 2 | Learning bottleneck troubleshooting map | `appendix/troubleshooting.md` | Troubleshooting flowchart |
| 3 | Hardware and cloud resource decision tree | `appendix/hardware.md` | Decision diagram |
| 4 | Job preparation funnel diagram | `appendix/job-prep.md` | Planning diagram |
| 5 | Three-layer continuous learning flywheel diagram | `appendix/continuous-learning.md` | Methodology diagram |
| 6 | Resource selection funnel diagram | `appendix/resources.md` | Learning resource navigation diagram |
| 7 | FAQ newcomer question routing tree | `appendix/faq.md` | Question routing diagram |
| 8 | AI project quick-reference overview | `appendix/resource-quick-ref.md` | Quick-reference map |
| 9 | Map showing the correspondence between chapter numbers and source directories | `appendix/course-numbering.md` | Maintenance guide diagram |
| 10 | Course image asset planning board | `appendix/visual-enhancement-plan.md` | Asset planning diagram |

## Generation strategy

For the first batch, make the P1 teaching diagrams first and do not make decorative images. Course visuals should serve understanding, not just make the page look lively.

For the second batch, add screenshots and UI sketches for project pages. Project pages most need to show newcomers what the final work should look like, especially for RAG, Agent, courseware generation assistants, and multimodal creative platforms.

For the third batch, add story images about historical figures, papers, and algorithms. These are best placed on history pages and chapter openings to help learners remember why a technique emerged.

For the appendix batch, I recommend first making the four P1 images for `ai-milestones`, `troubleshooting`, `hardware`, and `job-prep`, then adding resource, FAQ, continuous-learning, and quick-reference diagrams. This improves reference efficiency without turning the appendix into a pile of images.

## Suggestions for using the current image script

The current script `scripts/generate_course_images.py` already manages course image assets. Going forward, you can continue appending the high-priority images above to `IMAGE_JOBS` and generate them with the following commands:

```bash
pip install -r requirements-course-ai.txt
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_BASE_URL="https://cliproxy.airoads.org/v1"
export OPENAI_IMAGE_MODEL="gpt-image-2"
npm run images:dry-run
npm run images:generate
```

For math curves, training curves, detection boxes, and segmentation masks, prefer code generation because accuracy matters. For stage hero visuals, project workstations, and historical story diagrams, image generation models are a better fit.
