/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  courseSidebar: [
    "index",
    {
      type: "category",
      label: "🚀 开始之前",
      collapsed: false,
      items: [
        "intro/quick-experience",
        "intro/milestones",
        "intro/career-guide",
        "intro/learning-strategy",
        "intro/learning-path",
        "intro/environment-setup",
      ],
    },
    {
      type: "category",
      label: "第零阶段：开发者工具基础",
      collapsed: true,
      link: { type: "doc", id: "stage0/index" },
      items: [
        {
          type: "category",
          label: "第1章 终端与命令行",
          items: [
            "stage0/ch01-terminal/why-cli",
            "stage0/ch01-terminal/basic-operations",
            "stage0/ch01-terminal/package-managers",
          ],
        },
        {
          type: "category",
          label: "第2章 Git 与版本管理",
          items: [
            "stage0/ch02-git/git-basics",
            "stage0/ch02-git/core-operations",
            "stage0/ch02-git/remote-repos",
            "stage0/ch02-git/branches",
          ],
        },
        {
          type: "category",
          label: "第3章 开发环境配置",
          items: [
            "stage0/ch03-devenv/python-env",
            "stage0/ch03-devenv/vscode",
            "stage0/ch03-devenv/jupyter",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "第一阶段：Python 编程基础",
      collapsed: true,
      link: { type: "doc", id: "stage1/index" },
      items: [
        {
          type: "category",
          label: "第1章 Python 语言入门",
          items: [
            "stage1/ch01-basics/intro",
            "stage1/ch01-basics/data-types",
            "stage1/ch01-basics/operators",
            "stage1/ch01-basics/io",
            "stage1/ch01-basics/control-flow",
            "stage1/ch01-basics/data-structures",
            "stage1/ch01-basics/functions",
            "stage1/ch01-basics/modules",
          ],
        },
        {
          type: "category",
          label: "第2章 Python 进阶",
          items: [
            "stage1/ch02-advanced/oop",
            "stage1/ch02-advanced/exceptions",
            "stage1/ch02-advanced/file-io",
            "stage1/ch02-advanced/functional",
            "stage1/ch02-advanced/iterators-generators",
            "stage1/ch02-advanced/type-hints",
          ],
        },
        {
          type: "category",
          label: "第3章 实战项目",
          items: [
            "stage1/ch03-projects/todo-cli",
            "stage1/ch03-projects/web-scraper",
            "stage1/ch03-projects/web-api",
            "stage1/ch03-projects/ai-api-experience",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "第二阶段：数据分析与可视化",
      collapsed: true,
      link: { type: "doc", id: "stage2/index" },
      items: [
        {
          type: "category",
          label: "第1章 Python到数据分析的过渡",
          items: ["stage2/ch01-warmup/pure-python-data"],
        },
        {
          type: "category",
          label: "第2章 NumPy 科学计算",
          items: [
            "stage2/ch02-numpy/overview",
            "stage2/ch02-numpy/array-basics",
            "stage2/ch02-numpy/indexing-slicing",
            "stage2/ch02-numpy/operations",
            "stage2/ch02-numpy/reshaping",
            "stage2/ch02-numpy/linear-algebra",
            "stage2/ch02-numpy/random-stats",
          ],
        },
        {
          type: "category",
          label: "第3章 Pandas 数据处理",
          items: [
            "stage2/ch03-pandas/core-structures",
            "stage2/ch03-pandas/read-write",
            "stage2/ch03-pandas/selection-filter",
            "stage2/ch03-pandas/data-cleaning",
            "stage2/ch03-pandas/data-transform",
            "stage2/ch03-pandas/groupby",
            "stage2/ch03-pandas/merge",
            "stage2/ch03-pandas/time-series",
          ],
        },
        {
          type: "category",
          label: "第4章 数据可视化",
          items: [
            "stage2/ch04-visualization/matplotlib",
            "stage2/ch04-visualization/seaborn",
            "stage2/ch04-visualization/plotly",
            "stage2/ch04-visualization/best-practices",
          ],
        },
        {
          type: "category",
          label: "第5章 数据库基础【选修】",
          items: [
            "stage2/ch05-database/relational-db",
            "stage2/ch05-database/sql-basics",
            "stage2/ch05-database/python-db",
            "stage2/ch05-database/db-design",
          ],
        },
        {
          type: "category",
          label: "第6章 实战项目",
          items: [
            "stage2/ch06-projects/eda-project",
            "stage2/ch06-projects/multi-source-analysis",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "第三阶段：AI 数学基础",
      collapsed: true,
      link: { type: "doc", id: "stage3/index" },
      items: [
        {
          type: "category",
          label: "第1章 线性代数实战",
          items: [
            "stage3/ch01-linear-algebra/vectors",
            "stage3/ch01-linear-algebra/matrices",
            "stage3/ch01-linear-algebra/eigenvalues",
            "stage3/ch01-linear-algebra/vector-spaces",
          ],
        },
        {
          type: "category",
          label: "第2章 概率与统计实战",
          items: [
            "stage3/ch02-probability/probability-basics",
            "stage3/ch02-probability/distributions",
            "stage3/ch02-probability/statistical-inference",
            "stage3/ch02-probability/information-theory",
          ],
        },
        {
          type: "category",
          label: "第3章 微积分与优化实战",
          items: [
            "stage3/ch03-calculus/derivatives",
            "stage3/ch03-calculus/partial-derivatives-gradient",
            "stage3/ch03-calculus/gradient-descent",
            "stage3/ch03-calculus/chain-rule-backprop",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "第四阶段：机器学习",
      collapsed: true,
      link: { type: "doc", id: "stage4/index" },
      items: [
        {
          type: "category",
          label: "第1章 机器学习基础概念",
          items: [
            "stage4/ch01-ml-basics/what-is-ml",
            "stage4/ch01-ml-basics/sklearn-intro",
          ],
        },
        {
          type: "category",
          label: "第2章 监督学习核心算法",
          items: [
            "stage4/ch02-supervised/linear-regression",
            "stage4/ch02-supervised/logistic-regression",
            "stage4/ch02-supervised/decision-trees",
            "stage4/ch02-supervised/ensemble-learning",
          ],
        },
        {
          type: "category",
          label: "第3章 无监督学习算法",
          items: [
            "stage4/ch03-unsupervised/clustering",
            "stage4/ch03-unsupervised/dimensionality-reduction",
            "stage4/ch03-unsupervised/anomaly-detection",
          ],
        },
        {
          type: "category",
          label: "第4章 模型评估与选择",
          items: [
            "stage4/ch04-evaluation/metrics",
            "stage4/ch04-evaluation/cross-validation",
            "stage4/ch04-evaluation/bias-variance",
            "stage4/ch04-evaluation/hyperparameter-tuning",
          ],
        },
        {
          type: "category",
          label: "第5章 特征工程",
          items: [
            "stage4/ch05-feature-engineering/feature-understanding",
            "stage4/ch05-feature-engineering/preprocessing",
            "stage4/ch05-feature-engineering/feature-construction",
            "stage4/ch05-feature-engineering/feature-selection",
            "stage4/ch05-feature-engineering/pipeline",
          ],
        },
        {
          type: "category",
          label: "第6章 实战项目",
          items: [
            "stage4/ch06-projects/house-price",
            "stage4/ch06-projects/customer-churn",
            "stage4/ch06-projects/user-segmentation",
            "stage4/ch06-projects/kaggle",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "第五阶段：深度学习基础",
      collapsed: true,
      link: { type: "doc", id: "stage5/index" },
      items: [
        {
          type: "category",
          label: "第1章 神经网络基础",
          items: [
            "stage5/ch01-nn-basics/neurons-activation",
            "stage5/ch01-nn-basics/forward-backward",
            "stage5/ch01-nn-basics/optimizers",
            "stage5/ch01-nn-basics/regularization",
            "stage5/ch01-nn-basics/weight-init",
          ],
        },
        {
          type: "category",
          label: "第2章 PyTorch 框架",
          items: [
            "stage5/ch02-pytorch/sklearn-to-pytorch-bridge",
            "stage5/ch02-pytorch/pytorch-basics",
            "stage5/ch02-pytorch/autograd",
            "stage5/ch02-pytorch/nn-module",
            "stage5/ch02-pytorch/data-loading",
            "stage5/ch02-pytorch/training-loop",
            "stage5/ch02-pytorch/practical-tips",
          ],
        },
        {
          type: "category",
          label: "第3章 卷积神经网络 (CNN)",
          items: [
            "stage5/ch03-cnn/convolution-basics",
            "stage5/ch03-cnn/cnn-structure",
            "stage5/ch03-cnn/classic-architectures",
            "stage5/ch03-cnn/transfer-learning",
            "stage5/ch03-cnn/image-classification-practice",
          ],
        },
        {
          type: "category",
          label: "第4章 RNN 与序列模型",
          items: [
            "stage5/ch04-rnn/rnn-basics",
            "stage5/ch04-rnn/lstm-gru",
            "stage5/ch04-rnn/sequence-practice",
          ],
        },
        {
          type: "category",
          label: "第5章 注意力与 Transformer",
          items: [
            "stage5/ch05-transformer/attention-mechanism",
            "stage5/ch05-transformer/transformer-architecture",
          ],
        },
        {
          type: "category",
          label: "第6章 生成模型基础【选修】",
          items: [
            "stage5/ch06-generative/gan",
            "stage5/ch06-generative/vae",
          ],
        },
        {
          type: "category",
          label: "第7章 训练技巧与调优",
          items: [
            "stage5/ch07-training-tips/hyperparameter-tuning",
            "stage5/ch07-training-tips/training-diagnosis",
            "stage5/ch07-training-tips/model-compression",
          ],
        },
        {
          type: "category",
          label: "第8章 实战项目",
          items: [
            "stage5/ch08-projects/image-classification",
            "stage5/ch08-projects/sentiment-analysis",
            "stage5/ch08-projects/generative-practice",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "第六阶段：计算机视觉【方向选修】",
      collapsed: true,
      link: { type: "doc", id: "stage6/index" },
      items: [
        {
          type: "category",
          label: "第1章 CV基础与OpenCV",
          items: [
            "stage6/ch01-cv-basics/image-fundamentals",
            "stage6/ch01-cv-basics/opencv-basics",
            "stage6/ch01-cv-basics/image-processing",
          ],
        },
        {
          type: "category",
          label: "第2章 图像分类进阶",
          items: [
            "stage6/ch02-classification/data-augmentation",
            "stage6/ch02-classification/modern-architectures",
            "stage6/ch02-classification/training-tricks",
          ],
        },
        {
          type: "category",
          label: "第3章 目标检测",
          items: [
            "stage6/ch03-detection/detection-overview",
            "stage6/ch03-detection/classic-detectors",
            "stage6/ch03-detection/yolo-series",
            "stage6/ch03-detection/detection-practice",
          ],
        },
        {
          type: "category",
          label: "第4章 图像分割",
          items: [
            "stage6/ch04-segmentation/semantic-segmentation",
            "stage6/ch04-segmentation/instance-segmentation",
            "stage6/ch04-segmentation/segmentation-practice",
          ],
        },
        {
          type: "category",
          label: "第5章 CV进阶专题",
          items: [
            "stage6/ch05-advanced/face-detection",
            "stage6/ch05-advanced/video-analysis",
            "stage6/ch05-advanced/ocr",
            "stage6/ch05-advanced/04-3d-vision",
          ],
        },
        {
          type: "category",
          label: "第6章 综合项目",
          items: [
            "stage6/ch06-projects/security-detection",
            "stage6/ch06-projects/medical-imaging",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "第七阶段：自然语言处理【条件必修】",
      collapsed: true,
      link: { type: "doc", id: "stage7/index" },
      items: [
        {
          type: "category",
          label: "第1章 文本处理基础",
          items: [
            "stage7/ch01-text-basics/nlp-overview",
            "stage7/ch01-text-basics/text-preprocessing",
            "stage7/ch01-text-basics/text-representation",
          ],
        },
        {
          type: "category",
          label: "第2章 词嵌入与语言模型",
          items: [
            "stage7/ch02-embeddings/word-embedding",
            "stage7/ch02-embeddings/contextual-embedding",
            "stage7/ch02-embeddings/language-models",
          ],
        },
        {
          type: "category",
          label: "第3章 文本分类",
          items: [
            "stage7/ch03-classification/traditional-methods",
            "stage7/ch03-classification/deep-learning-methods",
            "stage7/ch03-classification/classification-practice",
          ],
        },
        {
          type: "category",
          label: "第4章 序列标注",
          items: [
            "stage7/ch04-sequence-labeling/ner-overview",
            "stage7/ch04-sequence-labeling/bilstm-crf",
            "stage7/ch04-sequence-labeling/ner-practice",
          ],
        },
        {
          type: "category",
          label: "第5章 Seq2Seq 与注意力",
          items: [
            "stage7/ch05-seq2seq/encoder-decoder",
            "stage7/ch05-seq2seq/attention-in-nlp",
            "stage7/ch05-seq2seq/machine-translation",
          ],
        },
        {
          type: "category",
          label: "第6章 预训练语言模型",
          items: [
            "stage7/ch06-pretrained/pretrain-paradigm",
            "stage7/ch06-pretrained/bert",
            "stage7/ch06-pretrained/gpt-series",
            "stage7/ch06-pretrained/t5",
            "stage7/ch06-pretrained/transformers-library",
          ],
        },
        {
          type: "category",
          label: "第7章 综合项目",
          items: [
            "stage7/ch07-projects/qa-system",
            "stage7/ch07-projects/text-summarization",
            "stage7/ch07-projects/information-extraction",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "第八A阶段：大模型原理与微调",
      collapsed: true,
      link: { type: "doc", id: "stage8a/index" },
      items: [
        {
          type: "category",
          label: "第1章 NLP核心速成",
          items: [
            "stage8a/ch01-nlp-crash/tokenizer",
            "stage8a/ch01-nlp-crash/embeddings",
            "stage8a/ch01-nlp-crash/pretrained-models",
            "stage8a/ch01-nlp-crash/huggingface-quickstart",
          ],
        },
        {
          type: "category",
          label: "第2章 大语言模型概述",
          items: [
            "stage8a/ch02-llm-overview/development-history",
            "stage8a/ch02-llm-overview/core-concepts",
            "stage8a/ch02-llm-overview/industry-landscape",
          ],
        },
        {
          type: "category",
          label: "第3章 Transformer 架构深入",
          items: [
            "stage8a/ch03-transformer-deep/architecture-review",
            "stage8a/ch03-transformer-deep/model-variants",
            "stage8a/ch03-transformer-deep/efficient-attention",
            "stage8a/ch03-transformer-deep/scale-computation",
          ],
        },
        {
          type: "category",
          label: "第4章 预训练技术",
          items: [
            "stage8a/ch04-pretraining/pretraining-data",
            "stage8a/ch04-pretraining/pretraining-methods",
            "stage8a/ch04-pretraining/pretraining-engineering",
          ],
        },
        {
          type: "category",
          label: "第5章 Prompt Engineering",
          items: [
            "stage8a/ch05-prompt/prompt-basics",
            "stage8a/ch05-prompt/advanced-prompting",
            "stage8a/ch05-prompt/structured-output",
            "stage8a/ch05-prompt/prompt-practice",
          ],
        },
        {
          type: "category",
          label: "第6章 微调技术",
          items: [
            "stage8a/ch06-finetuning/finetuning-overview",
            "stage8a/ch06-finetuning/lora-qlora",
            "stage8a/ch06-finetuning/other-peft",
            "stage8a/ch06-finetuning/finetuning-practice",
            "stage8a/ch06-finetuning/data-labeling",
          ],
        },
        {
          type: "category",
          label: "第7章 RLHF 与对齐",
          items: [
            "stage8a/ch07-alignment/alignment-problem",
            "stage8a/ch07-alignment/rlhf",
            "stage8a/ch07-alignment/alternative-methods",
          ],
        },
        {
          type: "category",
          label: "第8章 综合项目",
          items: ["stage8a/ch08-projects/domain-finetuning"],
        },
      ],
    },
    {
      type: "category",
      label: "第八B阶段：大模型应用与工程化",
      collapsed: true,
      link: { type: "doc", id: "stage8b/index" },
      items: [
        {
          type: "category",
          label: "第1章 RAG 检索增强生成",
          items: [
            "stage8b/ch01-rag/rag-basics",
            "stage8b/ch01-rag/document-processing",
            "stage8b/ch01-rag/vector-databases",
            "stage8b/ch01-rag/retrieval-strategies",
            "stage8b/ch01-rag/rag-optimization",
            "stage8b/ch01-rag/advanced-rag",
            "stage8b/ch01-rag/rag-evaluation",
          ],
        },
        {
          type: "category",
          label: "第2章 LLM 本地部署",
          items: [
            "stage8b/ch02-deployment/local-models",
            "stage8b/ch02-deployment/inference-servers",
            "stage8b/ch02-deployment/unified-api",
          ],
        },
        {
          type: "category",
          label: "第3章 大模型应用开发",
          items: [
            "stage8b/ch03-app-dev/llm-api-practice",
            "stage8b/ch03-app-dev/langchain-basics",
            "stage8b/ch03-app-dev/function-calling",
            "stage8b/ch03-app-dev/huggingface-deep",
            "stage8b/ch03-app-dev/dialog-system",
            "stage8b/ch03-app-dev/ai-assisted-coding",
          ],
        },
        {
          type: "category",
          label: "第4章 AI 工程化实践",
          items: [
            "stage8b/ch04-engineering/async-programming",
            "stage8b/ch04-engineering/api-design",
            "stage8b/ch04-engineering/logging-monitoring",
            "stage8b/ch04-engineering/docker-deployment",
          ],
        },
        {
          type: "category",
          label: "第5章 综合项目",
          items: [
            "stage8b/ch05-projects/enterprise-kb",
            "stage8b/ch05-projects/domain-rag-finetuning",
            "stage8b/ch05-projects/intelligent-assistant",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "第九阶段：AI Agent 与智能体",
      collapsed: true,
      link: { type: "doc", id: "stage9/index" },
      items: [
        {
          type: "category",
          label: "第1章 Agent 基础概念",
          items: [
            "stage9/ch01-agent-basics/what-is-agent",
            "stage9/ch01-agent-basics/development-history",
            "stage9/ch01-agent-basics/capability-levels",
            "stage9/ch01-agent-basics/system-architecture",
          ],
        },
        {
          type: "category",
          label: "第2章 推理与规划",
          items: [
            "stage9/ch02-reasoning/llm-reasoning",
            "stage9/ch02-reasoning/chain-reasoning",
            "stage9/ch02-reasoning/react",
            "stage9/ch02-reasoning/plan-and-execute",
            "stage9/ch02-reasoning/advanced-planning",
            "stage9/ch02-reasoning/reasoning-evaluation",
          ],
        },
        {
          type: "category",
          label: "第3章 工具使用与 Function Calling",
          items: [
            "stage9/ch03-tools/function-calling-deep",
            "stage9/ch03-tools/tool-description",
            "stage9/ch03-tools/tool-strategies",
            "stage9/ch03-tools/common-tools",
            "stage9/ch03-tools/tool-safety",
            "stage9/ch03-tools/advanced-patterns",
            "stage9/ch03-tools/code-agent",
            "stage9/ch03-tools/multi-tool-practice",
          ],
        },
        {
          type: "category",
          label: "第4章 记忆系统",
          items: [
            "stage9/ch04-memory/memory-overview",
            "stage9/ch04-memory/short-term-memory",
            "stage9/ch04-memory/long-term-memory",
            "stage9/ch04-memory/episodic-procedural",
            "stage9/ch04-memory/memory-engineering",
            "stage9/ch04-memory/memory-practice",
          ],
        },
        {
          type: "category",
          label: "第5章 MCP 协议",
          items: [
            "stage9/ch05-mcp/mcp-overview",
            "stage9/ch05-mcp/mcp-architecture",
            "stage9/ch05-mcp/mcp-server-dev",
            "stage9/ch05-mcp/mcp-client-integration",
            "stage9/ch05-mcp/mcp-ecosystem",
          ],
        },
        {
          type: "category",
          label: "第6章 Agent 开发框架",
          items: [
            "stage9/ch06-frameworks/framework-overview",
            "stage9/ch06-frameworks/langchain-langgraph",
            "stage9/ch06-frameworks/llamaindex",
            "stage9/ch06-frameworks/crewai",
            "stage9/ch06-frameworks/autogen",
            "stage9/ch06-frameworks/openai-agents-sdk",
            "stage9/ch06-frameworks/low-code-platforms",
            "stage9/ch06-frameworks/framework-selection",
          ],
        },
        {
          type: "category",
          label: "第7章 多 Agent 系统",
          items: [
            "stage9/ch07-multi-agent/architecture-patterns",
            "stage9/ch07-multi-agent/communication",
            "stage9/ch07-multi-agent/task-coordination",
            "stage9/ch07-multi-agent/practice-patterns",
            "stage9/ch07-multi-agent/challenges",
            "stage9/ch07-multi-agent/multi-agent-practice",
          ],
        },
        {
          type: "category",
          label: "第8章 Agent 评估与安全",
          items: [
            "stage9/ch08-eval-safety/evaluation-methods",
            "stage9/ch08-eval-safety/benchmarks",
            "stage9/ch08-eval-safety/agent-security",
            "stage9/ch08-eval-safety/guardrails",
            "stage9/ch08-eval-safety/observability",
          ],
        },
        {
          type: "category",
          label: "第9章 Agent 部署与运维",
          items: [
            "stage9/ch09-deployment/deployment-architecture",
            "stage9/ch09-deployment/runtime-management",
            "stage9/ch09-deployment/persistence-recovery",
            "stage9/ch09-deployment/cost-optimization",
            "stage9/ch09-deployment/production-best-practices",
          ],
        },
        {
          type: "category",
          label: "第10章 综合项目",
          items: [
            "stage9/ch10-projects/research-assistant",
            "stage9/ch10-projects/data-analysis-agent",
            "stage9/ch10-projects/multi-agent-dev-team",
          ],
        },
      ],
    },
    {
      type: "category",
      label: "第十阶段：AIGC 与多模态【方向选修】",
      collapsed: true,
      link: { type: "doc", id: "stage10/index" },
      items: [
        {
          type: "category",
          label: "第1章 多模态大模型",
          items: [
            "stage10/ch01-multimodal/multimodal-basics",
            "stage10/ch01-multimodal/vision-language",
            "stage10/ch01-multimodal/multimodal-apps",
          ],
        },
        {
          type: "category",
          label: "第2章 图像生成",
          items: [
            "stage10/ch02-image-gen/diffusion-models",
            "stage10/ch02-image-gen/stable-diffusion",
            "stage10/ch02-image-gen/sd-applications",
            "stage10/ch02-image-gen/sd-finetuning",
            "stage10/ch02-image-gen/latest-progress",
          ],
        },
        {
          type: "category",
          label: "第3章 视频生成与数字人",
          items: [
            "stage10/ch03-video-gen/video-generation",
            "stage10/ch03-video-gen/tts",
            "stage10/ch03-video-gen/digital-human",
          ],
        },
        {
          type: "category",
          label: "第4章 AIGC 前沿与伦理",
          items: [
            "stage10/ch04-frontier/frontier-trends",
            "stage10/ch04-frontier/ai-ethics",
            "stage10/ch04-frontier/ai-regulations",
          ],
        },
        {
          type: "category",
          label: "第5章 综合项目",
          items: ["stage10/ch05-projects/creative-platform"],
        },
      ],
    },
    {
      type: "category",
      label: "📦 选修模块",
      collapsed: true,
      items: [
        {
          type: "category",
          label: "模块A：C++ 与模型部署",
          link: { type: "doc", id: "electives/module-a/index" },
          items: [
            "electives/module-a/cpp-basics",
            "electives/module-a/cpp-advanced",
            "electives/module-a/model-optimization",
            "electives/module-a/inference-engines",
            "electives/module-a/edge-deployment",
            "electives/module-a/model-serving",
            "electives/module-a/projects",
          ],
        },
        {
          type: "category",
          label: "模块B：Python 进阶专题",
          link: { type: "doc", id: "electives/module-b/index" },
          items: [
            "electives/module-b/decorators-advanced",
            "electives/module-b/iterators-advanced",
            "electives/module-b/concurrency",
            "electives/module-b/metaprogramming",
          ],
        },
        {
          type: "category",
          label: "模块C：经典ML补充算法",
          link: { type: "doc", id: "electives/module-c/index" },
          items: [
            "electives/module-c/svm",
            "electives/module-c/knn",
            "electives/module-c/naive-bayes",
            "electives/module-c/lda",
          ],
        },
        "electives/module-d",
        "electives/module-e",
        "electives/module-f",
      ],
    },
    {
      type: "category",
      label: "📎 附录",
      collapsed: true,
      items: [
        "appendix/resources",
        "appendix/hardware",
        "appendix/faq",
        "appendix/continuous-learning",
        "appendix/troubleshooting",
        "appendix/resource-quick-ref",
        "appendix/job-prep",
      ],
    },
  ],
};

module.exports = sidebars;
