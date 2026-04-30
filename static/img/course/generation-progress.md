# Course Image Generation Progress

Last updated: 2026-04-30

Status: completed.

Generated with image2: 577 / 577

Detection note: all generated image2 PNG files are currently larger than the local preview placeholders.

编号说明：课程源码目录已经和网页展示章节号对齐；例如 `docs/ch01-tools` 对应第 1 章“开发者工具基础”，`docs/ch05-machine-learning` 对应第 5 章“机器学习入门到实战”。

第 4 章说明：目录 `docs/ch04-ai-math` 下原有 12 张数学补充图已经用 image2 重新生成，并覆盖早前的本地预览图；本轮新增 18 张真实 PNG，用于替换数学首页、学习指南、线性代数、概率统计、微积分与优化页面的入口学习地图。

第 5 章说明：目录 `docs/ch05-machine-learning` 下原有 22 张机器学习补充图已经用 image2 生成；本轮新增 24 张真实 PNG，用于替换机器学习首页、学习指南、基础概念、监督学习、无监督学习、模型评估、特征工程和项目实战页面的入口学习地图。新增图遵循“视觉优先、中英文自然混用”规则，公式、API、变量名和标准术语保留英文或数学形式。

第 5 章深度图解说明：本轮新增 16 张诊断型和流程型图解，覆盖训练/验证/测试泄漏护栏、sklearn Pipeline 组件、线性回归残差诊断、逻辑回归阈值权衡、决策树剪枝、集成学习纠错、聚类形状选择、PCA 方差解释、异常检测方法对比、ROC/PR 曲线读图、交叉验证防泄漏、学习曲线诊断、调参预算、特征泄漏红旗、ColumnTransformer 真实表格流水线和项目报告故事板。新增图用于把文字密集段落转成可扫读的读图提示，让新人更容易形成判断动作。

第 6～9 章基础说明：目录 `docs/ch06-deep-learning`、`docs/ch07-llm-principles`、`docs/ch08-rag`、`docs/ch09-agent` 下此前新增的 25 张机制图、流程图和系统图已经用 image2 生成，并插入到对应核心教程页面。

第 6 章补充说明：目录 `docs/ch06-deep-learning` 本轮新增 18 张入口学习地图已经用 image2 覆盖临时 Preview PNG，并插入到首页、学习指南、神经网络基础、PyTorch、CNN、RNN、Transformer、生成模型、训练技巧和项目实战页面。新增图遵循“视觉优先、中英文自然混用”规则，公式、API、变量名和标准术语保留英文或数学形式。

第 6 章深度图解说明：本轮新增 23 张机制型和诊断型图解，覆盖神经元线性打分、XOR 单层局限、反向传播责任分摊、NumPy 到 PyTorch 训练循环、优化器更新决策、正则化动作选择、Tensor shape 语义、Autograd 梯度生命周期、训练循环顺序、卷积 stride/padding、CNN 感受野、通道/空间权衡、迁移学习决策、RNN 滚动记忆、长依赖梯度衰减、LSTM 门控、Attention QKV 类比、Causal Mask、Transformer Block 职责、表示逐层精炼、GAN 对抗平衡、VAE 潜空间采样和训练诊断仪表盘。新增图用于把第 6 章抽象机制转成可扫读、可类比、可排障的学习支架，让新人更容易持续学下去。

第 7 章补充说明：目录 `docs/ch07-llm-principles` 本轮新增 18 张入口学习地图已经用 image2 生成，并插入到首页、学习指南、NLP 速成、LLM 概览、Transformer 深入、预训练、Prompt、微调、对齐和综合项目页面。新增图遵循“视觉优先、中英文自然混用”规则，公式、API、变量名和标准术语保留英文或数学形式。

第 7 章深度图解说明：本轮新增 27 张机制型、决策型和工程闭环图，覆盖 Tokenizer 粒度取舍、input_ids/attention_mask、Embedding 语义空间、上下文化表示、Next-token 生成、Context Window 预算、HuggingFace 工作流、Transformer block 数据流、架构 mask 与任务适配、高效注意力瓶颈、KV cache 与 MQA/GQA、规模成本旋钮、训练/推理成本差异、预训练数据治理、预训练目标对比、预训练工程生产线、Prompt 任务规格、高级 Prompt 技巧选择、结构化输出校验、微调决策、LoRA/QLoRA、PEFT 放置位置、数据标注飞轮、HHH 对齐张力、RLHF 奖励与 KL、DPO 捷径和垂直微调项目评估看板。新增图用于把第 7 章大模型原理、工程约束和项目决策转成新人可扫读的学习支架。

第 8 章补充说明：目录 `docs/ch08-rag` 本轮新增 17 张入口学习地图已经用 image2 生成，并插入到首页、学习指南、RAG 核心、模型部署、应用开发、工程化和综合项目页面。新增图遵循“视觉优先、中英文自然混用”规则，公式、API、变量名和标准术语保留英文或数学形式。

第 8 章深度图解说明：本轮新增 31 张机制型、决策型、工程型和项目闭环图，覆盖 RAG 分层故障定位、chunk size 与 overlap、课件知识块元数据、向量库记录与 metadata filter、精确搜索与 ANN、Hybrid Search 盲区、Query Rewrite 与 Rerank、RAG 优化排障、RAG 实验闭环、高级 RAG 架构选择、RAG 分层评估、Faithfulness 与 citation check、本地模型与云 API 决策、推理服务队列和 batch、统一 API 网关、LLM API 稳健客户端、LangChain 组件流水线、Function Calling 校验执行、HuggingFace 生态、多轮对话状态、AI 编程人工验证、PDF/Word/PPT 文档解析路由、模板渲染、异步并发控制、API 契约、可观测性、Docker 部署、企业知识库权限引用、RAG+微调职责拆分、智能助手 trace 和课件生成助手生产线。新增图用于把第 8 章从“能跑 RAG Demo”提升到“能理解、能调试、能交付 LLM 应用系统”。

第 9 章补充说明：目录 `docs/ch09-agent` 本轮新增 27 张入口学习地图已经用 image2 生成，并插入到首页、学习指南、Agent 基础、推理规划、工具、记忆、MCP、框架、多 Agent、评估安全、部署运维和综合项目页面。新增图遵循“视觉优先、中英文自然混用”规则，公式、API、变量名和标准术语保留英文或数学形式。

第 9 章深度图解说明：本轮新增 31 张机制型、工程型和项目型图解，覆盖工作流/聊天机器人/Agent 边界、Agent 行动 trace、系统架构数据流、链式推理自检、Plan-and-Execute 重规划、高级 DAG 规划、推理失败归因、Function Calling 校验、工具描述、工具安全、代码 Agent、记忆分层、长期记忆更新、记忆工程生命周期、MCP 消息流、MCP Server 契约、LangGraph 状态机、框架选型、多 Agent 模式、通信契约、协调成本、Agent 分层评估、Prompt Injection 安全隔离、Trace Span 可观测性、生产部署架构、Checkpoint/Event Log 恢复、成本路由缓存、生产 readiness、研究助手引用追踪、数据分析 Agent 可复核链路和多 Agent 开发团队交付闭环。新增图用于把第 9 章从“概念能读懂”推进到“系统怎么运行、怎么调试、怎么上线一眼能看懂”。

第 10～12 章补充说明：目录 `docs/ch10-computer-vision`、`docs/ch11-nlp`、`docs/ch12-multimodal` 本轮新增 28 张入口学习地图已经用 image2 生成，并插入到首页、学习指南、CV/NLP/多模态各子章导读和综合项目页面。新增图遵循“视觉优先、中英文自然混用”规则，公式、API、变量名和标准术语保留英文或数学形式。

第 10 章深度图解说明：本轮新增 19 张机制型、诊断型和项目型图解，覆盖图像数组 shape 与通道、OpenCV BGR/坐标/裁剪、经典图像处理操作选择、数据增强不变性风险、分类架构演进、分类训练诊断、检测输出与 IoU、R-CNN 家族共享特征、YOLO 阈值与 NMS、检测项目误检漏检分桶、语义分割 IoU 与边界、实例分割个体拆分、分割失败样本分桶、人脸识别阈值风险、视频抽帧跟踪与时序窗口、OCR 阅读顺序、3D 深度/视差/点云、安防告警去重和医学影像风险复核。新增图用于把第 10 章从“视觉任务名词可读”推进到“输入如何表示、模型输出如何评估、项目如何复盘更直观”。

第 11 章深度图解说明：本轮新增 14 张机制型、范式型和项目型图解，覆盖 NLP 任务全景、语言模型 next token 预测、传统文本分类基线、神经文本分类 embedding/pooling、BiLSTM-CRF 标签路径、NER 实体级评估、Seq2Seq 信息瓶颈、机器翻译错误分析、预训练迁移微调、T5 text-to-text 统一接口、Transformers 库调用链、问答系统证据闭环、摘要抽取/生成评估和信息抽取 schema 流程。新增图用于把第 11 章从“概念名词很多”推进到“文本如何被表示、任务如何被评估、项目如何交付一眼能看懂”。

第 12 章深度图解说明：本轮新增 9 张工程型、产品型和治理型图解，覆盖多模态应用工程链路、Stable Diffusion 应用模式选择、图像生成微调路线、图像生成趋势雷达、TTS 文本到语音链路、数字人多模块同步、AIGC 前沿趋势系统判断、伦理安全风险护栏和 AI 合规工程转译。新增图用于把第 12 章从“Demo 很炫”推进到“多模态系统如何设计、生成工作流如何交付、风险边界如何治理更直观”。

选修模块说明：目录 `docs/electives` 下原有 12 张 C++ 部署、Python 进阶、经典 ML、安全、前端和产品设计图已经用 image2 生成，并插入到对应选修页面。本轮补充 9 张模块入口和缺口页图，覆盖 C++ 部署总览、RAII/所有权、边缘部署、部署综合项目、Python 进阶总览、生成器管道、元编程注册器、经典 ML 总览和 LDA 监督式投影。

选修模块深度图解说明：本轮新增 9 张实战判断型图解，覆盖模型优化指标取舍、推理引擎选型矩阵、模型服务指标与版本路由、装饰器横切逻辑分层、asyncio 超时取消与限流、SVM 参数 C/kernel 决策、AI 安全威胁建模与回归集、AI 前端状态机和 AI 产品实验指标闭环。新增图用于把选修模块从“补充知识点”推进到“工程选型、上线运行、产品验证和安全治理怎么判断更直观”。

附录说明：目录 `docs/appendix` 下原有 10 张历史时间线、排障、硬件决策、求职、持续学习、资源、FAQ、速查、编号和视觉规划图已经用 image2 生成，并插入到对应附录页面。本轮补充 6 张 AI 历史谱系图，替换 `docs/appendix/ai-milestones.md` 中残留的 Mermaid 主线图，让论文、算法和课程项目的关系更直观。

第 1～3 章说明：目录 `docs/ch01-tools`、`docs/ch02-python`、`docs/ch03-data-analysis` 下新增的 59 张终端、环境、Git、Jupyter、Python 基础和数据分析补充图已经用 image2 生成，并插入到对应教程页面。其中第 1 章本轮新增 6 张真实 PNG，用于替换任务单、命令行、包管理器、Git 核心操作、远程仓库和 VS Code 页面的文字流程入口；第 2 章本轮新增 15 张真实 PNG，用于替换 Python 首页、学习指南、任务链、基础语法、文件处理、进阶机制和项目页的文字流程入口；第 3 章本轮新增 23 张真实 PNG，用于替换数据分析首页、学习指南、任务链、NumPy、Pandas、可视化和数据库页面的文字流程入口。

学习地图说明：目录 `docs/intro` 下新增的 8 张能力地图、现代技术栈、学习路线、四条主线、卡点诊断、项目作品集、角色路线和毕业项目闭环图已经用 image2 生成，并插入到对应导览页面。

历史节点补充说明：本轮新增 8 张 AI 历史与算法突破图，插入第 4 章概率统计历史、第 5 章机器学习历史与 SVM、第 6 章深度学习历史、第 9 章强化学习到 Agent、第 11 章 HMM/CRF、CTC/Deep Speech 和 AMR 语义图页面。新增图遵循“中文讲解为主，标准术语、论文名、公式和 API 保留英文”的规则。

## Completed

- `ai-fullstack-hero.png`
- `ai-learning-assistant-roadmap.png`
- `prompt-rag-agent-progression.png`
- `boss-challenge-map.png`
- `debug-detective-missions.png`
- `ch01-tools-foundation.png`
- `ch02-python-foundation.png`
- `ch03-data-visualization.png`
- `ch04-ai-math.png`
- `ch04-learning-quest-map.png`
- `ch04-ai-math-backbone.png`
- `ch04-study-guide-math-minimum-loop.png`
- `ch04-linear-algebra-chapter-flow.png`
- `ch04-vector-ai-meaning-map.png`
- `ch04-matrix-batch-transform-flow.png`
- `ch04-eigen-pca-direction-map.png`
- `ch04-vector-space-high-level-map.png`
- `ch04-probability-chapter-flow.png`
- `ch04-probability-bayes-update-flow.png`
- `ch04-probability-history-foundations-map.png`
- `ch04-distribution-random-world-map.png`
- `ch04-statistical-inference-data-to-parameter.png`
- `ch04-information-theory-loss-map.png`
- `ch04-calculus-training-flow.png`
- `ch04-derivative-change-rate-bridge.png`
- `ch04-gradient-parameter-knobs-map.png`
- `ch04-gradient-descent-iteration-loop.png`
- `ch04-backprop-chain-rule-training-bridge.png`
- `ch06-learning-quest-map.png`
- `ch06-training-loop-backbone.png`
- `ch06-study-guide-training-loop.png`
- `ch06-nn-basics-chapter-flow.png`
- `ch06-ml-to-dl-bridge-map.png`
- `ch06-dl-history-breakthrough-map.png`
- `ch06-weight-init-signal-stability-map.png`
- `ch06-pytorch-chapter-flow.png`
- `ch06-sklearn-to-pytorch-shift-map.png`
- `ch06-pytorch-tensor-lifecycle-map.png`
- `ch06-nn-module-parameter-flow.png`
- `ch06-cnn-chapter-flow.png`
- `ch06-rnn-chapter-flow.png`
- `ch06-transformer-chapter-flow.png`
- `ch06-transformer-global-context-map.png`
- `ch06-generative-chapter-flow.png`
- `ch06-training-tips-chapter-flow.png`
- `ch06-projects-portfolio-loop.png`
- `ch06-deep-learning-project-cycle.png`
- `ch06-neuron-linear-activation-gate.png`
- `ch06-xor-single-layer-limit-map.png`
- `ch06-backprop-error-responsibility-map.png`
- `ch06-numpy-to-pytorch-training-loop-map.png`
- `ch06-optimizer-gradient-to-update-map.png`
- `ch06-regularization-overfit-action-map.png`
- `ch06-tensor-shape-meaning-map.png`
- `ch06-autograd-gradient-lifecycle-map.png`
- `ch06-training-loop-order-guardrail.png`
- `ch06-conv-stride-padding-size-map.png`
- `ch06-cnn-receptive-field-growth-map.png`
- `ch06-cnn-channel-spatial-tradeoff-map.png`
- `ch06-transfer-learning-freeze-finetune-map.png`
- `ch06-rnn-hidden-state-rolling-memory-map.png`
- `ch06-rnn-long-dependency-vanishing-map.png`
- `ch06-lstm-gates-information-control-map.png`
- `ch06-attention-qkv-library-analogy-map.png`
- `ch06-causal-mask-no-peeking-map.png`
- `ch06-transformer-block-role-map.png`
- `ch06-transformer-representation-refinement-map.png`
- `ch06-gan-adversarial-balance-map.png`
- `ch06-vae-latent-continuity-sampling-map.png`
- `ch06-training-diagnosis-dashboard-map.png`
- `ch05-machine-learning.png`
- `ch05-learning-quest-map.png`
- `ch05-modeling-loop-backbone.png`
- `ch05-study-guide-project-loop.png`
- `ch05-basics-chapter-flow.png`
- `ch05-ml-history-breakthrough-map.png`
- `ch05-task-type-decision-map.png`
- `ch05-sklearn-fit-predict-loop.png`
- `ch05-math-to-ml-training-map.png`
- `ch05-supervised-chapter-flow.png`
- `ch05-linear-regression-learning-flow.png`
- `ch05-logistic-classification-flow.png`
- `ch05-decision-tree-learning-flow.png`
- `ch05-ensemble-bagging-boosting-flow.png`
- `ch05-svm-margin-map.png`
- `ch05-unsupervised-chapter-flow.png`
- `ch05-clustering-decision-flow.png`
- `ch05-dimensionality-reduction-purpose-map.png`
- `ch05-anomaly-detection-decision-flow.png`
- `ch05-evaluation-chapter-flow.png`
- `ch05-metrics-selection-flow.png`
- `ch05-cross-validation-stability-flow.png`
- `ch05-bias-variance-action-map.png`
- `ch05-hyperparameter-tuning-workflow.png`
- `ch05-feature-engineering-chapter-flow.png`
- `ch05-feature-understanding-workflow.png`
- `ch05-projects-portfolio-loop.png`
- `ch05-data-split-leakage-guardrail.png`
- `ch05-sklearn-pipeline-anatomy.png`
- `ch05-linear-regression-residual-diagnostics.png`
- `ch05-logistic-threshold-tradeoff.png`
- `ch05-tree-pruning-overfit-map.png`
- `ch05-ensemble-error-correction-lab.png`
- `ch05-clustering-shape-selection-map.png`
- `ch05-pca-explained-variance-map.png`
- `ch05-anomaly-method-comparison-map.png`
- `ch05-threshold-roc-pr-curve-map.png`
- `ch05-cv-leakage-safe-pipeline-map.png`
- `ch05-learning-curve-diagnosis-map.png`
- `ch05-search-space-budget-map.png`
- `ch05-feature-leakage-red-flags-map.png`
- `ch05-columntransformer-real-table-pipeline.png`
- `ch05-project-report-storyboard.png`
- `ch06-deep-learning.png`
- `ch10-computer-vision.png`
- `ch10-learning-quest-map.png`
- `ch10-visual-task-progression-map.png`
- `ch10-study-guide-output-granularity-map.png`
- `ch10-cv-basics-chapter-flow.png`
- `ch10-classification-chapter-flow.png`
- `ch10-detection-chapter-flow.png`
- `ch10-segmentation-chapter-flow.png`
- `ch10-advanced-vision-route-map.png`
- `ch10-projects-delivery-loop.png`
- `ch10-image-array-shape-channel-map.png`
- `ch10-opencv-bgr-coordinate-crop-map.png`
- `ch10-image-processing-operation-decision-map.png`
- `ch10-augmentation-invariance-risk-map.png`
- `ch10-classification-architecture-evolution-map.png`
- `ch10-classification-training-diagnosis-map.png`
- `ch10-detection-output-iou-error-map.png`
- `ch10-classic-detectors-shared-feature-map.png`
- `ch10-yolo-threshold-nms-map.png`
- `ch10-detection-practice-eval-buckets-map.png`
- `ch10-semantic-segmentation-iou-boundary-map.png`
- `ch10-instance-segmentation-count-mask-map.png`
- `ch10-segmentation-practice-failure-buckets-map.png`
- `ch10-face-recognition-threshold-pipeline-map.png`
- `ch10-video-frame-tracking-temporal-window-map.png`
- `ch10-ocr-layout-reading-order-map.png`
- `ch10-3d-depth-disparity-pointcloud-map.png`
- `ch10-security-detection-alert-dedup-map.png`
- `ch10-medical-imaging-risk-review-map.png`
- `ch11-nlp.png`
- `ch11-learning-quest-map.png`
- `ch11-nlp-to-llm-backbone.png`
- `ch11-study-guide-text-to-model-map.png`
- `ch11-text-basics-chapter-flow.png`
- `ch11-embeddings-chapter-flow.png`
- `ch11-classification-chapter-flow.png`
- `ch11-sequence-labeling-chapter-flow.png`
- `ch11-hmm-crf-sequence-history-map.png`
- `ch11-seq2seq-chapter-flow.png`
- `ch11-ctc-deep-speech-asr-map.png`
- `ch11-pretrained-chapter-flow.png`
- `ch11-projects-delivery-loop.png`
- `ch11-amr-semantic-graph-map.png`
- `ch11-nlp-task-landscape-map.png`
- `ch11-language-model-next-token-stack.png`
- `ch11-traditional-classification-baseline-map.png`
- `ch11-neural-classification-embedding-pooling-map.png`
- `ch11-bilstm-crf-label-path-map.png`
- `ch11-ner-project-entity-eval-loop.png`
- `ch11-seq2seq-encoder-decoder-bottleneck-map.png`
- `ch11-machine-translation-error-analysis-map.png`
- `ch11-pretraining-transfer-finetune-map.png`
- `ch11-t5-text-to-text-task-unification-map.png`
- `ch11-transformers-library-call-chain-map.png`
- `ch11-qa-retrieval-answer-evaluation-map.png`
- `ch11-summarization-extractive-generative-eval-map.png`
- `ch11-information-extraction-schema-pipeline-map.png`
- `ch07-llm-principles.png`
- `ch07-learning-quest-map.png`
- `ch07-llm-capability-backbone.png`
- `ch07-study-guide-evolution-line.png`
- `ch07-nlp-crash-chapter-flow.png`
- `ch07-llm-overview-chapter-flow.png`
- `ch07-llm-capability-stack.png`
- `ch07-transformer-deep-chapter-flow.png`
- `ch07-transformer-cost-task-map.png`
- `ch07-pretraining-chapter-flow.png`
- `ch07-pretraining-data-objective-engineering-map.png`
- `ch07-prompt-chapter-flow.png`
- `ch07-prompt-iteration-loop.png`
- `ch07-finetuning-chapter-flow.png`
- `ch07-finetuning-decision-loop.png`
- `ch07-alignment-chapter-flow.png`
- `ch07-alignment-app-safety-map.png`
- `ch07-projects-route-map.png`
- `ch07-project-method-choice-loop.png`
- `ch07-tokenizer-granularity-tradeoff-map.png`
- `ch07-tokenizer-inputids-mask-length-map.png`
- `ch07-embedding-onehot-dense-map.png`
- `ch07-contextual-embedding-sense-map.png`
- `ch07-next-token-generation-loop-map.png`
- `ch07-context-window-budget-map.png`
- `ch07-huggingface-workflow-object-map.png`
- `ch07-transformer-block-dataflow-map.png`
- `ch07-architecture-mask-task-fit-map.png`
- `ch07-efficient-attention-bottleneck-map.png`
- `ch07-kv-cache-mqa-gqa-map.png`
- `ch07-scale-cost-knobs-map.png`
- `ch07-train-inference-cost-split-map.png`
- `ch07-pretraining-data-governance-funnel.png`
- `ch07-pretraining-objective-comparison-map.png`
- `ch07-pretraining-engineering-production-line.png`
- `ch07-prompt-spec-three-layer-map.png`
- `ch07-advanced-prompt-technique-decision-map.png`
- `ch07-structured-output-contract-validation-map.png`
- `ch07-finetune-decision-rag-prompt-peft-map.png`
- `ch07-lora-qlora-low-rank-memory-map.png`
- `ch07-peft-placement-family-map.png`
- `ch07-data-labeling-flywheel-review-map.png`
- `ch07-alignment-hhh-tension-guardrail-map.png`
- `ch07-rlhf-reward-kl-loop-map.png`
- `ch07-dpo-rlhf-shortcut-map.png`
- `ch07-domain-finetune-evaluation-board-map.png`
- `ch08-rag-engineering.png`
- `ch08-learning-quest-map.png`
- `ch08-rag-system-backbone.png`
- `ch08-ragops-improvement-loop.png`
- `ch08-study-guide-four-layer-map.png`
- `ch08-rag-position-bridge.png`
- `ch08-rag-core-chapter-flow.png`
- `ch08-rag-data-to-answer-pipeline.png`
- `ch08-deployment-chapter-flow.png`
- `ch08-model-serving-decision-map.png`
- `ch08-app-dev-chapter-flow.png`
- `ch08-app-dev-learning-order-map.png`
- `ch08-llm-app-capability-loop.png`
- `ch08-engineering-chapter-flow.png`
- `ch08-llmops-trace-loop.png`
- `ch08-projects-route-map.png`
- `ch08-project-learning-order-map.png`
- `ch08-project-delivery-loop.png`
- `ch08-rag-layer-failure-debug-map.png`
- `ch08-chunk-size-overlap-tradeoff-map.png`
- `ch08-courseware-chunk-metadata-schema-map.png`
- `ch08-vector-record-metadata-filter-map.png`
- `ch08-ann-exact-search-tradeoff-map.png`
- `ch08-hybrid-retrieval-blindspot-map.png`
- `ch08-rerank-query-rewrite-funnel-map.png`
- `ch08-rag-optimization-debug-funnel-map.png`
- `ch08-rag-experiment-eval-loop-map.png`
- `ch08-advanced-rag-architecture-selection-map.png`
- `ch08-rag-evaluation-layered-dashboard-map.png`
- `ch08-faithfulness-citation-check-map.png`
- `ch08-local-model-api-decision-map.png`
- `ch08-inference-serving-queue-batch-map.png`
- `ch08-unified-api-provider-gateway-map.png`
- `ch08-llm-api-robust-client-loop-map.png`
- `ch08-langchain-component-pipeline-map.png`
- `ch08-function-calling-validation-dispatch-map.png`
- `ch08-huggingface-ecosystem-layers-map.png`
- `ch08-dialog-state-slot-memory-map.png`
- `ch08-ai-coding-human-review-loop-map.png`
- `ch08-document-parsing-format-router-map.png`
- `ch08-template-schema-to-render-map.png`
- `ch08-async-concurrency-semaphore-timeout-map.png`
- `ch08-api-contract-error-version-map.png`
- `ch08-observability-logs-metrics-trace-map.png`
- `ch08-docker-image-container-compose-map.png`
- `ch08-enterprise-kb-permission-citation-map.png`
- `ch08-rag-finetune-responsibility-split-map.png`
- `ch08-assistant-session-tool-trace-map.png`
- `ch08-courseware-assistant-production-line-map.png`
- `ch09-agent-systems.png`
- `ch09-learning-quest-map.png`
- `ch09-agent-vs-workflow-backbone.png`
- `ch09-agentops-control-loop.png`
- `ch09-study-guide-minimal-agent-loop.png`
- `ch09-basics-position-bridge.png`
- `ch09-basics-chapter-flow.png`
- `ch09-basics-execution-loop.png`
- `ch09-rl-agent-breakthroughs-map.png`
- `ch09-reasoning-chapter-flow.png`
- `ch09-tools-chapter-flow.png`
- `ch09-tools-action-layer-map.png`
- `ch09-tool-control-loop.png`
- `ch09-memory-chapter-flow.png`
- `ch09-memory-write-retrieve-loop.png`
- `ch09-mcp-chapter-flow.png`
- `ch09-mcp-capability-bridge.png`
- `ch09-frameworks-position-map.png`
- `ch09-framework-selection-map.png`
- `ch09-multi-agent-chapter-flow.png`
- `ch09-multi-agent-coordination-map.png`
- `ch09-eval-safety-chapter-flow.png`
- `ch09-agent-risk-debug-loop.png`
- `ch09-deployment-chapter-flow.png`
- `ch09-production-runtime-map.png`
- `ch09-deployment-observability-loop.png`
- `ch09-projects-route-map.png`
- `ch09-project-learning-order-map.png`
- `ch09-project-delivery-loop.png`
- `ch09-agent-boundary-workflow-chatbot-map.png`
- `ch09-agent-action-loop-trace-map.png`
- `ch09-agent-system-architecture-dataflow-map.png`
- `ch09-cot-self-check-structure-map.png`
- `ch09-plan-execute-monitor-replan-map.png`
- `ch09-advanced-planning-dag-critical-path-map.png`
- `ch09-reasoning-eval-failure-taxonomy-map.png`
- `ch09-tool-schema-validation-guardrail-map.png`
- `ch09-tool-description-quality-map.png`
- `ch09-tool-safety-permission-sandbox-map.png`
- `ch09-code-agent-sandbox-review-map.png`
- `ch09-memory-layer-selection-map.png`
- `ch09-long-term-memory-write-update-policy-map.png`
- `ch09-memory-engineering-lifecycle-map.png`
- `ch09-mcp-host-client-server-message-flow-map.png`
- `ch09-mcp-server-tool-contract-map.png`
- `ch09-langgraph-state-machine-map.png`
- `ch09-framework-selection-decision-map.png`
- `ch09-multi-agent-pattern-selection-map.png`
- `ch09-multi-agent-communication-contract-map.png`
- `ch09-multi-agent-coordination-cost-map.png`
- `ch09-agent-eval-layered-scorecard-map.png`
- `ch09-agent-security-prompt-injection-risk-map.png`
- `ch09-agent-observability-trace-span-map.png`
- `ch09-agent-runtime-state-queue-map.png`
- `ch09-agent-persistence-checkpoint-eventlog-map.png`
- `ch09-agent-cost-routing-cache-budget-map.png`
- `ch09-production-readiness-canary-rollback-map.png`
- `ch09-research-assistant-citation-trace-map.png`
- `ch09-data-analysis-agent-notebook-loop-map.png`
- `ch09-multi-agent-dev-team-delivery-map.png`
- `ch12-multimodal-aigc.png`
- `ch12-learning-quest-map.png`
- `ch12-multimodal-system-backbone.png`
- `ch12-study-guide-modal-workflow-map.png`
- `ch12-multimodal-chapter-flow.png`
- `ch12-image-gen-chapter-flow.png`
- `ch12-video-gen-chapter-flow.png`
- `ch12-frontier-ethics-route-map.png`
- `ch12-multimodal-rag-agent-bridge.png`
- `ch12-projects-delivery-loop.png`
- `ch12-multimodal-app-engineering-loop.png`
- `ch12-sd-application-mode-selector-map.png`
- `ch12-sd-finetuning-route-choice-map.png`
- `ch12-image-generation-trend-radar-map.png`
- `ch12-tts-text-to-speech-pipeline-map.png`
- `ch12-digital-human-sync-pipeline-map.png`
- `ch12-aigc-frontier-system-trend-map.png`
- `ch12-ai-ethics-safety-guardrail-map.png`
- `ch12-ai-regulation-engineering-translation-map.png`
- `git-four-areas.png`
- `pandas-dataframe-structure.png`
- `chart-selection-decision-tree.png`
- `gradient-descent-path.png`
- `ml-modeling-loop.png`
- `confusion-matrix-error-cost.png`
- `pytorch-training-loop.png`
- `cnn-convolution-kernel.png`
- `self-attention-qkv.png`
- `object-detection-output.png`
- `semantic-segmentation-mask.png`
- `bio-ner-recovery.png`
- `bert-gpt-t5-comparison.png`
- `prompt-before-after.png`
- `lora-parameter-update.png`
- `rag-document-answer-loop.png`
- `courseware-assistant-workflow.png`
- `agent-tool-trace.png`
- `agent-guardrails-layers.png`
- `diffusion-noise-denoise.png`
- `matplotlib-figure-axes.png`
- `seaborn-statistical-plots.png`
- `sql-table-join-map.png`
- `eda-analysis-workflow.png`
- `vector-dot-cosine-geometry.png`
- `matrix-linear-transform-grid.png`
- `probability-distribution-map.png`
- `information-entropy-uncertainty.png`
- `sklearn-estimator-pipeline.png`
- `linear-regression-loss-landscape.png`
- `logistic-regression-boundary.png`
- `decision-tree-split-path.png`
- `ensemble-learning-voting-forest.png`
- `clustering-kmeans-centroids.png`
- `pca-dimensionality-reduction.png`
- `neural-network-forward-backward.png`
- `imagenet-cnn-evolution.png`
- `lstm-gate-memory-flow.png`
- `transformer-block-architecture.png`
- `word2vec-embedding-neighborhood.png`
- `bert-masked-language-model.png`
- `gpt-autoregressive-generation.png`
- `rlhf-three-stage-loop.png`
- `rag-evaluation-triangle.png`
- `agent-memory-system.png`
- `ch04-linear-algebra-roadmap.png`
- `eigenvalue-special-directions.png`
- `vector-space-basis-span.png`
- `ch04-probability-roadmap.png`
- `distribution-family-comparison.png`
- `mle-likelihood-curve.png`
- `ch04-calculus-roadmap.png`
- `derivative-tangent-slope.png`
- `gradient-contour-field.png`
- `chain-rule-backprop-graph.png`
- `math-study-loop.png`
- `math-task-checklist.png`
- `ml-basics-roadmap.png`
- `math-to-ml-bridge.png`
- `supervised-learning-roadmap.png`
- `unsupervised-learning-roadmap.png`
- `anomaly-detection-outliers.png`
- `ml-evaluation-roadmap.png`
- `cross-validation-kfold.png`
- `bias-variance-tradeoff.png`
- `hyperparameter-tuning-search.png`
- `feature-engineering-roadmap.png`
- `feature-type-target-map.png`
- `feature-preprocessing-pipeline.png`
- `feature-construction-workshop.png`
- `feature-selection-methods.png`
- `column-transformer-pipeline.png`
- `ml-projects-roadmap.png`
- `house-price-project-flow.png`
- `customer-churn-project-flow.png`
- `user-segmentation-rfm.png`
- `kaggle-submission-loop.png`
- `ml-study-loop.png`
- `ml-task-checklist.png`
- `mlp-neuron-activation.png`
- `optimizer-comparison.png`
- `regularization-overfitting-controls.png`
- `pytorch-autograd-graph.png`
- `dataset-dataloader-batch-flow.png`
- `cnn-feature-map-pipeline.png`
- `rnn-unrolled-hidden-state.png`
- `gan-adversarial-loop.png`
- `vae-latent-space-flow.png`
- `training-curve-diagnosis.png`
- `tokenizer-subword-flow.png`
- `embedding-semantic-space.png`
- `llm-history-timeline.png`
- `pretraining-data-pipeline.png`
- `finetuning-alignment-pipeline.png`
- `document-processing-vectorization.png`
- `vector-database-similarity-search.png`
- `hybrid-search-rerank-flow.png`
- `function-calling-workflow.png`
- `template-doc-generation-pipeline.png`
- `agent-vs-chatbot-comparison.png`
- `agent-system-architecture.png`
- `react-reason-act-observe-loop.png`
- `mcp-host-client-server.png`
- `multi-agent-message-flow.png`
- `cv-pixel-rgb-grid.png`
- `cv-image-processing-pipeline.png`
- `cv-data-augmentation-gallery.png`
- `yolo-grid-detection-flow.png`
- `ocr-layout-recognition-pipeline.png`
- `text-preprocessing-pipeline.png`
- `bow-tfidf-representation.png`
- `contextual-embedding-comparison.png`
- `text-classification-pipeline.png`
- `seq2seq-attention-alignment.png`
- `multimodal-alignment-fusion.png`
- `vision-language-model-architecture.png`
- `stable-diffusion-components.png`
- `video-audio-generation-pipeline.png`
- `creative-platform-workflow.png`
- `elective-cpp-runtime-memory.png`
- `elective-model-optimization-map.png`
- `elective-inference-engine-hardware.png`
- `elective-model-serving-architecture.png`
- `elective-python-decorator-flow.png`
- `elective-asyncio-concurrency-control.png`
- `elective-svm-margin-support-vectors.png`
- `elective-knn-neighbor-voting.png`
- `elective-naive-bayes-evidence.png`
- `elective-ai-security-red-team-loop.png`
- `elective-ai-frontend-stack.png`
- `elective-ai-product-decision-matrix.png`
- `elective-cpp-deployment-module-map.png`
- `elective-cpp-raii-ownership-map.png`
- `elective-edge-deployment-constraint-map.png`
- `elective-deployment-project-delivery-loop.png`
- `elective-python-advanced-module-map.png`
- `elective-generator-stream-pipeline.png`
- `elective-metaprogramming-registry-map.png`
- `elective-classic-ml-module-map.png`
- `elective-lda-projection-map.png`
- `elective-optimization-tradeoff-dashboard.png`
- `elective-inference-engine-selection-matrix.png`
- `elective-serving-metrics-version-routing-map.png`
- `elective-decorator-crosscutting-layers.png`
- `elective-asyncio-timeout-cancel-rate-limit-map.png`
- `elective-svm-c-kernel-decision-map.png`
- `elective-ai-security-threat-regression-map.png`
- `elective-ai-frontend-state-machine-map.png`
- `elective-ai-product-experiment-metrics-loop.png`
- `appendix-ai-milestones-timeline.png`
- `appendix-troubleshooting-rescue-map.png`
- `appendix-hardware-cloud-decision-tree.png`
- `appendix-job-prep-funnel.png`
- `appendix-continuous-learning-flywheel.png`
- `appendix-resource-selection-funnel.png`
- `appendix-faq-decision-tree.png`
- `appendix-project-quick-reference-map.png`
- `appendix-course-numbering-map.png`
- `appendix-visual-enhancement-kanban.png`
- `appendix-ai-main-relay-map.png`
- `appendix-ai-project-lens-map.png`
- `appendix-classic-ml-branch-map.png`
- `appendix-nlp-llm-lineage-map.png`
- `appendix-agent-system-lineage-map.png`
- `appendix-multimodal-aigc-lineage-map.png`
- `ch01-terminal-path-command-map.png`
- `ch01-task-list-workflow.png`
- `ch01-cli-automation-workflow.png`
- `ch01-package-manager-flow.png`
- `ch01-git-daily-loop.png`
- `ch01-git-remote-sync.png`
- `ch01-vscode-workspace-flow.png`
- `ch01-python-env-stack.png`
- `ch01-git-branch-collaboration.png`
- `ch01-jupyter-kernel-state.png`
- `ch02-learning-quest-map.png`
- `ch02-python-ai-backbone.png`
- `ch02-study-guide-program-loop.png`
- `ch02-variable-object-reference.png`
- `ch02-control-flow-paths.png`
- `ch02-data-structures-comparison.png`
- `ch02-function-call-scope.png`
- `ch02-oop-class-object-map.png`
- `ch02-exception-flow.png`
- `ch02-ai-api-request-response.png`
- `ch02-task-list-workflow.png`
- `ch02-python-ai-workflow.png`
- `ch02-operators-decision-flow.png`
- `ch02-input-output-flow.png`
- `ch02-modules-package-structure.png`
- `ch02-file-io-serialization-flow.png`
- `ch02-functional-pipeline.png`
- `ch02-generator-streaming-data.png`
- `ch02-type-hints-quality-flow.png`
- `ch02-todo-cli-architecture.png`
- `ch02-web-scraper-pipeline.png`
- `ch02-web-api-request-response.png`
- `ch03-numpy-array-shape-axis.png`
- `ch03-numpy-broadcasting-vectorization.png`
- `ch03-pandas-groupby-split-apply-combine.png`
- `ch03-multi-source-analysis-architecture.png`
- `ch03-learning-quest-map.png`
- `ch03-data-analysis-backbone.png`
- `ch03-study-guide-data-loop.png`
- `ch03-task-list-workflow.png`
- `ch03-pure-python-data-flow.png`
- `ch03-numpy-overview-array-engine.png`
- `ch03-numpy-indexing-slicing-map.png`
- `ch03-numpy-reshape-axis-flow.png`
- `ch03-numpy-linear-algebra-toolkit.png`
- `ch03-numpy-random-statistics-map.png`
- `ch03-pandas-roadmap.png`
- `ch03-pandas-read-write-first-look.png`
- `ch03-pandas-selection-filter-map.png`
- `ch03-pandas-cleaning-workflow.png`
- `ch03-pandas-transform-pipeline.png`
- `ch03-pandas-merge-concat-join.png`
- `ch03-pandas-time-series-analysis.png`
- `ch03-visualization-roadmap.png`
- `ch03-plotly-interactive-dashboard.png`
- `ch03-database-roadmap.png`
- `ch03-relational-database-foundation.png`
- `ch03-python-database-bridge.png`
- `ch03-database-design-erd-normalization.png`
- `intro-ai-fullstack-capability-map.png`
- `intro-modern-ai-stack-map.png`
- `intro-learning-path-selection.png`
- `intro-four-main-routes-subway.png`
- `intro-blocker-diagnosis-flow.png`
- `intro-project-portfolio-roadmap.png`
- `intro-role-based-paths-map.png`
- `intro-graduation-project-loop.png`

## Remaining

- None.

## Resume Command

```bash
# No remaining images. To regenerate everything intentionally:
npm run images:generate -- --overwrite
```
