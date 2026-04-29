---
title: "课程视觉增强规划"
description: "按展示章节梳理哪些页面适合继续增加图片、示意图、流程图和代码可视化，帮助新人更快建立直觉。"
keywords: [课程配图, 教学设计, 可视化学习, AI课程图片, Mermaid, 图像生成]
---

# 课程视觉增强规划

![课程图片资产规划看板](/img/course/appendix-visual-enhancement-kanban.png)

现在课程已经有阶段首页主视觉，也有大量 Mermaid 流程图。下一步不是给每页都塞图片，而是把图片放在最能降低理解成本的位置。

:::info 章节编号约定
课程源码目录已经和网页展示章节号对齐：`ch01-tools` 对应第 1 章，`ch05-machine-learning` 对应第 5 章。侧边栏里的主线 1～4 是学习分组，不作为目录层级。
:::

一个简单原则是：

| 内容类型 | 更适合的视觉形式 | 说明 |
|---|---|---|
| 抽象概念 | 类比插图、结构示意图 | 帮新人先建立画面感 |
| 多步骤流程 | Mermaid 或流程图 | 帮新人知道先后顺序 |
| 数学与数据 | 代码生成图表 | 比 AI 生成图更准确 |
| 模型结构 | 模块结构图 | 帮新人理解输入、输出和中间层 |
| 项目实战 | 架构图、界面草图、结果样例图 | 帮新人知道最终要做成什么 |
| 历史背景 | 时间线图、人物与论文卡片 | 帮新人把算法放回时代里 |

## 优先级规则

| 优先级 | 适合先做的图 | 原因 |
|---|---|---|
| P0 | 每个章节首页主视觉 | 已完成，负责建立章节氛围 |
| P1 | 每章第一篇主干课的概念图 | 最影响新人第一印象 |
| P1 | 项目页的系统架构图和结果样例图 | 最能帮助上手 |
| P2 | 数学、训练、评估类代码可视化 | 能把抽象过程变成可观察结果 |
| P2 | 历史时间线和论文故事图 | 提升兴趣和记忆点 |
| P3 | 装饰型插图 | 只有在页面很干、读感疲劳时再加 |

## 第 1 章（目录 ch01-tools）：开发者工具基础

| 章节 | 建议加图 | 优先级 |
|---|---|---|
| 终端命令行 | 终端、目录树、命令执行结果之间的关系图 | P1 |
| Git 基础 | 工作区、暂存区、本地仓库、远程仓库四格图 | P1 |
| 开发环境 | Python 环境、VS Code、Jupyter、依赖文件的关系图 | P2 |

适合生成的图片：开发工作台、Git 存档系统、环境隔离实验室。

更适合代码或 Mermaid 的图：Git 状态流、分支合并过程、虚拟环境路径关系。

## 第 2 章（目录 ch02-python）：Python 编程基础

| 章节 | 建议加图 | 优先级 |
|---|---|---|
| Python 基础语法 | 变量、分支、循环、函数如何组成程序的积木图 | P1 |
| Python 进阶 | 对象、异常、文件、生成器的运行时关系图 | P2 |
| 项目实战 | CLI、Web API、AI API 的输入输出界面草图 | P1 |

适合生成的图片：Python 小工具工坊、API 请求和响应工作台。

更适合代码生成的图：函数调用栈、JSON 文件读写前后对比。

## 第 3 章（目录 ch03-data-analysis）：数据分析与可视化

| 章节 | 建议加图 | 优先级 |
|---|---|---|
| 纯 Python 数据热身 | 原始列表/字典如何变成表格的示意图 | P2 |
| NumPy | 数组 shape、切片、广播和矩阵乘法示意图 | P1 |
| Pandas | DataFrame、Index、列选择、groupby、merge 的表格动画感示意图 | P1 |
| 数据可视化 | 图表选择决策树和“同一数据不同图表”的对比图 | P1 |
| 数据库选修 | 表、主键、外键、SQL 查询路径图 | P2 |
| 项目实战 | EDA 报告样例图、多数据源合并流程图 | P1 |

适合生成的图片：数据侦探工作台、分析报告封面、数据库档案系统。

更适合代码生成的图：直方图、箱线图、散点图、折线图、热力图。

## 第 4 章（目录 ch04-ai-math）：AI 数学基础

| 章节 | 建议加图 | 优先级 |
|---|---|---|
| 线性代数 | 向量箭头、矩阵变换、特征向量方向、SVD 拆解图 | P1 |
| 概率与统计 | 概率树、分布曲线、采样误差、MLE/EM 侦探类比图 | P1 |
| 微积分与优化 | 函数曲线、切线、梯度箭头、下山路径、链式法则传递图 | P1 |

适合生成的图片：数学眼镜、梯度下山地图、概率侦探。

更适合代码生成的图：二维向量、正态分布、交叉熵曲线、梯度下降轨迹。

## 第 5 章（目录 ch05-machine-learning）：机器学习

| 章节 | 建议加图 | 优先级 |
|---|---|---|
| 机器学习基础 | 任务定义、数据划分、baseline、评估、复盘闭环图 | P1 |
| 监督学习 | 线性回归拟合线、逻辑回归决策边界、树模型分裂图 | P1 |
| 无监督学习 | 聚类结果、PCA 投影、异常点检测示意图 | P1 |
| 模型评估 | 混淆矩阵、偏差方差曲线、交叉验证折叠图 | P1 |
| 特征工程 | 原始字段到特征表的加工流水线图 | P2 |
| 项目实战 | 房价、流失、分群项目的报告版式和错误分析看板 | P1 |

适合生成的图片：建模侦探报告、模型训练流水线。

更适合代码生成的图：决策边界、残差图、ROC/PR 曲线、PCA 可视化。

## 第 6 章（目录 ch06-deep-learning）：深度学习与 Transformer

| 章节 | 建议加图 | 优先级 |
|---|---|---|
| 神经网络基础 | 神经元、MLP、激活函数、反向传播责任传递图 | P1 |
| PyTorch | Tensor、Dataset、DataLoader、Module、Training Loop 关系图 | P1 |
| CNN | 卷积核滑动、特征图、池化、经典架构演进图 | P1 |
| RNN | 时间展开、隐藏状态传递、LSTM 门控图 | P1 |
| Transformer | QKV、Self-Attention、Encoder/Decoder block 结构图 | P1 |
| 生成模型 | GAN 对抗关系、VAE 潜空间、生成样本演化图 | P2 |
| 训练技巧 | 训练曲线诊断图、压缩决策树、调参记录看板 | P2 |
| 项目实战 | 图像分类、情感分析、生成项目的结果展示模板 | P1 |

适合生成的图片：模型发动机舱、神经网络训练实验室。

更适合代码生成的图：loss 曲线、激活函数曲线、注意力热力图、潜空间散点图。

## 第 10 章（目录 ch10-computer-vision）：计算机视觉

| 章节 | 建议加图 | 优先级 |
|---|---|---|
| CV 基础 | 像素网格、RGB 通道、滤波、边缘检测前后对比 | P1 |
| 图像分类 | 数据增强前后对比、架构演进、错误样本墙 | P1 |
| 目标检测 | bounding box、IoU、NMS、YOLO 网格示意图 | P1 |
| 图像分割 | 语义分割 mask、实例分割 mask、边界错误示意图 | P1 |
| 高级视觉 | 人脸识别流水线、OCR 版面解析、视频帧分析、3D 点云 | P1 |
| 视觉项目 | 安防告警闭环、医学影像复核看板 | P1 |

适合生成的图片：视觉任务地图、图像理解工作台。

更适合真实/代码图：预处理前后图、检测框、分割 mask、OCR 框选结果。

## 第 11 章（目录 ch11-nlp）：自然语言处理

| 章节 | 建议加图 | 优先级 |
|---|---|---|
| 文本基础 | 文本清洗、分词、BoW/TF-IDF 表示对比图 | P1 |
| 表示学习 | one-hot 到 Word2Vec 再到上下文化表示的演进图 | P1 |
| 文本分类 | 传统分类和深度分类输入路径对比图 | P2 |
| 序列标注 | BIO 标签到实体恢复的彩色标注图 | P1 |
| Seq2Seq | Encoder-Decoder、Attention 对齐矩阵、翻译流程图 | P1 |
| 预训练模型 | BERT/GPT/T5 任务组织差异图 | P1 |
| NLP 项目 | 问答、摘要、信息抽取的输入输出样例图 | P1 |

适合生成的图片：文本理解助手、语言模型演进路线。

更适合代码生成的图：TF-IDF 表格、词向量近邻、注意力对齐热力图。

## 第 7 章（目录 ch07-llm-principles）：大模型原理、Prompt 与微调

| 章节 | 建议加图 | 优先级 |
|---|---|---|
| NLP 速成 | Tokenizer、Embedding、HuggingFace pipeline 运行图 | P2 |
| LLM 概览 | GPT/BERT/Transformer 发展时间线图 | P1 |
| Transformer 深入 | Attention 复杂度、模型变体、规模化计算结构图 | P1 |
| 预训练 | 数据清洗、训练目标、训练集群流水线图 | P1 |
| Prompt | 坏 Prompt 到好 Prompt 的对比卡片 | P1 |
| 微调 | Full fine-tune、LoRA、QLoRA 参数更新对比图 | P1 |
| 对齐 | SFT、Reward Model、RLHF 流程图 | P1 |
| 项目 | 领域微调方案文档和评估看板 | P2 |

适合生成的图片：大模型能力工厂、Prompt 实验台。

更适合 Mermaid/代码图：结构化输出 schema、LoRA 插件结构、RLHF 流程。

## 第 8 章（目录 ch08-rag）：LLM 应用开发与 RAG

| 章节 | 建议加图 | 优先级 |
|---|---|---|
| RAG | 文档解析、切块、向量库、检索、重排、引用闭环图 | P1 |
| 部署 | 本地模型、推理服务、统一 API 的系统拓扑图 | P1 |
| 应用开发 | Function Calling、对话状态、文档解析、模板导出的工作流图 | P1 |
| 工程化 | 异步并发、API、日志监控、Docker 部署架构图 | P2 |
| 综合项目 | 企业知识库、智能助手、课件生成助手的产品界面草图 | P1 |

适合生成的图片：知识库引擎室、课件生成助手工作台。

更适合 Mermaid/结构图：RAG trace、工具调用链、API 请求响应、日志字段流。

## 第 9 章（目录 ch09-agent）：AI Agent

| 章节 | 建议加图 | 优先级 |
|---|---|---|
| Agent 基础 | Agent 与聊天机器人、工作流、函数调用的边界对比图 | P1 |
| 推理与规划 | ReAct、Plan-and-Execute、反思循环图 | P1 |
| 工具 | 工具 schema、工具选择、工具安全边界图 | P1 |
| 记忆 | 短期记忆、长期记忆、情景记忆、程序记忆分层图 | P1 |
| MCP | Host、Client、Server、Resource、Tool 的协议结构图 | P1 |
| 框架 | LangGraph、LlamaIndex、AutoGen 等选型矩阵图 | P2 |
| 多 Agent | 角色协作、消息传递、任务分配图 | P1 |
| 评估安全 | trace 回放、护栏分层、权限确认图 | P1 |
| 部署运维 | Agent 运行时、队列、日志、恢复、成本监控架构图 | P1 |
| 项目 | 研究助手、数据分析 Agent、多 Agent 团队的执行轨迹图 | P1 |

适合生成的图片：Agent 指挥台、多工具协作控制室。

更适合 Mermaid/结构图：执行 trace、状态机、工具白名单、人工确认流程。

## 第 12 章（目录 ch12-multimodal）：AIGC 与多模态

| 章节 | 建议加图 | 优先级 |
|---|---|---|
| 多模态基础 | 文本、图像、语音、视频对齐与融合图 | P1 |
| 图像生成 | 扩散模型加噪去噪、Stable Diffusion 组件图、LoRA 微调前后对比 | P1 |
| 视频与语音 | 文案、分镜、TTS、视频生成、数字人流水线图 | P1 |
| 前沿与合规 | 风险分级、版权审核、人工复核流程图 | P2 |
| 综合项目 | AI 创意平台界面草图、资产版本流、导出包结构图 | P1 |

适合生成的图片：AI 创意工作台、多模态内容工厂。

更适合真实/流程图：生成工作流、资产版本树、审核清单、输出 bundle。

## 附录（目录 appendix）：查阅、排障与学习支持

附录不适合做太多装饰图，更适合做“快速定位问题”的查阅型图片。新人打开附录时，通常不是为了系统学习一章，而是为了快速判断：我现在卡在哪里、该查哪页、该补什么资源、该不该买硬件、项目和求职该怎么准备。

| 页面 | 建议加图 | 优先级 |
|---|---|---|
| 推荐学习资源 | 资源选择漏斗图：主线课程、卡点补充、项目验证、复盘沉淀 | P2 |
| 课程编号约定 | 展示章节号与源码目录对应地图 | P3 |
| AI 重要论文与算法时间线 | AI 历史接力赛长图或“论文节点卡片墙” | P1 |
| 课程视觉增强规划 | 课程图片资产规划看板：P0/P1/P2/P3 如何分批推进 | P3 |
| 硬件与云资源指南 | 硬件购买与云 GPU 决策树 | P1 |
| 常见问题 | 新人学习决策树：数学、GPU、项目、论文、求职等问题如何分流 | P2 |
| 持续学习方法论 | 基础学习、项目学习、前沿跟踪三层飞轮图 | P2 |
| 学习卡点救援 | 卡点排障流程图：环境、代码、训练、显存、项目、焦虑 | P1 |
| 学习资源速查 | AI 项目速查总览图：环境、baseline、评估、RAG、Agent、Prompt | P2 |
| 求职准备清单 | 求职准备漏斗图：岗位定位、项目打磨、简历表达、面试复盘 | P1 |

适合生成的图片：排障救援地图、硬件决策树、学习资源导航台、求职准备作战板、AI 历史时间线海报。

更适合 Mermaid/结构图：课程编号对应关系、FAQ 分流树、持续学习飞轮、图片资产生成批次规划。

## 最值得先生成的 20 张图

如果只先做一批，我建议按下面顺序生成或制作：

| 顺序 | 图片 | 建议插入位置 | 类型 |
|---|---|---|---|
| 1 | Git 工作区、暂存区、本地库、远程库四格图 | `ch01-tools/ch02-git/01-git-basics.md` | 流程示意 |
| 2 | Pandas DataFrame 结构图 | `ch03-data-analysis/ch03-pandas/01-core-structures.md` | 概念图 |
| 3 | 图表选择决策树 | `ch03-data-analysis/ch04-visualization/04-best-practices.md` | 决策图 |
| 4 | 梯度下降下山路径图 | `ch04-ai-math/ch03-calculus/03-gradient-descent.md` | 代码可视化 |
| 5 | 机器学习建模闭环图 | `ch05-machine-learning/ch01-ml-basics/01-what-is-ml.md` | 流程图 |
| 6 | 混淆矩阵和错误代价图 | `ch05-machine-learning/ch04-evaluation/01-metrics.md` | 教学图 |
| 7 | PyTorch 训练循环图 | `ch06-deep-learning/ch02-pytorch/05-training-loop.md` | 流程图 |
| 8 | CNN 卷积核滑动示意图 | `ch06-deep-learning/ch03-cnn/01-convolution-basics.md` | 概念图 |
| 9 | Self-Attention QKV 图 | `ch06-deep-learning/ch05-transformer/01-attention-mechanism.md` | 结构图 |
| 10 | 目标检测输出拆解图 | `ch10-computer-vision/ch03-detection/01-detection-overview.md` | 任务图 |
| 11 | 语义分割 mask 对比图 | `ch10-computer-vision/ch04-segmentation/01-semantic-segmentation.md` | 结果图 |
| 12 | BIO 标签到实体恢复图 | `ch11-nlp/ch04-sequence-labeling/01-ner-overview.md` | 标注图 |
| 13 | BERT/GPT/T5 对比图 | `ch11-nlp/ch06-pretrained/00-roadmap.md` | 对比图 |
| 14 | Prompt 改写前后对比卡 | `ch07-llm-principles/ch05-prompt/01-prompt-basics.md` | 对比图 |
| 15 | LoRA 参数更新对比图 | `ch07-llm-principles/ch06-finetuning/02-lora-qlora.md` | 结构图 |
| 16 | RAG 文档到答案闭环图 | `ch08-rag/ch01-rag/01-rag-basics.md` | 系统图 |
| 17 | 课件生成助手工作流图 | `ch08-rag/ch05-projects/04-courseware-assistant.md` | 项目架构图 |
| 18 | Agent 工具调用 trace 图 | `ch09-agent/ch03-tools/08-multi-tool-practice.md` | 执行轨迹图 |
| 19 | Agent 护栏分层图 | `ch09-agent/ch08-eval-safety/04-guardrails.md` | 安全图 |
| 20 | 扩散模型加噪去噪图 | `ch12-multimodal/ch02-image-gen/01-diffusion-models.md` | 模型过程图 |

## 第二批建议生成的 25 张图

第一批已经覆盖阶段入口和最核心概念。第二批继续补“新人容易卡住、但又是后续课程地基”的页面，优先补概念结构图、算法直觉图和项目流程图。

| 顺序 | 图片 | 建议插入位置 | 类型 |
|---|---|---|---|
| 1 | Matplotlib Figure 与 Axes 结构图 | `ch03-data-analysis/ch04-visualization/01-matplotlib.md` | 对象模型图 |
| 2 | Seaborn 统计图选择图 | `ch03-data-analysis/ch04-visualization/02-seaborn.md` | 图谱图 |
| 3 | SQL 表连接关系图 | `ch03-data-analysis/ch05-database/02-sql-basics.md` | 数据关系图 |
| 4 | EDA 探索性数据分析流程图 | `ch03-data-analysis/ch06-projects/01-eda-project.md` | 项目流程图 |
| 5 | 向量点积与余弦相似度几何图 | `ch04-ai-math/ch01-linear-algebra/01-vectors.md` | 几何直觉图 |
| 6 | 矩阵线性变换网格图 | `ch04-ai-math/ch01-linear-algebra/02-matrices.md` | 数学可视化 |
| 7 | 概率分布与贝叶斯更新图 | `ch04-ai-math/ch02-probability/01-probability-basics.md` | 概念图 |
| 8 | 信息熵与不确定性图 | `ch04-ai-math/ch02-probability/04-information-theory.md` | 概念图 |
| 9 | Scikit-learn Estimator 与 Pipeline 图 | `ch05-machine-learning/ch01-ml-basics/02-sklearn-intro.md` | 工程流程图 |
| 10 | 线性回归拟合与损失曲面图 | `ch05-machine-learning/ch02-supervised/01-linear-regression.md` | 算法直觉图 |
| 11 | 逻辑回归决策边界图 | `ch05-machine-learning/ch02-supervised/02-logistic-regression.md` | 分类边界图 |
| 12 | 决策树分裂路径图 | `ch05-machine-learning/ch02-supervised/03-decision-trees.md` | 算法结构图 |
| 13 | 集成学习投票与森林图 | `ch05-machine-learning/ch02-supervised/04-ensemble-learning.md` | 模型集成图 |
| 14 | K-Means 聚类中心迭代图 | `ch05-machine-learning/ch03-unsupervised/01-clustering.md` | 算法过程图 |
| 15 | PCA 降维投影图 | `ch05-machine-learning/ch03-unsupervised/02-dimensionality-reduction.md` | 空间投影图 |
| 16 | 神经网络前向与反向传播图 | `ch06-deep-learning/ch01-nn-basics/02-forward-backward.md` | 训练机制图 |
| 17 | 经典 CNN 架构演进图 | `ch06-deep-learning/ch03-cnn/03-classic-architectures.md` | 历史演进图 |
| 18 | LSTM 门控记忆流图 | `ch06-deep-learning/ch04-rnn/02-lstm-gru.md` | 结构机制图 |
| 19 | Transformer Block 架构图 | `ch06-deep-learning/ch05-transformer/02-transformer-architecture.md` | 模块结构图 |
| 20 | 词向量语义邻域图 | `ch11-nlp/ch02-embeddings/01-word-embedding.md` | 语义空间图 |
| 21 | BERT Masked Language Model 图 | `ch11-nlp/ch06-pretrained/02-bert.md` | 预训练目标图 |
| 22 | GPT 自回归生成图 | `ch11-nlp/ch06-pretrained/03-gpt-series.md` | 生成机制图 |
| 23 | RLHF 三阶段流程图 | `ch07-llm-principles/ch07-alignment/02-rlhf.md` | 对齐流程图 |
| 24 | RAG 评估三角图 | `ch08-rag/ch01-rag/07-rag-evaluation.md` | 评估框架图 |
| 25 | Agent 记忆系统分层图 | `ch09-agent/ch04-memory/01-memory-overview.md` | 系统结构图 |

## 附录建议生成的 10 张图

附录这批图建议优先做“查阅型”和“决策型”，帮助学习者在卡住时快速回到正确页面。

| 顺序 | 图片 | 建议插入位置 | 类型 |
|---|---|---|---|
| 1 | AI 历史接力赛时间线图 | `appendix/ai-milestones.md` | 历史主线图 |
| 2 | 学习卡点排障地图 | `appendix/troubleshooting.md` | 排障流程图 |
| 3 | 硬件与云资源决策树 | `appendix/hardware.md` | 决策图 |
| 4 | 求职准备漏斗图 | `appendix/job-prep.md` | 规划图 |
| 5 | 持续学习三层飞轮图 | `appendix/continuous-learning.md` | 方法论图 |
| 6 | 资源选择漏斗图 | `appendix/resources.md` | 学习资源导航图 |
| 7 | FAQ 新人问题分流树 | `appendix/faq.md` | 问题分流图 |
| 8 | AI 项目速查总览图 | `appendix/resource-quick-ref.md` | 速查地图 |
| 9 | 展示章节号与源码目录对应地图 | `appendix/course-numbering.md` | 维护说明图 |
| 10 | 课程图片资产规划看板 | `appendix/visual-enhancement-plan.md` | 资产规划图 |

## 生成策略

第一批先做 P1 教学图，不做装饰图。课程图要服务理解，不要只是让页面看起来热闹。

第二批补项目页截图和界面草图。项目页最需要让新人知道“最终作品长什么样”，尤其是 RAG、Agent、课件生成助手和多模态创意平台。

第三批再补历史人物、论文和算法故事图。这类图适合放在历史页和章节开头，帮助记忆技术为什么出现。

附录批次建议先补 `ai-milestones`、`troubleshooting`、`hardware`、`job-prep` 这 4 张 P1 图，再补资源、FAQ、持续学习和速查类图。这样既能提升查阅效率，又不会把附录做成图片堆。

## 使用当前图片脚本的建议

当前脚本 `scripts/generate_course_images.py` 已经能管理课程图片资产。后续可以继续把上面的高优先级图片追加到 `IMAGE_JOBS`，并用下面方式生成：

```bash
pip install -r requirements-course-ai.txt
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_BASE_URL="https://cliproxy.airoads.org/v1"
export OPENAI_IMAGE_MODEL="gpt-image-2"
npm run images:dry-run
npm run images:generate
```

对于数学曲线、训练曲线、检测框和分割 mask，优先用代码生成，因为它们需要准确。对于阶段主视觉、项目工作台和历史故事图，可以用图像生成模型制作。
