# AI Roads

<div align="center">
  <img src="./public/img/logo.svg" width="96" alt="AI Roads logo">
  <h3>面向实践者的免费多语言 AI 工程课程。</h3>
  <p>
    <a href="./README.md">English</a> |
    简体中文 |
    <a href="./README_ja.md">日本語</a>
  </p>
  <p>
    <a href="https://airoads.org">官方网站</a>
  </p>
</div>

---

AI Roads 是一个免费、适合初学者的 AI 工程课程。它从开发者基础开始，逐步进入 Python、数据分析、AI 数学、机器学习、深度学习、LLM 原理、RAG、AI Agent、多模态 AIGC，以及开源大模型部署与微调。

本仓库驱动公开学习网站 [airoads.org](https://airoads.org)。网站默认语言是英文，同时提供完整的简体中文和日文内容路线。

## 项目目标

AI Roads 不是资料链接合集，而是一条能持续产出项目的学习路线。它强调：

1. 从工具基础到 AI 项目的清晰顺序。
2. 每节课尽量提供可运行命令、代码、输出和证据。
3. 用图解、漫画、结果图和本地化图片降低理解难度。
4. 用项目检查点把知识变成作品集。
5. 同步维护英文、简体中文和日文内容。
6. 用验证、构建、SEO 和部署脚本支撑一个生产级静态课程站。

学习建议很简单：不要只读。每个阶段都要留下能运行、能解释、能展示的东西。

## 适合谁

- 想从零进入 AI 工程的初学者。
- 已经会一点编程，希望系统学习 AI 的开发者。
- 想做项目，而不是只看概念的学习者。
- 想围绕 LLM、RAG、Agent、CV、NLP 或多模态方向做作品集的学生。
- 想维护多语言静态课程站的内容维护者。

## 课程路线

```text
0   起步指南
1   开发者工具
2   Python 编程
3   数据分析与可视化
4   AI 最小数学基础
5   机器学习
6   深度学习与 Transformer 基础
7   LLM 原理、Prompt 与微调
8   LLM 应用开发与 RAG
9   AI Agent 与智能体系统
10  计算机视觉
11  LLM 之后的 NLP 专题
12  AIGC 与多模态
13  开源大模型部署与微调
E   选修模块
A   附录
```

推荐先完成 1-9 章，再根据项目方向选择 10-13 章。

## 主要内容

- **1 开发者工具**：终端、Git、本地开发环境。
- **2 Python 编程**：语法、数据结构、文件、OOP、API、项目。
- **3 数据分析**：NumPy、Pandas、可视化、数据库。
- **4 AI 数学**：线性代数、概率、微积分、优化。
- **5 机器学习**：监督学习、无监督学习、评估、特征工程。
- **6 深度学习**：PyTorch、神经网络、CNN、RNN、Transformer、生成模型。
- **7 LLM 原理**：NLP 基础、Transformer、预训练、Prompt、微调、对齐。
- **8 RAG 应用**：文档处理、向量数据库、检索、评估、部署。
- **9 AI Agent**：规划、工具、记忆、MCP、Agent 协议契约、权限沙箱、多 Agent、可观测性、安全。
- **10 计算机视觉**：分类、检测、分割、OCR、视频、3D 视觉。
- **11 NLP 专题**：文本表示、分类、抽取、Seq2Seq、预训练模型。
- **12 多模态 AIGC**：视觉语言模型、图像/视频/音频生成、伦理和产品原型。
- **13 开源大模型**：本地 CPU、免费 Colab、租 GPU 三条路线，开放权重模型选择、模型运行、服务化、评估、GPU 租用纪律、LoRA 决策。

## 技术栈

| 层级 | 选择 |
|---|---|
| 静态站框架 | Astro 6 |
| 文档 UI | Astro Starlight |
| 内容格式 | Markdown / MDX-compatible Starlight docs |
| 搜索 | Starlight Pagefind integration |
| 多语言 | 英文根路径、简体中文 `/zh-cn/`、日文 `/ja/` |
| 生产域名 | `https://airoads.org` |
| 课程素材 | `public/img/course/` |
| 验证 | Markdown、内部链接、侧边栏、课程结构、图表、生成站点 QA |

## 本地运行

需要 Node.js 18 或更新版本。

```bash
npm install
npm run dev
```

构建完整静态站点：

```bash
npm run build
```

预览构建结果：

```bash
npm run serve
```

## 质量检查

常用命令：

```bash
npm run build
npm run qa:all
npm run qa:diagrams
npm run qa:dist
npm run qa:course
npm run qa:code
npm run qa:examples
npm run qa:images
npm run qa:image-teaching
npm run qa:readability
npm run qa:completion
npm run seo:indexnow:dry-run
```

`npm run qa:all` 是课程改动前后的预检命令：它会检查图表、课程质量信号、代码块、课程图片引用、图片教学信号、可读性信号和完成度评分，再进入完整构建。

`npm run qa:course` 会报告可操作的课程内容缺口。附录、导航页和 study guide 不计入折叠讲解提示，这样剩余样例会更集中地指向可能需要补 walkthrough 的正文页面。

`npm run qa:code` 会审计全课程的 fenced code blocks，提前发现围栏格式错误、语法错误和未完成的占位代码，避免示例到了学习者手里才坏掉。

`npm run qa:examples` 会抽取一组完整、无需联网的课程脚本，在临时目录中真实运行，并检查关键输出和生成文件。结果写入 `reports/code-example-smoke-tests.json`；依赖可选包的样例只有在本机缺包时才会跳过。

`npm run qa:image-teaching` 会写入 `reports/course-images/image-teaching-audit.json`，提示 alt 太弱、附近缺少讲解、单页重复用图或语言版本不匹配的图片引用。

`npm run qa:readability` 会写入 `reports/readability-audit.json`，提示密集表格、过长表头和可能应该改成卡片/窄表/终端输出样式的纯文本块。

`npm run qa:completion` 会写入 `reports/course-completion-report.json` 和 `reports/course-completion-report.md`，按语言和章节汇总最低分页面，让后续优化有稳定 backlog。

定期检查最新 AI 技术并更新课程时，使用 [COURSE_UPDATE_POLICY.md](./COURSE_UPDATE_POLICY.md)。

直接验证脚本：

```bash
python3 validate_markdown_fences.py
python3 validate_internal_links.py
python3 validate_sidebars.py
python3 validate_course_structure.py
python3 scripts/validate_course_image_refs.py
python3 scripts/audit_code_blocks.py
python3 scripts/run_example_smoke_tests.py
python3 scripts/audit_image_teaching.py
python3 scripts/audit_readability.py
python3 scripts/course_completion_report.py
```

## 目录结构

```text
src/content/docs/        英文、简体中文、日文课程内容
public/img/course/       课程图解、漫画和本地化图片
public/img/logo.svg      AI Roads 公共 Logo
public/img/social-card.png
astro.config.mjs         Astro Starlight 配置、语言、侧边栏、sitemap、元数据
scripts/                 验证、sitemap、图片生成、SEO 和维护脚本
docker/                  Docker 部署用 Nginx 配置
nginx/                   生产代理示例
COURSE_UPDATE_POLICY.md  定期 AI 技术扫描与课程更新规则
```

## 视觉素材

课程图片是教学内容的一部分，位于 `public/img/course/`。它们包括图解、结果图、漫画和本地化教学图片。新增或替换图片时，请确保图片和附近课程、代码或输出证据紧密相关。

## 参与贡献

欢迎提交 issue 和 PR。适合贡献的内容包括：

- 修复不清楚的课程说明或失效链接。
- 增加可运行示例、预期输出和排障说明。
- 同步英文、简体中文和日文内容。
- 改进课程图解和本地化教学图片。
- 加强验证、部署或 SEO 脚本。

修改课程内容时，请尽量保持三种语言结构一致、含义一致。

## 许可证

课程内容和项目代码使用 MIT License 发布。
