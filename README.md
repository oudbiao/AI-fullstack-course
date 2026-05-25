# AI Roads

<div align="center">
  <img src="./public/img/logo.svg" width="96" alt="AI Roads logo">
  <h3>A free multilingual AI engineering curriculum for practical builders.</h3>
  <p>
    English |
    <a href="./README_zh.md">简体中文</a> |
    <a href="./README_ja.md">日本語</a>
  </p>
  <p>
    <a href="https://airoads.org">Official Site</a>
  </p>
</div>

---

AI Roads is a free, beginner-friendly AI engineering curriculum. It starts with developer foundations and grows into Python, data analysis, AI math, machine learning, deep learning, LLM principles, RAG, AI Agents, multimodal AIGC, and open-source LLM deployment.

This repository powers the public learning website at [airoads.org](https://airoads.org). The site defaults to English, with full Simplified Chinese and Japanese routes available from the language switcher.

## Project Goal

AI Roads is designed for learners who want a practical path into modern AI engineering instead of a scattered list of articles. The course emphasizes:

1. Clear learning order from tools to production-oriented AI projects.
2. Runnable lessons with commands, code, outputs, and evidence to keep.
3. Visual explanations, diagrams, and localized teaching assets.
4. Project checkpoints that turn learning into portfolio work.
5. Multilingual content for English, Simplified Chinese, and Japanese.
6. Validation and deployment scripts for a production static learning site.

The learning principle is simple: do not only read. Each stage should leave something runnable, explainable, and presentable.

## Who This Is For

AI Roads is suitable for:

- Beginners who want to enter AI engineering from the ground up.
- Developers who know some programming and want a structured AI path.
- Learners who want to build projects, not only read concepts.
- Students preparing portfolios around LLM, RAG, Agent, CV, NLP, or multimodal work.
- Course maintainers who want a multilingual static documentation site with strong QA checks.

## Course Roadmap

```text
0   Start-here guides
1   Developer Tools
2   Python Programming
3   Data Analysis and Visualization
4   Minimal Math Foundations for AI
5   Machine Learning
6   Deep Learning and Transformer Foundations
7   LLM Principles, Prompting, and Fine-Tuning
8   LLM Application Development and RAG
9   AI Agents and Agentic Systems
10  Computer Vision
11  NLP Specialization After LLMs
12  AIGC and Multimodal
13  Open-Source LLM Deployment and Fine-Tuning
E   Elective modules
A   Appendix
```

The recommended route is to finish chapters 1-9 first, then choose chapters 10-13 based on your project direction.

## Learning Stations

- **1 Developer Tools**: terminal, Git, local development environment.
- **2 Python Programming**: syntax, data structures, files, OOP, APIs, projects.
- **3 Data Analysis**: NumPy, Pandas, visualization, databases.
- **4 AI Math**: linear algebra, probability, calculus, optimization.
- **5 Machine Learning**: supervised learning, unsupervised learning, evaluation, feature engineering.
- **6 Deep Learning**: PyTorch, neural networks, CNN, RNN, Transformer, generative models.
- **7 LLM Principles**: NLP basics, Transformer internals, pretraining, prompting, fine-tuning, alignment.
- **8 RAG Applications**: document processing, vector databases, retrieval, evaluation, deployment.
- **9 AI Agents**: planning, tools, memory, MCP, multi-agent systems, observability, safety.
- **10 Computer Vision**: classification, detection, segmentation, OCR, video, 3D vision.
- **11 NLP Specialization**: text representation, classification, extraction, Seq2Seq, pretrained models.
- **12 Multimodal AIGC**: vision-language models, image/video/audio generation, ethics, product prototypes.
- **13 Open-Source LLMs**: local CPU, free Colab, and rented GPU routes; model runtime, serving, evaluation, GPU rental discipline, LoRA decisions.

## Website Stack

| Layer | Choice |
|---|---|
| Static site framework | Astro 6 |
| Documentation UI | Astro Starlight |
| Content format | Markdown / MDX-compatible Starlight docs |
| Search | Starlight Pagefind integration |
| Locales | English root, Simplified Chinese `/zh-cn/`, Japanese `/ja/` |
| Production domain | `https://airoads.org` |
| Course assets | `public/img/course/` |
| Validation | Markdown, internal links, sidebars, course structure, diagrams, generated-site QA |

## Repository Structure

```text
src/content/docs/        Course content in English, zh-cn, and ja
public/img/course/       Course diagrams, comics, and localized images
public/img/logo.svg      Public AI Roads logo
public/img/social-card.png
src/styles/starlight.css Site-level Starlight custom styles
astro.config.mjs         Astro Starlight config, locales, sidebar, sitemap, metadata
scripts/                 Validation, sitemap, image generation, SEO, and maintenance scripts
docker/                  Nginx runtime configuration for Docker deployment
nginx/                   Production proxy examples
```

## Local Development

Use Node.js 18 or newer.

```bash
npm install
npm run dev
```

Build the full static site:

```bash
npm run build
```

Serve the generated site:

```bash
npm run serve
```

## Quality Checks

The production build validates diagrams, generates the Astro/Starlight site, filters legacy sitemap redirects, strips unexpected NUL bytes, and runs generated-site QA.

Useful commands:

```bash
npm run build
npm run qa:diagrams
npm run qa:dist
npm run qa:course
npm run qa:code
npm run seo:indexnow:dry-run
```

`npm run qa:course` reports actionable content gaps. Appendix, navigation, and study-guide pages are exempt from the folded-answer advisory so the remaining samples point to lesson pages that may need clearer walkthroughs.

`npm run qa:code` audits fenced code blocks across the course for malformed fences, syntax errors, and unfinished placeholder snippets before examples reach learners.

For direct script checks:

```bash
python3 validate_markdown_fences.py
python3 validate_internal_links.py
python3 validate_sidebars.py
python3 validate_course_structure.py
python3 scripts/validate_course_image_refs.py
python3 scripts/audit_code_blocks.py
```

## Visual Assets

Course visuals are part of the learning material. They live under `public/img/course/` and include diagrams, result maps, comics, and localized teaching images. When adding or replacing course images, keep the image tied to the nearby lesson, code, or output evidence.

Brand assets:

```text
src/assets/logo.svg
public/img/logo.svg
public/img/social-card.png
public/img/favicon.svg
public/img/favicon.png
```

Regenerate brand assets after changing the brand design:

```bash
node scripts/generate_brand_assets.mjs
```

## Deployment

The deployment flow builds a new static site, validates generated pages, and serves the production output through the configured runtime. The Docker path builds a new image while the old container keeps serving traffic, then replaces it only after preflight checks pass.

Before publishing a site change, run:

```bash
npm run build
```

For SEO maintenance:

```bash
npm run seo:indexnow
```

## Contributing

Issues and pull requests are welcome. Good contributions usually fall into one of these categories:

- Fix unclear lessons or broken links.
- Add runnable examples, expected outputs, or troubleshooting notes.
- Improve multilingual consistency across English, Simplified Chinese, and Japanese.
- Improve course diagrams or localized teaching images.
- Strengthen validation, deployment, or SEO scripts.

When changing course content, keep the three language routes aligned in structure and meaning.

## License

Course content and project code are released under the MIT License.
