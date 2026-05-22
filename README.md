# AI Roads

> A free, beginner-friendly AI engineering curriculum that starts with developer foundations and grows into machine learning, deep learning, LLM applications, RAG, AI Agents, and multimodal AIGC.

## Official Site

[https://airoads.org](https://airoads.org)

The website defaults to English. Learners can switch to Simplified Chinese or Japanese from the language dropdown in the navigation bar.

## What This Project Is

This repository powers the AI Roads learning website. It is designed for beginners who want a practical, project-driven path into modern AI engineering rather than a scattered list of articles.

The site is now built with Astro Starlight as a static documentation experience. Starlight handles the docs shell, search, sidebar, table of contents, multilingual routes, sitemap output, and dark/light theme behavior, while this repository owns the course content, visual assets, branding, validation scripts, and deployment glue.

The course combines:

- Step-by-step lessons for new learners.
- Visual explanations, diagrams, and comics for difficult concepts.
- Project checkpoints that turn knowledge into portfolio work.
- Internationalized content for English, Simplified Chinese, and Japanese.
- A homepage overview that invites learners to follow airoads.org, with a future study group considered if enough learners gather.
- Deployment, validation, sitemap, and maintenance scripts for a production learning site.

## Website Stack

| Layer | Current choice |
|---|---|
| Static site framework | Astro 6 |
| Documentation UI | Astro Starlight |
| Content format | Markdown / MDX-compatible Starlight docs |
| Search index | Starlight Pagefind integration |
| Locales | English root, Simplified Chinese `/zh-cn/`, Japanese `/ja/` |
| Production domain | `https://airoads.org` |
| Social preview image | `public/img/social-card.png` |
| Favicon | `public/img/favicon.svg` and `public/img/favicon.png` |

## Course Path

The public course uses a clear, hierarchical numbering system:

```text
0         Start-here guides before Chapter 1
1-12      Main course chapters
N.0       Chapter study guide and task sheet
N.M       Section inside a chapter
N.M.K     Individual lesson page inside a section
E.X       Elective module
E.X.K     Elective lesson page
A.K       Appendix page
```

For example, `4.1.2` means Chapter 4, section 1, lesson 2. It should not appear as a local-only `1.2`, because readers who open a page directly need to know where they are in the whole course.

The recommended path is:

```mermaid
flowchart LR
  A[1 Developer Tools] --> B[2 Python]
  B --> C[3 Data Analysis]
  C --> D[4 AI Math]
  D --> E[5 Machine Learning]
  E --> F[6 Deep Learning]
  F --> G[7 LLM Principles]
  G --> H[8 RAG Applications]
  H --> I[9 Agent Systems]
  I --> J{Specialization}
  J --> K[10 Computer Vision]
  J --> L[11 NLP Specialization]
  J --> M[12 Multimodal AIGC]
```

## Learning Stations

| Station | Focus | Outcome |
|---|---|---|
| 1 Developer Tools Foundations | Terminal, Git, development environment | Run code, manage projects, and work independently |
| 2 Python Programming Foundations | Python syntax, data structures, files, OOP, projects | Build CLI tools, scrapers, APIs, and small AI API demos |
| 3 Data Analysis and Visualization | NumPy, Pandas, charts, databases | Clean, analyze, and explain real datasets |
| 4 Minimal Math Foundations for AI | Linear algebra, probability, calculus, optimization | Understand vectors, matrices, probability, gradients, and loss |
| 5 Machine Learning from Basics to Practice | Supervised learning, unsupervised learning, evaluation, features | Build prediction, churn, and segmentation projects |
| 6 Deep Learning and Transformer Foundations | Neural networks, PyTorch, CNN, RNN, Transformer, generative models | Train and diagnose deep learning models |
| 7 LLM Principles, Prompting, and Fine-Tuning | NLP, Transformer internals, pretraining, prompting, fine-tuning, alignment | Choose between prompting, RAG, fine-tuning, and alignment methods |
| 8 LLM Application Development and RAG | RAG, document processing, vector databases, deployment, evaluation | Build cited, logged, evaluated knowledge-base assistants |
| 9 AI Agents and Agentic Systems | Planning, tools, memory, MCP, multi-agent systems, safety | Build traceable AI Agent workflows with guardrails |
| 10 Computer Vision | Classification, detection, segmentation, OCR, video, 3D vision | Build visual AI projects with metrics and failure analysis |
| 11 NLP Specialization After LLMs | Text basics, embeddings, classification, extraction, Seq2Seq, pretrained models | Build text projects for QA, extraction, summarization, or semantic graphs after the main LLM/RAG/Agent path |
| 12 AIGC and Multimodal | Vision-language models, image/video/audio generation, ethics, product projects | Build multimodal creative AI prototypes |

## Beginner Learning Strategy

Treat the course as a project-upgrade path:

- First, read the learning map and station guide.
- Then follow stations 1-9 in order.
- Finally choose station 10, 11, or 12 for a specialization project.
- Do not only read pages. Each stage should leave you with something runnable, explainable, and presentable.

## Internationalization

| Locale | URL pattern | Role |
|---|---|---|
| English | `/` | Default language and canonical root experience |
| Simplified Chinese | `/zh-cn/` | Full localized course content and localized visuals |
| Japanese | `/ja/` | Full localized course content and localized visuals |

Default English content lives in `src/content/docs/`. Localized content lives under `src/content/docs/zh-cn/` and `src/content/docs/ja/`.

## Visual Learning Assets

The course includes many static diagrams, comics, and localized images under `public/img/course/`. These visuals are part of the learning experience, especially for math, machine learning, deep learning, LLMs, RAG, Agent systems, and AI history.

README intentionally stays mostly text-based so it remains fast to load, easy to maintain, and easy to read in package managers, GitHub previews, and terminals. Course visuals belong in the website pages where they can support the lesson context directly.

## Brand Assets

AI Roads uses a minimal AI learning road mark for the site logo, favicon, and social sharing card.

| Asset | Path | Notes |
|---|---|---|
| Header logo source | `src/assets/logo.svg` | Imported by Starlight from `astro.config.mjs` |
| Public logo | `public/img/logo.svg` | Reusable public SVG mark |
| Social card | `public/img/social-card.png` | 1200 x 630 PNG for Open Graph and Twitter previews |
| Favicon SVG | `public/img/favicon.svg` | Primary browser favicon |
| Favicon PNG | `public/img/favicon.png` | 32 x 32 fallback favicon |

Regenerate brand image assets after changing the brand design:

```bash
node scripts/generate_brand_assets.mjs
```

After regenerating, run `npm run build` and confirm the generated HTML still points at `https://airoads.org/img/social-card.png`.

## Repository Structure

| Path | Purpose |
|---|---|
| `src/content/docs/` | Starlight course content, including English root docs and localized `zh-cn` / `ja` docs |
| `public/img/course/` | Course diagrams, comics, and localized images |
| `public/img/logo.svg`, `public/img/social-card.png`, `public/img/favicon.*` | AI Roads brand and sharing assets |
| `src/styles/starlight.css` | Site-level style customizations |
| `astro.config.mjs` | Astro Starlight configuration, locales, sidebar, sitemap, and metadata |
| `scripts/` | Validation, sitemap, image-generation, and maintenance scripts |
| `docker/` | Nginx runtime configuration for Docker deployment |
| `.github/workflows/` | GitHub Actions deployment workflow |

Folder names such as `ch01-tools/` and `ch12-multimodal/` are maintenance paths. Learners should follow the public numbering shown in the sidebar: `0` for start-here pages, `1-12` for the main course, `E` for electives, and `A` for appendix pages.

## Local Development

Use Node.js 18 or newer.

Install dependencies:

```bash
npm install
```

Run the development site:

```bash
npm run dev
```

Build the full static site:

```bash
npm run build
```

Validate course structure, internal links, sitemap filtering, and generated HTML cleanup:

```bash
npm run validate:docs
```

Normalize Starlight doc links after moving or renaming Markdown pages:

```bash
npm run links:starlight
```

Run the generated-site QA audit against `dist/`:

```bash
npm run qa:dist
```

Serve the generated build:

```bash
npm run serve
```

Clean generated Astro output:

```bash
npm run clean
```

## Deployment

The deployment flow builds a new image while the old container keeps serving traffic. It then runs a preflight check against the new image and only replaces the production container after the new build is ready. This reduces downtime compared with stopping the old container before compilation.

The production build also removes legacy `/zh-Hans/` redirect URLs from the sitemap, strips unexpected NUL bytes from generated HTML, and runs `scripts/qa_generated_site.mjs`. That generated-site QA checks page title / H1 shape, canonical URLs, locale alternates, local asset references, image alt text, sitemap hygiene, old-domain residue, Docusaurus residue, and homepage social preview metadata. English, Simplified Chinese, and Japanese pages are generated together from Astro Starlight.

Before publishing a site change, run:

```bash
npm run build
```

For branding or metadata changes, also inspect the generated homepage HTML for canonical URLs, favicons, and social preview metadata.

## License

Course content and project code are released under the MIT License.
