import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import remarkCourseTextBlocks from "./src/utils/remarkCourseTextBlocks.mjs";

const siteDescription =
  "A complete free learning path from AI fundamentals to AI Agent development, covering Python, data analysis, machine learning, deep learning, LLMs, RAG, and AI Agents.";

const siteUrl = "https://airoads.org";
const siteTitle = "AI Roads";
const socialCardUrl = `${siteUrl}/img/social-card.png`;
const repositoryUrl = "https://github.com/oudbiao/AI-fullstack-course";
const courseKeywords = [
  "AI full-stack course",
  "learn AI",
  "AI engineering",
  "Python tutorial",
  "machine learning for beginners",
  "deep learning",
  "data analysis",
  "PyTorch",
  "LLM",
  "RAG",
  "AI agents",
];
const structuredData = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Organization",
      "@id": `${siteUrl}/#organization`,
      name: siteTitle,
      url: siteUrl,
      logo: `${siteUrl}/img/logo.svg`,
      sameAs: [repositoryUrl],
    },
    {
      "@type": "WebSite",
      "@id": `${siteUrl}/#website`,
      url: siteUrl,
      name: siteTitle,
      alternateName: ["AI Full-Stack Course", "AI 全栈工程课程"],
      description: siteDescription,
      inLanguage: ["en-US", "zh-CN", "ja-JP"],
      image: socialCardUrl,
      publisher: { "@id": `${siteUrl}/#organization` },
    },
    {
      "@type": "Course",
      "@id": `${siteUrl}/#course`,
      name: "AI Roads: AI Full-Stack Engineering Course",
      alternateName: [
        "AI Full-Stack Course",
        "AI 全栈工程课程",
        "AI フルスタックエンジニアリングコース",
      ],
      url: siteUrl,
      description: siteDescription,
      provider: {
        "@type": "Organization",
        "@id": `${siteUrl}/#organization`,
        name: siteTitle,
        url: siteUrl,
        sameAs: repositoryUrl,
      },
      publisher: { "@id": `${siteUrl}/#organization` },
      image: socialCardUrl,
      educationalLevel: "Beginner",
      isAccessibleForFree: true,
      inLanguage: ["en-US", "zh-CN", "ja-JP"],
      teaches: [
        "Python",
        "data analysis",
        "machine learning",
        "deep learning",
        "large language models",
        "retrieval-augmented generation",
        "AI agents",
        "multimodal AI applications",
      ],
      audience: {
        "@type": "Audience",
        audienceType: "Beginner AI learners and career switchers",
      },
      hasCourseInstance: {
        "@type": "CourseInstance",
        courseMode: ["online", "self-paced"],
        courseWorkload: "Self-paced 12-week learning path",
      },
    },
  ],
};

const docsRoot = fileURLToPath(new URL("./src/content/docs/", import.meta.url));

const localeLanguageTags = {
  "zh-cn": "zh-CN",
  ja: "ja-JP",
};

const sectionDefinitions = [
  {
    label: "0 Start Here",
    translations: { "zh-CN": "0 起步指南", "ja-JP": "0 はじめに" },
    directory: "intro",
    collapsed: false,
    prefixItems: [
      {
        label: "Overview",
        translations: { "zh-CN": "课程总览", "ja-JP": "コース概要" },
        link: "/",
      },
    ],
  },
  {
    label: "1 Developer Tools",
    translations: { "zh-CN": "1 开发者工具", "ja-JP": "1 開発者ツール" },
    directory: "ch01-tools",
  },
  {
    label: "2 Python",
    translations: { "zh-CN": "2 Python 编程", "ja-JP": "2 Python" },
    directory: "ch02-python",
  },
  {
    label: "3 Data Analysis",
    translations: { "zh-CN": "3 数据分析", "ja-JP": "3 データ分析" },
    directory: "ch03-data-analysis",
  },
  {
    label: "4 AI Math",
    translations: { "zh-CN": "4 AI 数学", "ja-JP": "4 AI 数学" },
    directory: "ch04-ai-math",
  },
  {
    label: "5 Machine Learning",
    translations: { "zh-CN": "5 机器学习", "ja-JP": "5 機械学習" },
    directory: "ch05-machine-learning",
  },
  {
    label: "6 Deep Learning",
    translations: { "zh-CN": "6 深度学习", "ja-JP": "6 深層学習" },
    directory: "ch06-deep-learning",
  },
  {
    label: "7 LLM Principles",
    translations: { "zh-CN": "7 LLM 原理", "ja-JP": "7 LLM 原理" },
    directory: "ch07-llm-principles",
  },
  {
    label: "8 RAG Applications",
    translations: { "zh-CN": "8 RAG 应用", "ja-JP": "8 RAG アプリ" },
    directory: "ch08-rag",
  },
  {
    label: "9 Agent Systems",
    translations: { "zh-CN": "9 Agent 系统", "ja-JP": "9 Agent システム" },
    directory: "ch09-agent",
  },
  {
    label: "10 Computer Vision",
    translations: { "zh-CN": "10 计算机视觉", "ja-JP": "10 コンピュータビジョン" },
    directory: "ch10-computer-vision",
  },
  {
    label: "11 NLP",
    translations: { "zh-CN": "11 NLP 专项", "ja-JP": "11 NLP 専門" },
    directory: "ch11-nlp",
  },
  {
    label: "12 Multimodal AIGC",
    translations: { "zh-CN": "12 多模态 AIGC", "ja-JP": "12 マルチモーダル AIGC" },
    directory: "ch12-multimodal",
  },
  {
    label: "E Electives",
    translations: { "zh-CN": "E 选修模块", "ja-JP": "E 選択モジュール" },
    directory: "electives",
  },
  {
    label: "A Appendix",
    translations: { "zh-CN": "A 附录", "ja-JP": "A 付録" },
    directory: "appendix",
  },
];

const titleCaseWordMap = {
  ai: "AI",
  aigc: "AIGC",
  api: "API",
  cnn: "CNN",
  ctc: "CTC",
  cv: "CV",
  db: "Database",
  devenv: "Development Environment",
  dl: "Deep Learning",
  gan: "GAN",
  git: "Git",
  io: "I/O",
  llm: "LLM",
  mcp: "MCP",
  ml: "Machine Learning",
  nlp: "NLP",
  numpy: "NumPy",
  oop: "OOP",
  pandas: "Pandas",
  peft: "PEFT",
  pytorch: "PyTorch",
  qa: "QA",
  rag: "RAG",
  rnn: "RNN",
  sql: "SQL",
  svm: "SVM",
  vae: "VAE",
  vscode: "VS Code",
};

const directoryLabelOverrides = {
  "ch01-tools/ch01-terminal": {
    label: "Terminal",
    translations: { "zh-CN": "命令行", "ja-JP": "ターミナル" },
  },
  "ch01-tools/ch02-git": {
    label: "Git",
    translations: { "zh-CN": "Git", "ja-JP": "Git" },
  },
  "ch01-tools/ch03-devenv": {
    label: "Development Environment",
    translations: { "zh-CN": "开发环境", "ja-JP": "開発環境" },
  },
  "ch01-tools/ch04-workshop": {
    label: "Tools Workshop",
    translations: { "zh-CN": "工具实战", "ja-JP": "ツール実践" },
  },
  "ch02-python/ch01-basics": {
    label: "Python Basics",
    translations: { "zh-CN": "Python 基础", "ja-JP": "Python 基礎" },
  },
  "ch02-python/ch02-advanced": {
    label: "Python Advanced",
    translations: { "zh-CN": "Python 进阶", "ja-JP": "Python 応用" },
  },
  "ch02-python/ch03-projects": {
    label: "Python Projects",
    translations: { "zh-CN": "Python 项目", "ja-JP": "Python プロジェクト" },
  },
  "ch03-data-analysis/ch01-warmup": {
    label: "Data Warmup",
    translations: { "zh-CN": "数据热身", "ja-JP": "データ準備" },
  },
  "ch03-data-analysis/ch02-numpy": {
    label: "NumPy",
    translations: { "zh-CN": "NumPy", "ja-JP": "NumPy" },
  },
  "ch03-data-analysis/ch06-projects": {
    label: "Data Analysis Projects",
    translations: { "zh-CN": "数据分析项目", "ja-JP": "データ分析プロジェクト" },
  },
};

function readText(filePath) {
  return fs.existsSync(filePath) ? fs.readFileSync(filePath, "utf8") : "";
}

function cleanFrontmatterValue(value) {
  const trimmed = value.trim();
  if (!trimmed) return "";
  const quoted = trimmed.match(/^(['"])(.*)\1$/);
  return quoted ? quoted[2].replace(/\\"/g, '"').replace(/\\'/g, "'") : trimmed;
}

function parseFrontmatter(filePath) {
  const text = readText(filePath);
  if (!text.startsWith("---")) return {};
  const end = text.indexOf("\n---", 3);
  if (end === -1) return {};

  const frontmatter = text.slice(3, end);
  const title = frontmatter.match(/^title:\s*(.+)$/m)?.[1];
  const sidebarStart = frontmatter.match(/^sidebar:\s*$/m);
  const sidebarBlock = sidebarStart
    ? frontmatter
        .slice(sidebarStart.index + sidebarStart[0].length)
        .match(/^\n((?:[ \t]+.*(?:\n|$))*)/)?.[1] ?? ""
    : "";
  const sidebarLabel = sidebarBlock.match(/^\s+label:\s*(.+)$/m)?.[1];
  const sidebarOrder = sidebarBlock.match(/^\s+order:\s*(-?\d+(?:\.\d+)?)\s*$/m)?.[1];
  const sidebarHidden = /^\s+hidden:\s*true\s*$/m.test(sidebarBlock);

  return {
    title: title ? cleanFrontmatterValue(title) : undefined,
    sidebarLabel: sidebarLabel ? cleanFrontmatterValue(sidebarLabel) : undefined,
    order: sidebarOrder === undefined ? undefined : Number(sidebarOrder),
    hidden: sidebarHidden,
  };
}

function slugFromRelativeFile(relativeFile) {
  const withoutExtension = relativeFile.replace(/\.mdx?$/, "");
  if (withoutExtension === "index") return "index";
  return withoutExtension.endsWith("/index")
    ? withoutExtension.slice(0, -"/index".length)
    : withoutExtension;
}

function numericPrefixOrder(name) {
  const match = name.match(/^(?:ch)?(\d+)(?:-|$)/);
  return match ? Number(match[1]) : Number.MAX_VALUE;
}

const specialFileOrders = {
  "index.md": 0,
  "00-roadmap.md": 0,
  "study-guide.md": 0.5,
};

function fileOrder(relativeFile) {
  const meta = parseFrontmatter(path.join(docsRoot, relativeFile));
  const fileName = path.basename(relativeFile);
  return specialFileOrders[fileName] ?? meta.order ?? numericPrefixOrder(fileName);
}

function prettifyDirectoryName(relativeDir) {
  const baseName = path.basename(relativeDir).replace(/^ch\d+-/, "").replace(/^\d+-/, "");
  return baseName
    .split("-")
    .filter(Boolean)
    .map((word) => titleCaseWordMap[word.toLowerCase()] ?? word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

function joinDirectoryPrefix(prefix, label) {
  if (!prefix || !label) return label || prefix;
  const escapedPrefix = prefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  if (new RegExp(`^${escapedPrefix}(?:\\s|$)`).test(label)) return label;
  return `${prefix} ${label}`;
}

function stripNavigationNumberPrefix(label) {
  return label
    .replace(/^(?:\d+(?:\.\d+)*|[A-Z](?:\.[A-Z0-9]+)+)\s+/i, "")
    .trim();
}

function shortGroupLabelFromTitle(title) {
  return stripNavigationNumberPrefix(title)
    .split(/[:：]/)[0]
    .replace(/\s*(?:Learning\s+)?(?:Roadmap|Learning Path|Path)\s*$/i, "")
    .replace(/\s*(?:路线图|学习路线|ロードマップ)\s*$/u, "")
    .trim();
}

function firstMarkdownFile(relativeDir) {
  const absoluteDir = path.join(docsRoot, relativeDir);
  if (!fs.existsSync(absoluteDir)) return undefined;

  return fs
    .readdirSync(absoluteDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && /\.mdx?$/.test(entry.name) && entry.name !== "index.md")
    .map((entry) => path.posix.join(relativeDir, entry.name))
    .sort((a, b) => fileOrder(a) - fileOrder(b) || a.localeCompare(b))[0];
}

function indexFileForDirectory(relativeDir) {
  for (const fileName of ["00-roadmap.md", "index.md"]) {
    const relativeFile = path.posix.join(relativeDir, fileName);
    if (fs.existsSync(path.join(docsRoot, relativeFile))) return relativeFile;
  }
  return undefined;
}

function sourceFileForDirectory(relativeDir) {
  const indexFile = indexFileForDirectory(relativeDir);
  if (indexFile) return indexFile;
  return firstMarkdownFile(relativeDir);
}

function localizedTitle(relativeFile, locale) {
  const localeFile = path.join(docsRoot, locale, relativeFile);
  const meta = parseFrontmatter(localeFile);
  return meta.sidebarLabel || meta.title;
}

function fallbackDirectoryLabel(relativeDir, languageTag, prefix) {
  const override = directoryLabelOverrides[relativeDir];
  const baseLabel =
    (languageTag && override?.translations?.[languageTag]) ||
    override?.label ||
    prettifyDirectoryName(relativeDir);
  return joinDirectoryPrefix(prefix, baseLabel);
}

function sourceTitleForDirectory(relativeFile, locale) {
  if (!relativeFile) return "";
  if (locale) return localizedTitle(relativeFile, locale) ?? "";
  const meta = parseFrontmatter(path.join(docsRoot, relativeFile));
  return meta.sidebarLabel || meta.title || "";
}

function sourceTitleForFile(relativeFile, locale) {
  if (locale) return localizedTitle(relativeFile, locale) ?? "";
  const meta = parseFrontmatter(path.join(docsRoot, relativeFile));
  return meta.sidebarLabel || meta.title || "";
}

const specialSidebarLabels = {
  index: {
    label: "Overview",
    translations: { "zh-CN": "概览", "ja-JP": "概要" },
  },
  studyGuide: {
    label: "Study Guide & Tasks",
    translations: { "zh-CN": "学习指南与任务单", "ja-JP": "学習ガイドとタスクリスト" },
  },
  roadmap: {
    label: "Roadmap",
    translations: { "zh-CN": "路线图", "ja-JP": "ロードマップ" },
  },
};

function specialSidebarLabelForFile(relativeFile, languageTag) {
  const fileName = path.basename(relativeFile);
  const key =
    fileName === "index.md"
      ? "index"
      : fileName === "study-guide.md"
        ? "studyGuide"
        : fileName === "00-roadmap.md"
          ? "roadmap"
          : undefined;
  if (!key) return undefined;

  const labels = specialSidebarLabels[key];
  return (languageTag && labels.translations[languageTag]) || labels.label;
}

function sidebarLabelForFile(relativeFile, locale, languageTag) {
  const specialLabel = specialSidebarLabelForFile(relativeFile, languageTag);
  if (specialLabel) return specialLabel;

  const title = sourceTitleForFile(relativeFile, locale) || sourceTitleForFile(relativeFile);
  if (title) return stripNavigationNumberPrefix(title);

  return prettifyDirectoryName(relativeFile.replace(/\.mdx?$/, ""));
}

function sidebarLabelTranslations(relativeFile) {
  const translations = {};
  for (const [locale, lang] of Object.entries(localeLanguageTags)) {
    translations[lang] = sidebarLabelForFile(relativeFile, locale, lang);
  }
  return translations;
}

function directoryLabelFromSource(relativeDir, sourceFile, locale, languageTag, prefix) {
  const override = directoryLabelOverrides[relativeDir];
  if (override) return fallbackDirectoryLabel(relativeDir, languageTag, prefix);

  const sourceTitle = sourceTitleForDirectory(sourceFile, locale);
  const shortTitle = sourceTitle ? shortGroupLabelFromTitle(sourceTitle) : "";
  const baseLabel = shortTitle || fallbackDirectoryLabel(relativeDir, languageTag, "");
  return joinDirectoryPrefix(prefix, baseLabel);
}

function groupLabelInfo(relativeDir) {
  const sourceFile = sourceFileForDirectory(relativeDir);
  const label = directoryLabelFromSource(relativeDir, sourceFile, undefined, undefined, "");
  const translations = {};

  for (const [locale, lang] of Object.entries(localeLanguageTags)) {
    translations[lang] = directoryLabelFromSource(relativeDir, sourceFile, locale, lang, "");
  }

  return { label, translations };
}

function directoryOrder(relativeDir) {
  const baseName = path.basename(relativeDir);
  const numericOrder = numericPrefixOrder(baseName);
  if (numericOrder !== Number.MAX_VALUE) return numericOrder;

  const sourceFile = sourceFileForDirectory(relativeDir);
  return sourceFile ? fileOrder(sourceFile) : Number.MAX_VALUE;
}

function buildDirectoryItems(relativeDir) {
  const absoluteDir = path.join(docsRoot, relativeDir);
  if (!fs.existsSync(absoluteDir)) return [];

  const entries = fs.readdirSync(absoluteDir, { withFileTypes: true }).flatMap((entry) => {
    if (entry.isDirectory()) {
      if (entry.name === "zh-cn" || entry.name === "ja") return [];
      const childDir = path.posix.join(relativeDir, entry.name);
      const items = buildDirectoryItems(childDir);
      if (!items.length) return [];
      const { label, translations } = groupLabelInfo(childDir);
      return [
        {
          order: directoryOrder(childDir),
          label,
          item: { label, translations, collapsed: true, items },
        },
      ];
    }

    if (!entry.isFile() || !/\.mdx?$/.test(entry.name)) return [];
    const relativeFile = path.posix.join(relativeDir, entry.name);
    const meta = parseFrontmatter(path.join(docsRoot, relativeFile));
    if (meta.hidden) return [];

    const slug = slugFromRelativeFile(relativeFile);
    const label = sidebarLabelForFile(relativeFile);
    return [
      {
        order: fileOrder(relativeFile),
        label,
        item: { slug, label, translations: sidebarLabelTranslations(relativeFile) },
      },
    ];
  });

  return entries
    .sort((a, b) => a.order - b.order || a.label.localeCompare(b.label))
    .map((entry) => entry.item);
}

const sidebarGroups = sectionDefinitions.map((section) => ({
  label: section.label,
  translations: section.translations,
  collapsed: section.collapsed ?? true,
  items: [...(section.prefixItems ?? []), ...buildDirectoryItems(section.directory)],
}));

function courseDiagramRenderer() {
  return {
    name: "ai-roads-course-diagram-renderer",
    hooks: {
      "astro:config:setup": ({ injectScript }) => {
        injectScript("page", 'import "/src/scripts/render-course-diagrams.js";');
        injectScript("page", 'import "/src/scripts/code-block-wrap-toggle.js";');
      },
    },
  };
}

export default defineConfig({
  site: siteUrl,
  trailingSlash: "ignore",
  vite: {
    build: {
      chunkSizeWarningLimit: 1200,
    },
  },
  markdown: {
    remarkPlugins: [remarkCourseTextBlocks],
  },
  integrations: [
    courseDiagramRenderer(),
    starlight({
      title: siteTitle,
      description: siteDescription,
      logo: {
        src: "./src/assets/logo.svg",
        alt: "AI Roads",
      },
      favicon: "/img/favicon.svg",
      locales: {
        root: { label: "English", lang: "en-US" },
        "zh-cn": { label: "简体中文", lang: "zh-CN" },
        ja: { label: "日本語", lang: "ja-JP" },
      },
      defaultLocale: "root",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: repositoryUrl,
        },
      ],
      editLink: {
        baseUrl: `${repositoryUrl}/edit/main/`,
      },
      expressiveCode: {
        themes: ["starlight-dark", "github-light-high-contrast"],
        useStarlightUiThemeColors: true,
        shiki: {
          langAlias: {
            "course-map": "plaintext",
          },
        },
      },
      credits: false,
      customCss: ["/src/styles/starlight.css"],
      head: [
        {
          tag: "meta",
          attrs: {
            name: "keywords",
            content: courseKeywords.join(", "),
          },
        },
        {
          tag: "meta",
          attrs: {
            name: "author",
            content: "AI Roads",
          },
        },
        {
          tag: "meta",
          attrs: {
            name: "robots",
            content: "index, follow, max-image-preview:large",
          },
        },
        {
          tag: "meta",
          attrs: {
            property: "og:type",
            content: "website",
          },
        },
        {
          tag: "meta",
          attrs: {
            property: "og:site_name",
            content: siteTitle,
          },
        },
        {
          tag: "meta",
          attrs: {
            property: "og:image",
            content: socialCardUrl,
          },
        },
        {
          tag: "meta",
          attrs: {
            property: "og:image:width",
            content: "1200",
          },
        },
        {
          tag: "meta",
          attrs: {
            property: "og:image:height",
            content: "630",
          },
        },
        {
          tag: "meta",
          attrs: {
            property: "og:image:alt",
            content: "AI Roads practical AI learning path",
          },
        },
        {
          tag: "meta",
          attrs: {
            name: "twitter:card",
            content: "summary_large_image",
          },
        },
        {
          tag: "meta",
          attrs: {
            name: "twitter:image",
            content: socialCardUrl,
          },
        },
        {
          tag: "meta",
          attrs: {
            name: "twitter:image:alt",
            content: "AI Roads practical AI learning path",
          },
        },
        {
          tag: "script",
          attrs: { type: "application/ld+json" },
          content: JSON.stringify(structuredData),
        },
      ],
      sidebar: sidebarGroups,
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 4,
      },
      lastUpdated: false,
      pagination: true,
      disable404Route: true,
    }),
  ],
});
