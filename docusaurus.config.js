// @ts-check
const { themes: prismThemes } = require("prism-react-renderer");

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "AI Full-Stack Learning Course",
  tagline: "A complete learning path from AI fundamentals to AI Agent development",
  favicon: "img/favicon.ico",
  url: "https://learning.airoads.org",
  baseUrl: "/",
  organizationName: "ai-fullstack-course",
  projectName: "ai-fullstack-course",
  onBrokenLinks: "warn",

  // ===== Global SEO meta tags =====
  headTags: [
    // ----- Search engine webmaster verification. Uncomment after copying each platform token. -----
    // Google Search Console: https://search.google.com/search-console
    // {
    //   tagName: "meta",
    //   attributes: {
    //     name: "google-site-verification",
    //     content: "从 Google Search Console 复制的验证码",
    //   },
    // },
    // Bing Webmaster: https://www.bing.com/webmasters
    // {
    //   tagName: "meta",
    //   attributes: {
    //     name: "msvalidate.01",
    //     content: "从 Bing Webmaster 复制的验证码",
    //   },
    // },
    // 百度搜索资源平台: https://ziyuan.baidu.com
    // {
    //   tagName: "meta",
    //   attributes: {
    //     name: "baidu-site-verification",
    //     content: "6dfc73e25c48a3078c0e61b8dd196079",
    //   },
    // },
    {
      tagName: "meta",
      attributes: {
        name: "keywords",
        content:
          "AI full-stack course,learn AI,Python tutorial,machine learning for beginners,deep learning,data analysis,PyTorch,LLM,large language models,RAG,AI agents,AI フルスタック学習,人工知能 入門,Python チュートリアル,機械学習 初心者,深層学習,データ分析,大規模言語モデル,AI Agent",
      },
    },
    {
      tagName: "meta",
      attributes: {
        name: "author",
        content: "AI Full-Stack Learning Course",
      },
    },
    // JSON-LD structured data helps search engines understand the site.
    {
      tagName: "script",
      attributes: {
        type: "application/ld+json",
      },
      innerHTML: JSON.stringify({
        "@context": "https://schema.org",
        "@type": "WebSite",
        name: "AI Full-Stack Learning Course / AI フルスタック学習コース",
        url: "https://learning.airoads.org",
        description:
          "A complete learning path from AI fundamentals to AI Agent development. AI の基礎から AI Agent 開発まで学べる学習コース。",
        inLanguage: ["en-US", "zh-Hans", "ja-JP"],
      }),
    },
    {
      tagName: "script",
      attributes: {
        type: "application/ld+json",
      },
      innerHTML: JSON.stringify({
        "@context": "https://schema.org",
        "@type": "Course",
        name: "AI Full-Stack Learning Course / AI フルスタック学習コース",
        description:
          "A free AI full-stack course covering Python, data analysis, math, machine learning, deep learning, LLMs, RAG, and AI Agents. AI 入門から RAG・Agent 開発まで学べる無料コース。",
        provider: {
          "@type": "Organization",
          name: "AI Full-Stack Learning Course / AI フルスタック学習コース",
          sameAs: "https://github.com/oudbiao/AI-fullstack-course",
        },
        educationalLevel: "Beginner",
        isAccessibleForFree: true,
        inLanguage: ["en-US", "zh-Hans", "ja-JP"],
        teaches: [
          "Python programming",
          "Data analysis",
          "Machine learning",
          "Deep learning",
          "Large language models",
          "AI Agent development",
          "Python プログラミング",
          "データ分析",
          "機械学習",
          "深層学習",
          "大規模言語モデル",
          "AI Agent 開発",
        ],
      }),
    },
  ],

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
  },
  themes: ["@docusaurus/theme-mermaid"],
  i18n: {
    defaultLocale: "en",
    locales: ["en", "zh-Hans", "ja"],
    localeConfigs: {
      en: {
        label: "English",
        htmlLang: "en-US",
      },
      "zh-Hans": {
        label: "简体中文",
        htmlLang: "zh-Hans",
      },
      ja: {
        label: "日本語",
        htmlLang: "ja-JP",
      },
    },
  },
  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          routeBasePath: "/",
          showLastUpdateTime: false,
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
        // Sitemap configuration for search engine discovery.
        sitemap: {
          lastmod: "date",
          changefreq: "weekly",
          priority: 0.5,
          filename: "sitemap.xml",
        },
      }),
    ],
  ],
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Global metadata injected into each page.
      metadata: [
        { name: "description", content: "AI Full-Stack Learning Course / AI フルスタック学習コース — a complete free learning path from AI fundamentals to AI Agent development, covering Python, data analysis, machine learning, deep learning, LLMs, RAG, and AI Agents." },
        { property: "og:type", content: "website" },
        { property: "og:locale", content: "en_US" },
        { property: "og:site_name", content: "AI Full-Stack Learning Course" },
        { name: "twitter:card", content: "summary_large_image" },
      ],
      image: "img/social-card.png",
      navbar: {
        title: "AI Full-Stack Learning Course",
        logo: {
          alt: "AI Full-Stack Learning Course Logo",
          src: "img/logo.svg",
        },
        items: [
          {
            type: "docSidebar",
            sidebarId: "courseSidebar",
            position: "left",
            label: "📚 Course Content",
          },
          {
            href: "https://github.com/oudbiao/AI-fullstack-course",
            label: "GitHub",
            position: "right",
          },
          {
            type: "localeDropdown",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Course",
            items: [
              { label: "Course Overview", to: "/" },
              { label: "Learning Path", to: "/intro/learning-path" },
            ],
          },
          {
            title: "Community",
            items: [
              { label: "GitHub", href: "https://github.com/oudbiao/AI-fullstack-course" },
              { label: "Kaggle", href: "https://www.kaggle.com/" },
              { label: "Hugging Face", href: "https://huggingface.co/" },
            ],
          },
          {
            title: "Resources",
            items: [
              { label: "Learning Resources", to: "/appendix/resources" },
              { label: "Hardware Guide", to: "/appendix/hardware" },
              { label: "Job Preparation", to: "/appendix/job-prep" },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} AI Full-Stack Learning Course`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ["python", "bash", "json", "sql", "cpp"],
      },
      tableOfContents: {
        minHeadingLevel: 2,
        maxHeadingLevel: 4,
      },
      colorMode: {
        defaultMode: "light",
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      mermaid: {
        theme: {
          light: "default",
          dark: "dark",
        },
        options: {
          themeVariables: {
            fontSize: "14px",
          },
        },
      },
    }),
};

module.exports = config;
