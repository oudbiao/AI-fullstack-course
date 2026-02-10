// @ts-check
const { themes: prismThemes } = require("prism-react-renderer");

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "AI 全栈学习教程",
  tagline: "从零基础到 AI Agent 开发的完整学习路径",
  favicon: "img/favicon.ico",
  url: "https://learning.airoads.org",
  baseUrl: "/",
  organizationName: "ai-fullstack-course",
  projectName: "ai-fullstack-course",
  onBrokenLinks: "warn",

  // ===== SEO 全局 meta 标签 =====
  headTags: [
    // ----- 搜索引擎站长验证（在对应平台获取验证码后，取消注释并填入 content）-----
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
          "AI全栈学习教程,人工智能学习,Python教程,机器学习入门,深度学习,数据分析,PyTorch,LLM,大语言模型,AI Agent,自学课程,零基础学AI",
      },
    },
    {
      tagName: "meta",
      attributes: {
        name: "author",
        content: "AI 全栈学习教程",
      },
    },
    // 结构化数据 JSON-LD（帮助搜索引擎理解网站类型）
    {
      tagName: "script",
      attributes: {
        type: "application/ld+json",
      },
      innerHTML: JSON.stringify({
        "@context": "https://schema.org",
        "@type": "WebSite",
        name: "AI 全栈学习教程",
        url: "https://learning.airoads.org",
        description: "从零基础到 AI Agent 开发的完整学习路径，涵盖 Python、数据分析、机器学习、深度学习、LLM 等技术栈",
        inLanguage: "zh-Hans",
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
        name: "AI 全栈学习教程",
        description:
          "从零基础到 AI Agent 开发的完整免费学习路径，涵盖 Python 编程、数据分析与可视化、数学基础、机器学习、深度学习、计算机视觉、自然语言处理、大语言模型等",
        provider: {
          "@type": "Organization",
          name: "AI 全栈学习教程",
          sameAs: "https://github.com/oudbiao/AI-fullstack-course",
        },
        educationalLevel: "Beginner",
        isAccessibleForFree: true,
        inLanguage: "zh-Hans",
        teaches: [
          "Python 编程",
          "数据分析",
          "机器学习",
          "深度学习",
          "大语言模型",
          "AI Agent 开发",
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
    defaultLocale: "zh-Hans",
    locales: ["zh-Hans"],
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
        // Sitemap 配置（搜索引擎自动发现页面）
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
      // SEO：全局 meta 标签（会注入到每个页面的 <head> 中）
      metadata: [
        { name: "description", content: "AI 全栈学习教程 —— 从零基础到 AI Agent 开发的完整免费学习路径，涵盖 Python、数据分析、机器学习、深度学习、LLM 大语言模型等技术栈。" },
        { property: "og:type", content: "website" },
        { property: "og:locale", content: "zh_CN" },
        { property: "og:site_name", content: "AI 全栈学习教程" },
        { name: "twitter:card", content: "summary_large_image" },
      ],
      image: "img/social-card.png",
      navbar: {
        title: "AI 全栈学习教程",
        logo: {
          alt: "AI 全栈学习教程 Logo",
          src: "img/logo.svg",
        },
        items: [
          {
            type: "docSidebar",
            sidebarId: "courseSidebar",
            position: "left",
            label: "📚 课程内容",
          },
          {
            href: "https://github.com/oudbiao/AI-fullstack-course",
            label: "GitHub",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "课程",
            items: [
              { label: "课程总览", to: "/" },
              { label: "学习路线", to: "/intro/learning-path" },
              { label: "职业方向", to: "/intro/career-guide" },
            ],
          },
          {
            title: "社区",
            items: [
              { label: "GitHub", href: "https://github.com/oudbiao/AI-fullstack-course" },
              { label: "Kaggle", href: "https://www.kaggle.com/" },
              { label: "HuggingFace", href: "https://huggingface.co/" },
            ],
          },
          {
            title: "资源",
            items: [
              { label: "学习资源", to: "/appendix/resources" },
              { label: "硬件指南", to: "/appendix/hardware" },
              { label: "求职准备", to: "/appendix/job-prep" },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} AI 全栈学习教程`,
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
