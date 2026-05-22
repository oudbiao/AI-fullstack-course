---
title: "12.5.1 综合项目路线图：创意内容包工作流"
description: "AIGC 综合项目章的简明实操路线图：把 brief 转成文案、图像提示词、视频脚本、素材版本、审核和导出。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AIGC 项目指南, 创意平台, 多模态项目, 内容生成工作流"
---

# 12.5.1 综合项目路线图：创意内容包工作流

毕业项目不是把很多模型 API 串起来，而是给用户一个完整工作流：输入 brief，生成素材，对比版本，人工编辑，审核，最后导出可用内容包。

## 先看产品闭环

![AIGC 创意平台项目交付闭环图](/img/course/ch12-projects-delivery-loop.webp)

![创意内容包流程图](/img/course/ch12-workshop-creative-package-pipeline-map.webp)

![提示词、素材与版本图](/img/course/ch12-workshop-prompt-asset-version-map.webp)

![审核与导出图](/img/course/ch12-workshop-review-export-map.webp)

先养成一个习惯：每个生成结果都要作为素材保存，带上来源、提示词、版本、审核状态和导出目标。

## 建一个最小内容包状态

```python
brief = {
    "topic": "RAG mini course",
    "audience": "new learners",
}
package = {
    "brief_ready": True,
    "assets": ["title", "cover_prompt", "video_script", "review_checklist"],
    "has_versions": True,
    "has_review": True,
}

ready = package["brief_ready"] and package["has_versions"] and package["has_review"] and len(package["assets"]) >= 4

print("package_ready:", ready)
print("assets:", ", ".join(package["assets"]))
```

预期输出：

```text
package_ready: True
assets: title, cover_prompt, video_script, review_checklist
```

![最小内容包状态验收运行结果图](/img/course/ch12-package-state-readiness-result-map.webp)

如果缺少这个状态结构，项目就容易像演示，而不是产品。

## 先从工作坊开始

扩展大型创意平台之前，先运行 [12.5.3 实践：构建可复现的多模态创意内容包](./02-hands-on-multimodal-workshop.md)。它会给你一个最小可复现闭环：brief 收集、提示词记录、素材版本、分镜导出、安全审核和失败分析。

## 项目交付标准

| 交付物 | 最低要求 |
|---|---|
| README | 写清目标、运行命令、依赖、素材来源和示例 |
| 内容包样例 | 1 个完整 brief，包含生成素材和审核备注 |
| 版本记录 | 至少 2 个候选输出，或 1 次人工修改记录 |
| 安全审核 | 版权、肖像、声音、敏感内容、导出标识 |
| 失败记录 | 1 个真实失败案例和下一步修复计划 |

## 通过标准

项目能接收 brief，生成结构化创意内容包，保存版本，执行审核，并导出别人可以检查的 Markdown 或 JSON，就算通过本章。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要说清涉及哪些模态、输入输出契约是什么，以及文字、图像、音频或视频证据如何对齐。
2. 证据应包含真实媒体产物或 trace，并附上质量、安全和失败案例说明。
3. 自检时要能判断任务需要的是生成、理解、检索、工具编排还是人工复核，而不是把所有多模态问题都当成同一种 demo。

</details>


## 留下的证据

学完这一页，至少保留这张证据卡：

```text
简介：用户目标、受众、素材、约束和导出格式
工件：源文件、提示词、生成候选、选定输出和被拒绝版本
审查：事实检查、版权/肖像/敏感内容检查，以及人工决定
集成: RAG 记录、Agent trace、创意包、故事板或导出预览
期望产出：可复现的资产包，包含 README、复查清单和失败说明
```
