---
title: "12.5.1 综合项目路线图：创意内容包工作流"
sidebar_position: 0
description: "AIGC 综合项目章的简明实操路线图：把 brief 转成文案、图像提示词、视频脚本、素材版本、审核和导出。"
keywords: [AIGC 项目指南, 创意平台, 多模态项目, 内容生成工作流]
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

如果缺少这个状态结构，项目就容易像 demo，而不是产品。

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
