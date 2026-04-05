---
title: "5.2 项目：AI 创意内容平台"
sidebar_position: 15
description: "把文生图、改图、配音和资产管理真正组织成一个多模态创意平台的作品级项目闭环。"
keywords: [creative platform, multimodal project, image generation, editing, voice, asset management]
---

# 项目：AI 创意内容平台

:::tip 本节定位
AI 创意平台特别容易做成“功能堆叠页”：

- 文生图一个按钮
- 配音一个按钮
- 改图一个按钮

但这还不够叫平台。  
平台真正的难点是：

> **把多模态能力组织成连续工作流，并把中间资产稳定管理起来。**

这一节会把它往“作品级产品项目”再推一层。
:::

## 学习目标

- 学会把多模态生成能力组织成真实创作流程
- 学会定义创意平台里的资产结构和版本逻辑
- 学会把这个题材做成有产品感的作品级项目
- 理解创意平台为什么不只是单步生成功能集合

---

## 一、什么样的题目才像“平台项目”？

一个更像作品的题目应该是：

> **做一个活动海报创作平台：用户输入需求，系统生成海报、支持一次改图、再生成宣传配音，最后导出一个完整资产包。**

### 为什么这个范围合适？

- 流程完整
- 资产明确
- 展示起来很直观

### 为什么不建议一开始就做“大而全创作平台”？

因为：

- 功能太多会把主线冲淡
- 资产管理和路由逻辑会很快失控

---

## 二、作品级创意平台最小闭环长什么样？

1. 用户给需求
2. 路由到合适模块
3. 生成初始资产
4. 在已有资产基础上做修改
5. 生成配套语音或文案
6. 导出统一内容包

只要这 6 步跑顺，项目就已经很像产品了。

## 三、推荐推进顺序

对新人来说，更稳的顺序通常是：

1. 先做单步海报生成
2. 再补一次改图
3. 再补配音资产
4. 最后再做 bundle、日志和版本管理

这样你才更容易把“平台感”一步步做出来。

---

## 四、先跑一个更像平台的工作流示例

```python
from dataclasses import dataclass, field


@dataclass
class AssetBundle:
    images: list = field(default_factory=list)
    voices: list = field(default_factory=list)
    logs: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def route_task(user_request):
    if "配音" in user_request or "语音" in user_request:
        return "tts"
    if "改图" in user_request or "修图" in user_request:
        return "image_editing"
    if "海报" in user_request or "图片" in user_request:
        return "image_generation"
    return "general"


def generate_image(prompt, style):
    return f"image_asset[{style}]::{prompt}"


def edit_image(image_name, instruction):
    return f"edited::{image_name}::{instruction}"


def generate_voice(script, speaker="default"):
    return f"voice_asset[{speaker}]::{script}"


def run_creative_project(requests):
    bundle = AssetBundle(metadata={"style": "futuristic", "project_name": "tech_event_campaign"})

    for req in requests:
        task_type = route_task(req)
        bundle.logs.append({"request": req, "task_type": task_type})

        if task_type == "image_generation":
            asset = generate_image(req, style=bundle.metadata["style"])
            bundle.images.append(asset)

        elif task_type == "image_editing" and bundle.images:
            asset = edit_image(bundle.images[-1], req)
            bundle.images.append(asset)

        elif task_type == "tts":
            asset = generate_voice(req, speaker="brand_voice")
            bundle.voices.append(asset)

    return bundle


requests = [
    "做一张科技大会海报",
    "改图：把背景改成深蓝色并加一点发光效果",
    "为这张海报生成一句宣传配音",
]

bundle = run_creative_project(requests)
print(bundle)
```

### 4.1 这个版本比前一版强在哪？

这次不只是有：

- images
- voices

还多了：

- `logs`
- 更明确的 `metadata`

这让它更接近真实平台里的：

- 资产流
- 操作流

### 4.2 为什么 `logs` 很值得展示？

因为平台项目最怕用户只看到最后结果，  
看不到中间过程。

而作品级展示里，中间过程往往就是亮点。

---

## 五、创意平台最容易失控的地方

### 5.1 资产版本混乱

例如：

- 初始图
- 改图 1
- 改图 2

如果命名和归档不清楚，系统很快就乱。

### 5.2 路由逻辑不清

例如：

- 同一句里既像图像请求又像语音请求

这会导致结果难预测。

### 5.3 多模态风格不一致

例如：

- 海报风格偏未来感
- 配音文案却像官方新闻播报

这类不一致很适合在项目里单独拿出来分析。

---

## 六、作品级创意平台最该展示什么？

建议至少展示：

1. 用户需求
2. 路由结果
3. 初始海报
4. 改图后版本
5. 配音资产
6. 最终 bundle 结构

### 为什么这比只贴一张海报更强？

因为这样别人能看到：

- 这是工作流系统
- 不是单次生成 demo

---

## 七、一个很适合补上的错误分析层

例如你可以额外记录：

- 哪类需求最容易路由错
- 哪类 prompt 最容易让图像和配音风格不一致
- 哪些资产最容易在导出时丢元数据

这会让项目显得非常成熟。

---

## 小结

这节最重要的是建立一个作品级判断：

> **AI 创意内容平台真正像平台的地方，不是功能多，而是能否把任务路由、资产版本和多步工作流组织成稳定、可展示的生产链路。**

只要这条链路讲清楚，这个项目就会非常像一个有产品感的多模态作品。

## 项目交付时最好补上的内容

- 一张工作流图
- 一段从需求到 bundle 的完整 trace
- 一组风格一致 / 不一致的对比案例
- 一段你对资产管理和版本设计的说明

---

## 练习

1. 给 `AssetBundle` 再加一个 `video_scripts` 字段，想一想它在工作流里该怎么生成。
2. 为什么创意平台比单步生成功能更依赖资产管理？
3. 如果图像和配音风格总不一致，你会把问题归到路由、提示词，还是资产层？为什么？
4. 如果你把这个项目放进作品集，首页最值得展示哪 5 个模块？
