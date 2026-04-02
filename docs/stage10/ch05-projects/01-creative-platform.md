---
title: "5.1 项目：AI 创意内容平台"
sidebar_position: 15
description: "把文生图、改图、配音和资产管理真正组织成一个多模态创意平台项目闭环。"
keywords: [creative platform, multimodal project, image generation, editing, voice, asset management]
---

# 项目：AI 创意内容平台

:::tip 本节定位
这类项目最容易做成“功能堆叠页”：

- 文生图一个按钮
- 配音一个按钮
- 视频一个按钮

但那还不够叫平台。  
真正的平台感来自：

> **任务路由、资产管理、工作流衔接和结果导出。**

这节课会把它收成一个更像产品的项目骨架。
:::

## 学习目标

- 理解创意平台和单一生成功能的差别
- 学会设计多模态资产在平台里的流转方式
- 通过可运行示例建立任务路由 + 资产管理的闭环
- 学会把这个题材做成更像作品级产品项目

---

## 一、平台和“多个功能按钮”差在哪？

平台最核心的不是功能数量，  
而是：

- 任务路由
- 资产复用
- 多步创作流程

例如：

1. 先生成海报图
2. 再对海报做局部修改
3. 再根据海报文案生成配音
4. 最后导出成内容包

这就是平台感。

---

## 二、先定义最小产品目标

先把范围收窄成：

- 输入一个创意需求
- 产出图像资产
- 产出配音资产
- 把它们打包成一个项目结果

不要一开始就做：

- 社区
- 团队协作
- 商业化运营后台

---

## 三、先跑一个更像平台的最小示例

```python
from dataclasses import dataclass, field


@dataclass
class AssetBundle:
    images: list = field(default_factory=list)
    voices: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def route_task(user_request):
    if "配音" in user_request or "语音" in user_request:
        return "tts"
    if "海报" in user_request or "图片" in user_request:
        return "image_generation"
    if "改图" in user_request or "修图" in user_request:
        return "image_editing"
    return "general"


def generate_image(prompt, style):
    return f"image_asset[{style}]::{prompt}"


def edit_image(image_name, instruction):
    return f"edited::{image_name}::{instruction}"


def generate_voice(script, speaker="default"):
    return f"voice_asset[{speaker}]::{script}"


def run_creative_project(requests):
    bundle = AssetBundle(metadata={"style": "futuristic"})

    for req in requests:
        task_type = route_task(req)
        if task_type == "image_generation":
            bundle.images.append(generate_image(req, style="futuristic"))
        elif task_type == "image_editing" and bundle.images:
            bundle.images.append(edit_image(bundle.images[-1], req))
        elif task_type == "tts":
            bundle.voices.append(generate_voice(req, speaker="brand_voice"))

    return bundle


requests = [
    "做一张科技大会海报",
    "改图：把背景改成深蓝色并加一点发光效果",
    "为这张海报生成一句宣传配音",
]

bundle = run_creative_project(requests)
print(bundle)
```

### 3.1 这个例子最重要的地方是什么？

它已经不是简单的：

- 单次生成

而是：

- 多任务路由
- 上一步资产供下一步复用
- 最终组织成一个 bundle

这就是平台型项目最核心的结构。

### 3.2 为什么 `AssetBundle` 很关键？

因为多模态项目真正难的地方之一就是：

- 资产管理

如果没有统一 bundle，后面很快就会乱：

- 图像版本很多
- 音频文件分散
- 元数据丢失

---

## 四、平台项目最该怎么展示？

### 4.1 工作流，而不只是功能

建议展示：

1. 用户输入需求
2. 路由到了哪些模块
3. 生成了哪些中间资产
4. 最终导出结果是什么

### 4.2 资产版本

很值得展示：

- 初始海报
- 改图后版本
- 对应配音

### 4.3 失败案例

例如：

- 图像风格和文案不一致
- 配音语气和海报品牌调性不匹配

这会让项目更真实。

---

## 五、最容易踩的坑

### 5.1 只做单步调用

这样更像 demo，不像平台。

### 5.2 不做资产管理

多模态输出一多，系统很快就会乱。

### 5.3 不做工作流展示

用户看到的只是结果，很难感受到平台逻辑。

---

## 六、小结

这节最重要的是建立一个平台项目判断：

> **AI 创意内容平台的核心，不是单个生成模块多强，而是能否把多模态能力通过任务路由和资产管理组织成连续创作工作流。**

只要这条工作流讲清楚，这个项目就会很像一个真正的多模态产品。

---

## 练习

1. 给示例再加一个 `video_script` 资产类型。
2. 想一想：为什么平台项目比单功能 demo 更强调资产管理？
3. 如果图像和配音风格不一致，你会把问题归到哪一层？
4. 如果做作品集首页，你会怎样展示“从需求到资产包”的整个过程？
