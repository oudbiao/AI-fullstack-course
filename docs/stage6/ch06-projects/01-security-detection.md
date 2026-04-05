---
title: "6.2 项目：智能安防系统"
sidebar_position: 18
description: "围绕一个可展示的安防检测系统，从目标定义、规则、告警去重、评估到展示方式，建立作品级视觉项目闭环。"
keywords: [security detection, surveillance, alerting, tracking, false alarm, vision project]
---

# 项目：智能安防系统

:::tip 本节定位
安防项目很容易做成“检测到人就画框”的 demo。  
但真正能交付的安防系统，关注的通常不是框本身，而是：

- 告警准不准
- 会不会重复报警
- 时延够不够低
- 误报会不会把人烦死

所以这节课的重点是把它做成一个**作品级系统项目**，而不是单次检测展示。
:::

## 学习目标

- 学会定义一个可交付的安防检测任务边界
- 学会把检测、规则、告警和去重串成一条闭环
- 学会设计最基础的评估与失败分析
- 学会把这个项目做成有说服力的作品集展示

---

## 一、先把项目题目定义清楚

一个适合练手、又很像真实业务的题目可以是：

> **做一个“禁区入侵告警系统”，输入监控帧序列，输出“是否触发告警 + 告警发生在哪一帧”。**

这个题目好在：

- 目标简单
- 业务意义清楚
- 很容易解释误报和漏报

### 为什么不建议一开始就做很大？

例如：

- 同时做烟火检测、摔倒检测、安全帽检测、车辆识别

这种范围太大，项目容易只剩功能堆叠，没有一个清楚主线。

---

## 二、作品级安防项目最小闭环长什么样？

1. 定义监控目标和禁区
2. 做检测
3. 把检测框映射成告警逻辑
4. 做去重 / 跟踪
5. 评估告警质量
6. 展示成功与失败案例

如果只做前两步，那更像模型 demo；  
做到后面几步，才更像一个系统项目。

---

## 三、先跑一个“检测 -> 告警 -> 去重”的最小闭环

下面这个示例会做三件非常关键的事：

1. 读取逐帧检测结果
2. 判断是否进入危险区域
3. 对同一目标的连续多帧命中做告警去重

```python
detections = [
    {"frame": 1, "track_id": 101, "label": "person", "box": (40, 40, 80, 120)},
    {"frame": 2, "track_id": 101, "label": "person", "box": (42, 42, 82, 122)},
    {"frame": 3, "track_id": 101, "label": "person", "box": (44, 45, 84, 125)},
    {"frame": 4, "track_id": 202, "label": "person", "box": (150, 150, 180, 210)},
]

danger_zone = (30, 30, 100, 140)


def is_inside(box, zone):
    bx1, by1, bx2, by2 = box
    zx1, zy1, zx2, zy2 = zone
    return bx1 >= zx1 and by1 >= zy1 and bx2 <= zx2 and by2 <= zy2


def build_alerts(detections, zone):
    active_tracks = set()
    alerts = []

    for det in detections:
        inside = det["label"] == "person" and is_inside(det["box"], zone)
        if inside and det["track_id"] not in active_tracks:
            alerts.append(
                {
                    "frame": det["frame"],
                    "track_id": det["track_id"],
                    "alert": "intrusion",
                }
            )
            active_tracks.add(det["track_id"])
        elif not inside and det["track_id"] in active_tracks:
            active_tracks.remove(det["track_id"])

    return alerts


alerts = build_alerts(detections, danger_zone)
print(alerts)
```

### 3.1 这个例子为什么比“检测到人就报警”强得多？

因为它已经体现了安防系统里最重要的一层工程判断：

- 同一个人连续 3 帧都在禁区
- 不能报警 3 次

### 3.2 为什么 `track_id` 很重要？

没有跟踪信息，你很难判断：

- 这是同一个人
- 还是三个不同的人

所以安防项目从“检测”走向“系统”，  
往往就卡在这一层。

---

## 四、一个作品级项目最该怎么评估？

### 4.1 不是只看检测精度

安防项目更应该至少拆成两层评估：

1. 检测层  
   目标有没有找到
2. 告警层  
   告警触发是否合理

### 4.2 最小告警评估示例

```python
pred_alerts = [
    {"frame": 1, "track_id": 101, "alert": "intrusion"},
]

gold_alerts = [
    {"frame": 1, "track_id": 101, "alert": "intrusion"},
    {"frame": 8, "track_id": 303, "alert": "intrusion"},
]


def alert_recall(pred_alerts, gold_alerts):
    gold_set = {(x["frame"], x["track_id"], x["alert"]) for x in gold_alerts}
    pred_set = {(x["frame"], x["track_id"], x["alert"]) for x in pred_alerts}
    hit = len(gold_set & pred_set)
    return hit / len(gold_set) if gold_set else 1.0


print("alert_recall:", round(alert_recall(pred_alerts, gold_alerts), 4))
```

### 4.3 为什么“告警层指标”比 mAP 更像项目价值？

因为实际交付给用户的不是框，而是：

- 告警

如果检测很准但告警策略很糟，  
项目实际体验仍然会很差。

---

## 五、安防项目最值得展示的失败案例

### 5.1 误报

例如：

- 背景人像海报被识别成真人

### 5.2 漏报

例如：

- 光线很暗时漏掉入侵者

### 5.3 重复报警

例如：

- 同一个目标每帧都触发一次

### 5.4 为什么这些要单独展示？

因为这类失败恰好最能体现：

- 系统设计是否成熟

---

## 六、怎么把这个项目做成作品级展示？

建议页面至少展示：

1. 任务定义
2. 危险区域示意图
3. 检测结果和告警结果的对比
4. 一条连续视频片段中的告警轨迹
5. 误报 / 漏报 / 重复报警分析

这样它就不再像“检测 demo”，而更像完整安防系统。

---

## 七、最常见误区

### 7.1 只看模型精度，不看告警体验

### 7.2 没有去重逻辑

### 7.3 只展示成功视频

---

## 小结

这节最重要的是建立一个作品级判断：

> **安防系统的价值，不在于它能画出多少框，而在于它能否把检测结果稳定、低误报地转成可信告警。**

只要这一层做扎实，这个项目就会非常像真实业务系统。

---

## 练习

1. 给示例再加一个 `helmet` 类别，并设计“未戴安全帽”告警。
2. 想一想：为什么安防项目比普通检测项目更需要跟踪和去重？
3. 如果误报很多，你会先查数据、模型还是告警逻辑？为什么？
4. 如果把这个项目做成作品集，你最想突出哪一条完整视频 trace？
