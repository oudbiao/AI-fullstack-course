---
title: "6.6 AutoGen【选修】"
sidebar_position: 34
description: "从多 Agent 对话、角色协作到代码执行回路，理解 AutoGen 风格框架为什么特别适合“会话式多 Agent”系统。"
keywords: [AutoGen, multi-agent, role dialogue, conversation loop, code execution, agent collaboration]
---

# AutoGen【选修】

:::tip 本节定位
如果说有些框架更像“工作流图”或“知识组织层”，那 AutoGen 给人的第一感觉通常是：

> **多个 Agent 通过一轮轮消息对话来协作完成任务。**

它的关键不是“角色多”，而是“对话推进任务”这件事。
:::

## 学习目标

- 理解 AutoGen 风格系统为什么强调多 Agent 对话
- 分清它和 LangGraph / CrewAI 这类框架的核心差异
- 看懂一个最小的 AutoGen 风格消息循环
- 知道这类框架什么时候特别适合用，什么时候容易失控

---

## 一、AutoGen 风格最核心的直觉是什么？

### 1.1 它不是先把流程画成图，而是先让角色“说起来”

很多框架会先问：

- 当前状态是什么？
- 下一步去哪个节点？

AutoGen 风格更像在问：

- planner 应该怎么给 coder 发任务？
- coder 写完后怎么把结果交给 reviewer？
- reviewer 的反馈又怎样继续推动下一轮？

也就是说，它把系统抽象成：

> **一组会彼此发送消息的角色。**

### 1.2 一个生活里的类比

你可以把 AutoGen 风格系统想成一个群聊工作群：

- 产品经理说需求
- 开发接任务
- 评审反馈问题
- 大家继续来回讨论

这个类比非常重要，因为它直接决定了这种框架擅长的任务形状。

---

## 二、为什么这种“对话式多 Agent”会让人觉得很自然？

因为很多复杂任务本来就有这种形态：

- 先提需求
- 再尝试执行
- 再根据反馈修正

例如：

- 代码生成与审查
- 研究报告撰写
- 问题排查

这些任务本身不像一条直线，而更像多轮往返。  
所以 AutoGen 风格的抽象很贴近人类协作直觉。

---

## 三、一个最小的 AutoGen 风格示例

先不用真实框架，先用纯 Python 把“多轮对话协作”的味道走通。

```python
messages = []

def send(sender, receiver, content):
    msg = {
        "from": sender,
        "to": receiver,
        "content": content
    }
    messages.append(msg)
    return msg

send("planner", "coder", "请实现一个判断退款资格的函数。")
send("coder", "reviewer", "我写好了第一版，请帮我检查。")
send("reviewer", "coder", "请补上学习进度超过 20% 的处理逻辑。")

for msg in messages:
    print(msg)
```

### 3.2 这段代码虽然简单，但它在教什么？

它在教你：

- 协作单位是“消息”
- 系统推进依赖“谁对谁说了什么”
- 多 Agent 不一定非要先有显式状态图

这就是 AutoGen 风格最核心的入口。

---

## 四、为什么 AutoGen 经常和“代码执行”场景绑在一起？

### 4.1 因为这类场景天然适合多轮反馈

代码任务很少是一轮就结束的：

1. 先写代码
2. 再运行
3. 再看报错
4. 再修改

这和 AutoGen 的消息往返模式非常贴合。

### 4.2 一个最小“写代码 -> 运行 -> 反馈”示意

```python
conversation = [
    {"from": "planner", "to": "coder", "content": "请写一个折扣计算函数"},
    {"from": "coder", "to": "executor", "content": "def discount(price): return price * 0.7"},
    {"from": "executor", "to": "reviewer", "content": "运行结果：discount(100)=70"},
    {"from": "reviewer", "to": "coder", "content": "请补充非法输入处理"}
]

for turn in conversation:
    print(turn)
```

这个例子已经很像很多 AutoGen 教学场景里的工作流骨架了。

---

## 五、AutoGen 的真正优势是什么？

### 5.1 对“多角色来回协作”的表达很自然

它特别擅长表达：

- `planner <-> worker`
- `writer <-> reviewer`
- `coder <-> executor <-> critic`

这种多轮反馈关系。

### 5.2 对原型和实验非常友好

因为你不一定要一开始就把状态图画得很完整。  
你可以先：

- 定义几个角色
- 让他们开始对话
- 再观察系统如何推进

这种方式在探索阶段很有价值。

---

## 六、但 AutoGen 风格的风险也要看清

### 6.1 消息轮数很容易失控

因为一旦系统主要靠消息往返推进，就容易出现：

- 说太多轮
- 重复讨论
- 明明任务已经够信息了还在继续说

### 6.2 角色边界容易漂

如果你不给每个角色明确边界，就可能出现：

- planner 开始写代码
- reviewer 开始做检索

结果角色分工越来越乱。

### 6.3 收敛条件必须非常明确

如果没有“什么时候结束”的规则，系统会非常容易越跑越长。

所以 AutoGen 的一条很重要的工程原则是：

> **对话可以自然，但终止条件一定要明确。**

---

## 七、它和 CrewAI、LangGraph 的差别到底在哪？

### 7.1 和 CrewAI 的差别

CrewAI 更强调：

- 角色
- 任务
- 团队

AutoGen 更强调：

- 角色之间消息如何往返

所以一个粗暴但很好记的区分是：

- CrewAI 更像“团队排班表”
- AutoGen 更像“团队聊天协作”

### 7.2 和 LangGraph 的差别

LangGraph 更强调：

- 显式状态
- 节点
- 条件边

AutoGen 更强调：

- 对话轮次
- 回合推进

所以 AutoGen 在表达“像聊天一样推进任务”的系统时，会更自然。

---

## 八、什么时候值得考虑 AutoGen？

特别适合：

- 多轮协商型任务
- 代码生成 + 运行 + 反馈
- 写作 + 评审 + 修改
- 原型和实验探索

不一定特别适合：

- 需要严格状态机控制的生产系统
- 分支复杂但必须强可控的流程

也就是说，它更像：

> 一个非常适合表达“会话式协作”的框架。 

---

## 九、一个很实用的工程提醒

如果你真的要把 AutoGen 风格系统做深，最好尽早补上：

- trace
- 轮数上限
- 角色权限边界
- 失败回退

否则系统就会很容易从：

- 看起来很聪明

滑向：

- 很会聊，但效率很低

---

## 小结

这一节最重要的不是记住 AutoGen 这个名字，而是理解：

> **它最擅长表达“多个角色通过对话轮流推进任务”的系统。**

当任务天然像群聊协作时，这种框架会很自然；  
但如果你需要强状态控制，就要格外小心它的轮次和收敛问题。

---

## 练习

1. 设计一个 `planner -> coder -> reviewer` 的 3 角色消息流。
2. 想一想：为什么 AutoGen 风格任务特别容易出现“聊太多轮”的问题？
3. 用自己的话解释：AutoGen 和 CrewAI 的核心区别是什么？
4. 如果你的任务需要强状态机控制，你还会优先选这种对话式抽象吗？为什么？
