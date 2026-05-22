---
title: "2.1.5 流程控制"
description: "掌握条件判断和循环结构"
sidebar:
  order: 5
---

# 2.1.5 流程控制

![Python 流程控制执行路径图](/img/course/ch02-control-flow-paths.webp)

## 本节定位

这一节学习让程序“做判断”和“重复执行”。条件判断和循环是所有自动化脚本、数据处理流程和模型训练代码的骨架，掌握它们后，你的代码就不再只是从上到下执行。

## 学习目标

- 掌握 `if/elif/else` 条件判断
- 掌握 `for` 循环和 `while` 循环
- 学会使用 `break`、`continue` 控制循环
- 能编写包含嵌套逻辑的程序

---

## 什么是流程控制？

到目前为止，你写的代码都是**从上到下一行一行执行**的。但真实的程序需要做判断、需要重复执行——这就是流程控制。

想象你早上出门的决策过程：

```
如果 下雨了:
    带伞
否则如果 太阳很大:
    戴帽子
否则:
    直接出门
```

这就是**条件判断**。

再想象你背单词：

```
重复 100 次:
    看一个新单词
    记住它
```

这就是**循环**。

---

## 条件判断：if / elif / else

### 基本的 if

```python
failed_tests = 3

if failed_tests > 0:
    print("暂停发布，先检查失败的测试。")
```

**语法规则：**
1. `if` 后面跟条件表达式
2. 条件后面有一个**冒号 `:`**（很多新手会忘记）
3. 条件成立时执行的代码需要**缩进 4 个空格**

### if...else

```python
all_checks_passed = False

if all_checks_passed:
    print("构建可以部署")
    print("编写发布说明")
else:
    print("构建继续保持评审状态")
    print("先修复失败的检查")
```

### if...elif...else

`elif` 是 "else if" 的缩写，用来检查多个条件：

```python
latency_ms = 185

if latency_ms < 100:
    status = "很快"
elif latency_ms < 200:
    status = "健康"
elif latency_ms < 500:
    status = "偏慢"
else:
    status = "严重"

print(f"API 延迟: {latency_ms} ms，状态: {status}")
# 输出: API 延迟: 185 ms，状态: 健康
```

:::caution[注意执行顺序]
Python 从上到下依次检查每个条件，**一旦某个条件成立，就执行对应的代码块，然后跳过剩余所有的 elif 和 else**。所以条件的顺序很重要！

```python
latency_ms = 95

# 错误的顺序 ❌
if latency_ms < 500:
    print("需要关注")    # 95 < 500 成立，直接执行这个
elif latency_ms < 100:
    print("很快")        # 不会被执行！

# 正确的顺序 ✅：从严到宽
if latency_ms < 100:
    print("很快")        # 95 < 100 成立，执行这个
elif latency_ms < 500:
    print("需要关注")
```
:::
### 条件判断的简写

```python
# 三元表达式（一行搞定简单的 if-else）
latency_ms = 185
status = "在预算内" if latency_ms <= 200 else "需要关注"
print(status)  # 在预算内

# 等价于
if latency_ms <= 200:
    status = "在预算内"
else:
    status = "需要关注"
```

### 嵌套的 if

条件里面可以再套条件：

```python
has_approval = True
all_tests_passed = False

if has_approval:
    if all_tests_passed:
        print("部署这个构建")
    else:
        print("等待测试套件通过")
else:
    print("先申请发布审批")
```

不过嵌套太多层会让代码难以阅读，通常不建议超过 3 层。

---

## for 循环

`for` 循环用来**遍历**一个序列（列表、字符串、范围等）中的每个元素。

### 遍历列表

```python
services = ["登录 API", "搜索 API", "Worker", "仪表盘"]

for service in services:
    print(f"检查 {service}")

# 输出:
# 检查 登录 API
# 检查 搜索 API
# 检查 Worker
# 检查 仪表盘
```

理解方式：`for service in services` 的意思是"对于 services 中的每一个 service，执行下面的代码"。

### 遍历字符串

```python
word = "Python"

for char in word:
    print(char, end=" ")

# 输出: P y t h o n
```

### range() 函数

`range()` 生成一个数字序列，是 `for` 循环最常用的搭档：

```python
# range(5) 生成 0, 1, 2, 3, 4
for i in range(5):
    print(i, end=" ")
# 输出: 0 1 2 3 4

# range(start, stop) 从 start 到 stop-1
for i in range(1, 6):
    print(i, end=" ")
# 输出: 1 2 3 4 5

# range(start, stop, step) 带步长
for i in range(0, 10, 2):
    print(i, end=" ")
# 输出: 0 2 4 6 8

# 倒数
for i in range(5, 0, -1):
    print(i, end=" ")
# 输出: 5 4 3 2 1
```

### 实际案例：统计评审时间

```python
total_minutes = 0
for day in range(1, 6):
    total_minutes += 30
print(f"5 天评审总分钟数: {total_minutes}")  # 150
```

### enumerate()：同时获取索引和值

```python
tasks = ["设计登录表单", "构建 API 接口", "编写冒烟测试"]

# 普通写法
for i in range(len(tasks)):
    print(f"任务 {i+1}: {tasks[i]}")

# 更 Pythonic 的写法：用 enumerate
for i, task in enumerate(tasks):
    print(f"任务 {i+1}: {task}")

# 指定起始编号
for i, task in enumerate(tasks, start=1):
    print(f"任务 {i}: {task}")
```

---

## while 循环

`while` 循环在**条件成立时**持续执行，直到条件不成立为止。

### 基本用法

```python
count = 0

while count < 5:
    print(f"当前计数: {count}")
    count += 1   # 别忘了更新条件！

print("循环结束")

# 输出:
# 当前计数: 0
# 当前计数: 1
# 当前计数: 2
# 当前计数: 3
# 当前计数: 4
# 循环结束
```

:::caution[小心死循环！]
如果忘了更新条件变量，循环永远不会停止：

```python
# 死循环示例（不要运行！）
count = 0
while count < 5:
    print(count)
    # 忘了 count += 1，count 永远是 0，循环永不结束
```

如果不小心陷入死循环，按 `Ctrl+C` 强制终止程序。
:::
### while 的典型场景

`while` 适合**不确定循环次数**的情况：

```python
# 场景：等待后台任务完成
job_status = "queued"
poll_count = 0

while job_status != "finished":
    poll_count += 1
    print(f"第 {poll_count} 次轮询: {job_status}")

    if poll_count == 1:
        job_status = "running"
    elif poll_count == 2:
        job_status = "finished"

print(f"任务在 {poll_count} 次轮询后完成")
```

### for vs while 怎么选？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 遍历列表/字符串 | `for` | 天然适合 |
| 循环固定次数 | `for + range()` | 简洁明确 |
| 不确定循环次数 | `while` | 灵活控制 |
| 等待某个条件成立 | `while` | 自然直觉 |

**经验法则：能用 `for` 就用 `for`，它更安全（不会死循环）。**

---

## break 和 continue

### break：立即终止循环

```python
# 找到第一个慢请求就停下
latencies_ms = [120, 145, 310, 180, 260]

for latency_ms in latencies_ms:
    if latency_ms > 250:
        print(f"第一个慢请求: {latency_ms} ms")
        break
    print(f"{latency_ms} ms 仍在范围内，继续检查...")

# 输出:
# 120 ms 仍在范围内，继续检查...
# 145 ms 仍在范围内，继续检查...
# 第一个慢请求: 310 ms
```

### continue：跳过本次循环，继续下一次

```python
# 只打印慢请求，跳过健康请求
latencies_ms = [95, 210, 180, 260, 130]

for latency_ms in latencies_ms:
    if latency_ms <= 200:
        continue   # 跳过健康请求
    print(latency_ms, end=" ")

# 输出: 210 260
```

### break 和 continue 的区别

```python
# break：直接离开循环
for i in range(10):
    if i == 5:
        break       # 循环到 5 就整个停了
    print(i, end=" ")
# 输出: 0 1 2 3 4

# continue：跳过当前，继续下一个
for i in range(10):
    if i == 5:
        continue    # 跳过 5，继续 6, 7, 8, 9
    print(i, end=" ")
# 输出: 0 1 2 3 4 6 7 8 9
```

---

## 循环中的 else

Python 的循环有一个独特的 `else` 子句——当循环**正常结束**（不是被 `break` 终止）时执行：

```python
# 检查是否缺少必要评审
completed_checks = ["unit-test", "lint", "api-test"]
required_check = "security-review"

for check in completed_checks:
    if check == required_check:
        print(f"{required_check} 已完成")
        break
else:
    # 循环没有被 break 终止，说明没有找到必要检查
    print(f"{required_check} 缺失")

# 输出: security-review 缺失
```

---

## 嵌套循环

循环里面可以再套循环：

```python
# 打印模块/检查矩阵
modules = ["API", "UI", "DB"]
checks = ["lint", "test"]

for module in modules:
    for check in checks:
        print(f"{module}:{check}", end="\t")
    print()   # 每个模块结束后换行
```

输出：

```
API:lint	API:test
UI:lint	UI:test
DB:lint	DB:test
```

---

## 综合案例

### 案例 1：模拟 AI 模型训练过程

```python
import random

print("=== 开始训练模型 ===")
print(f"{'Epoch':<10}{'Loss':<15}{'Accuracy':<15}{'Status'}")
print("-" * 50)

loss = 2.5
accuracy = 0.10

for epoch in range(1, 21):
    # 模拟训练：损失逐渐下降，准确率逐渐上升
    loss *= random.uniform(0.85, 0.95)
    accuracy = min(accuracy + random.uniform(0.03, 0.06), 1.0)

    # 判断训练状态
    if accuracy >= 0.95:
        status = "✅ 达标"
    elif accuracy >= 0.80:
        status = "📈 良好"
    else:
        status = "🔄 训练中"

    print(f"{epoch:<10}{loss:<15.4f}{accuracy:<15.2%}{status}")

    # 如果准确率达到 98%，提前停止
    if accuracy >= 0.98:
        print(f"\n提前停止！在第 {epoch} 轮达到目标准确率")
        break
else:
    print(f"\n训练完成！最终准确率: {accuracy:.2%}")
```

### 案例 2：密码强度检查器

```python
password = input("请输入密码: ")

has_upper = False    # 是否有大写字母
has_lower = False    # 是否有小写字母
has_digit = False    # 是否有数字
has_special = False  # 是否有特殊字符

for char in password:
    if char.isupper():
        has_upper = True
    elif char.islower():
        has_lower = True
    elif char.isdigit():
        has_digit = True
    else:
        has_special = True

# 计算强度分数
score = 0
if len(password) >= 8:
    score += 1
if has_upper:
    score += 1
if has_lower:
    score += 1
if has_digit:
    score += 1
if has_special:
    score += 1

# 输出结果
print(f"\n密码强度: {'★' * score}{'☆' * (5 - score)} ({score}/5)")

if score <= 2:
    print("弱密码！建议加强")
elif score <= 4:
    print("中等强度")
else:
    print("强密码！")
```

---

## 动手练习

### 练习 1：发布检查标签

用一个小的发布检查标签器练习分支顺序：

打印 1 到 50 的样本编号，但是：
- 如果数字能被 15 整除，打印 "FullCheck"
- 如果数字能被 3 整除，打印 "Lint"
- 如果数字能被 5 整除，打印 "Test"
- 其他情况打印数字本身

```python
for i in range(1, 51):
    if i % 15 == 0:
        print("FullCheck")
    elif i % 3 == 0:
        print("Lint")
    elif i % 5 == 0:
        print("Test")
    else:
        print(i)
```

提示：先判断能否被 15 整除，再判断 3 和 5。

### 练习 2：延迟告警循环（限制样本数）

最多检查 7 个延迟样本，一旦超过阈值就停止。

```python
latencies_ms = [120, 180, 260, 140, 310, 190, 170]
threshold_ms = 250
max_samples = 7

for sample_no, latency_ms in enumerate(latencies_ms[:max_samples], start=1):
    print(f"样本 {sample_no}: {latency_ms} ms")

    if latency_ms <= threshold_ms:
        print("健康")
        continue

    print("告警：延迟超过阈值")
    break
else:
    print("已检查的样本都在阈值内。")
```

:::tip[如何减少调试挫败感]
先使用固定的小列表，再把慢请求移动到不同位置，分别测试健康路径、告警路径和全部通过路径。
:::
### 练习 3：打印部署进度条

用循环打印以下进度形状：

```
#
##
###
####
#####
```

然后试着打印倒计时进度条：

```
#####
####
###
##
#
```

### 练习 4：找出失败检查

打印所有状态不是 `"passed"` 的检查名称。

```python
checks = [
    ("lint", "passed"),
    ("unit-test", "failed"),
    ("api-test", "passed"),
    ("security-review", "failed"),
]

for check_name, status in checks:
    if status == "passed":
        continue
    print(f"{check_name}: {status}")
```

<details>
<summary>参考实现与讲解</summary>

1. 发布检查标签要先判断能否被 `15` 整除，否则 `15` 可能会提前输出成 `Lint` 或 `Test`。
2. 延迟列表先用固定数据测试：健康路径、告警路径，以及把慢请求移走后的全部通过路径。
3. 进度条可以用 `for n in range(1, 6): print("#" * n)` 打印，倒计时则用反向 `range`。
4. 失败检查筛选应对 `"passed"` 使用 `continue`，只打印 `failed` 或其他非通过状态。
5. 注意边界错误：`range(1, 51)` 包含 `50`，`range(1, 50)` 不包含 `50`。

</details>

---

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
概念：变量、类型、运算符、输入/输出、分支、循环、结构、函数或模块
代码：用于说明该概念的最小可运行 Python 代码片段
输出：打印值、类型、分支结果、循环 trace，或返回值
失败检查：类型不匹配、缩进错误、越界、可变数据或导入路径问题
期望产出：代码和打印结果，证明概念可行
```

## 小结

| 语法 | 用途 | 关键点 |
|------|------|--------|
| `if/elif/else` | 条件判断 | 条件从上到下检查，冒号和缩进不能忘 |
| `for...in` | 遍历序列 | 配合 `range()`、列表、字符串使用 |
| `while` | 条件循环 | 注意更新条件，避免死循环 |
| `break` | 终止循环 | 立即跳出整个循环 |
| `continue` | 跳过本次 | 跳过当前迭代，继续下一次 |
| `range()` | 生成数字序列 | `range(start, stop, step)` |

:::tip[核心理解]
流程控制是编程的**骨架**。变量是数据，运算符是操作，而流程控制决定了"在什么条件下做什么"和"做多少次"。学会了流程控制，你就能写出有"逻辑"的程序了。
:::