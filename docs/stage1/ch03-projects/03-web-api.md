---
title: "1.3 项目：Web API 开发"
sidebar_position: 3
description: "使用 FastAPI 构建你的第一个 Web API"
---

# 项目：Web API 开发

## 项目目标

- 理解什么是 API，为什么 AI 工程师需要会写 API
- 学会使用 FastAPI 框架构建 Web API
- 掌握 RESTful API 的基本设计原则
- 构建一个可以被其他程序调用的 AI 服务接口

---

## 为什么 AI 工程师需要会写 API？

你训练了一个很棒的 AI 模型——然后呢？

模型训练好只是第一步。要让别人**使用**你的模型，你需要把它**包装成一个 API 服务**：

```
你的 AI 模型  →  包装成 API  →  手机 App / 网站 / 其他程序 调用

具体例子：
- ChatGPT 模型 → 通过 API 提供服务 → 各种 App 调用
- 图像识别模型 → 通过 API → 用户上传图片获得识别结果
- 推荐算法 → 通过 API → 电商网站展示推荐商品
```

所以 **API 是连接 AI 模型和真实世界的桥梁**。

---

## 什么是 API？

**API（Application Programming Interface）** = 应用程序编程接口。

简单理解：API 就是一个**程序和程序之间的"对话窗口"**。

你去餐厅吃饭的过程就是一个 API 调用：

```
你（客户端）  →  向服务员（API）说"一碗牛肉面"（请求）
服务员       →  传给后厨（服务器）
后厨         →  做好面
服务员       →  把面端给你（响应）
```

你不需要知道后厨怎么做的面，你只需要知道**怎么点餐（发请求）和怎么接面（收响应）**。

### Web API 的核心概念

| 概念 | 说明 | 类比 |
|------|------|------|
| **URL（端点）** | API 的地址 | 餐厅地址 |
| **HTTP 方法** | 操作类型 | 点餐 / 退菜 / 加菜 |
| **请求体** | 发送的数据 | 你要的菜名 |
| **响应** | 返回的结果 | 端上来的菜 |
| **状态码** | 操作是否成功 | 200=成功, 404=没这道菜 |

### HTTP 方法

| 方法 | 用途 | 示例 |
|------|------|------|
| `GET` | 获取数据 | 获取任务列表 |
| `POST` | 创建数据 | 添加新任务 |
| `PUT` | 更新数据（整体） | 修改任务全部信息 |
| `DELETE` | 删除数据 | 删除一个任务 |

---

## 第一步：安装 FastAPI

```bash
pip install fastapi uvicorn
```

| 库 | 作用 |
|---|------|
| `fastapi` | Web 框架，用来编写 API |
| `uvicorn` | ASGI 服务器，用来运行 FastAPI 应用 |

---

## 第二步：Hello World API

创建文件 `main.py`：

```python
from fastapi import FastAPI

# 创建应用实例
app = FastAPI(title="我的第一个 API", version="1.0")

# 定义一个端点
@app.get("/")
def root():
    return {"message": "Hello, World!", "status": "running"}

@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"你好，{name}！", "name": name}
```

启动服务器：

```bash
uvicorn main:app --reload
```

- `main` = 文件名（`main.py`）
- `app` = FastAPI 实例名
- `--reload` = 代码修改后自动重启（开发时用）

打开浏览器访问：
- `http://127.0.0.1:8000` → 看到 Hello World
- `http://127.0.0.1:8000/hello/小明` → 看到个性化问候
- `http://127.0.0.1:8000/docs` → **自动生成的交互式 API 文档！**

:::tip FastAPI 的杀手锏：自动文档
访问 `/docs`，你会看到一个精美的交互式 API 文档（基于 Swagger UI）。你可以直接在浏览器中测试 API，不需要写任何前端代码。这是 FastAPI 最受欢迎的功能之一。
:::

---

## 第三步：构建任务管理 API

让我们把之前的命令行任务管理器改造成 Web API：

```python
"""
任务管理 API
运行: uvicorn main:app --reload
文档: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

# 创建应用
app = FastAPI(
    title="任务管理 API",
    description="一个简单的 RESTful 任务管理接口",
    version="1.0"
)

# ---------- 数据模型 ----------

class TaskCreate(BaseModel):
    """创建任务时的请求体"""
    title: str
    priority: str = "中"

class TaskUpdate(BaseModel):
    """更新任务时的请求体"""
    title: str | None = None
    priority: str | None = None
    done: bool | None = None

class Task(BaseModel):
    """任务的完整数据"""
    id: int
    title: str
    priority: str
    done: bool
    created_at: str

# ---------- 模拟数据库（内存存储） ----------

tasks_db: list[dict] = []
next_id: int = 1

# ---------- API 端点 ----------

@app.get("/")
def root():
    """API 首页"""
    return {
        "name": "任务管理 API",
        "version": "1.0",
        "endpoints": {
            "查看所有任务": "GET /tasks",
            "创建任务": "POST /tasks",
            "查看单个任务": "GET /tasks/{task_id}",
            "更新任务": "PUT /tasks/{task_id}",
            "删除任务": "DELETE /tasks/{task_id}",
            "API 文档": "GET /docs"
        }
    }


@app.get("/tasks")
def get_tasks(done: bool | None = None):
    """
    获取所有任务。

    可选参数:
    - done: 过滤已完成(true)或未完成(false)的任务
    """
    if done is not None:
        filtered = [t for t in tasks_db if t["done"] == done]
        return {"count": len(filtered), "tasks": filtered}
    return {"count": len(tasks_db), "tasks": tasks_db}


@app.post("/tasks", status_code=201)
def create_task(task: TaskCreate):
    """创建新任务"""
    global next_id

    new_task = {
        "id": next_id,
        "title": task.title,
        "priority": task.priority,
        "done": False,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    tasks_db.append(new_task)
    next_id += 1

    return {"message": "任务创建成功", "task": new_task}


@app.get("/tasks/{task_id}")
def get_task(task_id: int):
    """获取指定 ID 的任务"""
    for task in tasks_db:
        if task["id"] == task_id:
            return task

    # 找不到任务，返回 404
    raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")


@app.put("/tasks/{task_id}")
def update_task(task_id: int, task_update: TaskUpdate):
    """更新任务"""
    for task in tasks_db:
        if task["id"] == task_id:
            if task_update.title is not None:
                task["title"] = task_update.title
            if task_update.priority is not None:
                task["priority"] = task_update.priority
            if task_update.done is not None:
                task["done"] = task_update.done
            return {"message": "更新成功", "task": task}

    raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")


@app.delete("/tasks/{task_id}")
def delete_task(task_id: int):
    """删除任务"""
    for i, task in enumerate(tasks_db):
        if task["id"] == task_id:
            removed = tasks_db.pop(i)
            return {"message": "删除成功", "task": removed}

    raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")


@app.get("/stats")
def get_stats():
    """获取任务统计信息"""
    total = len(tasks_db)
    done = sum(1 for t in tasks_db if t["done"])
    return {
        "total": total,
        "done": done,
        "pending": total - done,
        "completion_rate": f"{done/total:.1%}" if total > 0 else "0%"
    }
```

### 运行和测试

```bash
# 启动服务器
uvicorn main:app --reload
```

然后打开 `http://127.0.0.1:8000/docs`，你可以在浏览器中直接测试所有 API。

也可以用命令行测试：

```bash
# 创建任务
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"title": "学习 FastAPI", "priority": "高"}'

# 查看所有任务
curl http://127.0.0.1:8000/tasks

# 完成任务
curl -X PUT http://127.0.0.1:8000/tasks/1 \
  -H "Content-Type: application/json" \
  -d '{"done": true}'

# 删除任务
curl -X DELETE http://127.0.0.1:8000/tasks/1
```

或者用 Python 的 `requests` 库：

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

# 创建任务
resp = requests.post(f"{BASE_URL}/tasks", json={"title": "学习 Python", "priority": "高"})
print(resp.json())

# 获取所有任务
resp = requests.get(f"{BASE_URL}/tasks")
print(resp.json())
```

---

## 理解 Pydantic 数据模型

FastAPI 使用 **Pydantic** 来验证请求数据——你定义好数据模型，FastAPI 自动帮你检查：

```python
from pydantic import BaseModel, Field

class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100, description="任务标题")
    priority: str = Field(default="中", description="优先级: 高/中/低")

# 如果用户发送了不合法的数据
# POST /tasks {"title": ""} → 自动返回 422 错误（标题太短）
# POST /tasks {} → 自动返回 422 错误（缺少 title）
# POST /tasks {"title": "OK"} → 成功，priority 使用默认值 "中"
```

你不需要手动写验证代码——Pydantic 和 FastAPI 帮你搞定了。

---

## 扩展挑战

### 挑战 1：添加文件持久化

现在数据存在内存里，服务器重启就没了。改成用 JSON 文件保存。

### 挑战 2：添加搜索功能

```python
@app.get("/tasks/search")
def search_tasks(keyword: str):
    """按关键词搜索任务"""
    pass
```

### 挑战 3：添加分页

当任务很多时，支持分页返回：

```
GET /tasks?page=1&size=10
```

### 挑战 4：接入 AI 模型

提前预习：创建一个 `/predict` 端点，接收文本输入，返回情感分析结果。

---

## 项目自查清单

- [ ] API 能正常启动和访问
- [ ] 实现了完整的 CRUD（增删改查）操作
- [ ] 使用了 Pydantic 做数据验证
- [ ] 有适当的错误处理（HTTPException）
- [ ] 自动生成的 API 文档（`/docs`）可以正常使用
- [ ] 能用 curl 或 requests 测试所有端点

:::tip 项目经验
FastAPI 是 AI 工程师最常用的 Web 框架之一。很多 AI 项目的部署方式都是：训练好模型 → 用 FastAPI 包装成 API → 部署到服务器。掌握 FastAPI，你就具备了**将 AI 模型产品化**的关键能力。而且 FastAPI 的自动文档功能，让前后端协作变得非常顺畅。
:::
