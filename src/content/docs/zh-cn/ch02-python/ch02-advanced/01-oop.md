---
title: "2.2.1 面向对象编程"
description: "掌握 Python 面向对象编程的核心概念"
sidebar:
  order: 1
---

# 2.2.1 面向对象编程

![类、对象、属性与方法关系图](/img/course/ch02-oop-class-object-map.webp)

## 本节定位

这一节介绍 Python 中组织复杂代码的一种重要方式。面向对象不是一开始就必须精通，但理解类、对象、属性和方法，能帮助你读懂后面的模型类、数据集类、API 服务类和第三方库源码。

## 学习目标

- 理解面向对象编程（OOP）的基本思想
- 掌握类和对象的定义与使用
- 理解属性和方法
- 掌握继承、封装的基本用法
- 了解常用的魔术方法

---

## 为什么需要面向对象？

假设你在为 AI 作品集应用开发一个小型项目追踪器，需要记录每个功能的负责人和工作记录：

```python
# 用变量和字典的方式
feature1_name = "登录 API"
feature1_owner = "Mina"
feature1_hours = [2, 5, 1]

feature2_name = "RAG 演示"
feature2_owner = "Kai"
feature2_hours = [3, 4, 2]

# 或者用字典
feature1 = {"name": "登录 API", "owner": "Mina", "hours": [2, 5, 1]}
feature2 = {"name": "RAG 演示", "owner": "Kai", "hours": [3, 4, 2]}

# 计算总工时的函数
def total_hours(feature):
    return sum(feature["hours"])
```

这样写有几个问题：
- 数据和操作是**分离的**（功能数据在字典里，计算函数在外面）
- 没有**约束**（谁都可以往字典里加奇怪的键，或者删掉必要的键）
- 当功能的属性越来越多时，代码会**越来越乱**

面向对象编程的思路是：**把数据和操作打包在一起，形成一个"对象"。**

```python
class FeatureTask:
    def __init__(self, name, owner, hours):
        self.name = name
        self.owner = owner
        self.hours = hours

    def total_hours(self):
        return sum(self.hours)

# 创建功能任务对象
task1 = FeatureTask("登录 API", "Mina", [2, 5, 1])
task2 = FeatureTask("RAG 演示", "Kai", [3, 4, 2])

# 数据和操作绑在一起，使用起来更自然
print(f"{task1.name} 总工时: {task1.total_hours():.1f}")
print(f"{task2.name} 总工时: {task2.total_hours():.1f}")
```

---

## 类和对象的基本概念

用一个生活中的类比：

- **类（Class）** = 蓝图/模板。比如"手机"是一个概念/类别
- **对象（Object/Instance）** = 用蓝图造出来的实体。比如"你手里的那台 iPhone 15"

```
类：FeatureTask（功能任务的模板）
    └── 属性：name, owner, hours
    └── 方法：total_hours(), is_over_budget()

对象（实例）：
    └── task1 = FeatureTask("登录 API", "Mina", [2, 5, 1])
    └── task2 = FeatureTask("RAG 演示", "Kai", [3, 4, 2])
```

---

## 定义类

### 最简单的类

```python
class Dog:
    """一只狗"""

    def __init__(self, name, breed):
        """初始化方法，创建对象时自动调用"""
        self.name = name      # 实例属性
        self.breed = breed    # 实例属性

    def bark(self):
        """方法：狗叫"""
        print(f"{self.name} 说: 汪汪汪！")

    def info(self):
        """方法：显示信息"""
        print(f"名字: {self.name}, 品种: {self.breed}")

# 创建对象（实例化）
my_dog = Dog("旺财", "金毛")
your_dog = Dog("小黑", "拉布拉多")

# 访问属性
print(my_dog.name)     # 旺财
print(your_dog.breed)  # 拉布拉多

# 调用方法
my_dog.bark()      # 旺财 说: 汪汪汪！
your_dog.info()    # 名字: 小黑, 品种: 拉布拉多
```

### 关键点解读

**1. `__init__` 方法（构造方法）**

`__init__` 在你创建对象时**自动调用**，用来初始化对象的属性。

```python
my_dog = Dog("旺财", "金毛")
# Python 自动做了这些事：
# 1. 创建一个新的 Dog 对象
# 2. 调用 __init__(self, "旺财", "金毛")
# 3. self.name = "旺财"
# 4. self.breed = "金毛"
# 5. 返回这个对象给 my_dog
```

**2. `self` 是什么？**

`self` 代表**对象自己**。当你调用 `my_dog.bark()` 时，Python 会自动把 `my_dog` 作为 `self` 传给 `bark` 方法。

```python
my_dog.bark()
# 等价于
Dog.bark(my_dog)
```

所以 `self.name` 就是"这个对象的 name"。

:::tip[self 的命名]
`self` 只是一个惯例（convention），你可以叫它 `this` 或任何名字，但**强烈建议**用 `self`——这是所有 Python 程序员的约定。
:::
---

## 属性和方法

### 实例属性 vs 类属性

```python
class FeatureTask:
    # 类属性：所有实例共享
    project = "AI 作品集"
    task_count = 0

    def __init__(self, name, owner):
        # 实例属性：每个实例独有
        self.name = name
        self.owner = owner
        FeatureTask.task_count += 1  # 每创建一个任务，计数加 1

t1 = FeatureTask("登录 API", "Mina")
t2 = FeatureTask("RAG 演示", "Kai")

# 类属性通过类名或实例都能访问
print(FeatureTask.project)     # AI 作品集
print(t1.project)              # AI 作品集
print(FeatureTask.task_count)  # 2

# 实例属性只属于各自的实例
print(t1.name)   # 登录 API
print(t2.owner)  # Kai
```

### 方法

```python
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        """计算面积"""
        return 3.14159 * self.radius ** 2

    def perimeter(self):
        """计算周长"""
        return 2 * 3.14159 * self.radius

    def scale(self, factor):
        """缩放半径"""
        self.radius *= factor  # 修改属性

c = Circle(5)
print(f"面积: {c.area():.2f}")       # 78.54
print(f"周长: {c.perimeter():.2f}")   # 31.42

c.scale(2)  # 半径变为 10
print(f"缩放后面积: {c.area():.2f}") # 314.16
```

---

## 魔术方法（双下划线方法）

Python 中以 `__` 开头和结尾的方法叫魔术方法（Magic Methods），它们让你的类可以像内置类型一样使用。

### `__str__`：定义 print 的输出

```python
class FeatureTask:
    def __init__(self, name, owner):
        self.name = name
        self.owner = owner

    def __str__(self):
        return f"FeatureTask({self.name}, owner={self.owner})"

task = FeatureTask("登录 API", "Mina")
print(task)  # FeatureTask(登录 API, owner=Mina)
# 如果没有 __str__，print 会输出 <__main__.FeatureTask object at 0x...>
```

### `__repr__`：定义开发者看到的表示

```python
class FeatureTask:
    def __init__(self, name, owner):
        self.name = name
        self.owner = owner

    def __repr__(self):
        return f"FeatureTask('{self.name}', '{self.owner}')"

task = FeatureTask("登录 API", "Mina")
print(repr(task))   # FeatureTask('登录 API', 'Mina')
# 在交互模式中直接输入 task 也会显示这个
```

### `__len__`：定义 len() 的行为

```python
class Playlist:
    def __init__(self, name, songs):
        self.name = name
        self.songs = songs

    def __len__(self):
        return len(self.songs)

my_playlist = Playlist("学习音乐", ["歌曲A", "歌曲B", "歌曲C"])
print(len(my_playlist))  # 3
```

### `__eq__`：定义 == 的行为

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

p1 = Point(3, 4)
p2 = Point(3, 4)
p3 = Point(1, 2)

print(p1 == p2)  # True
print(p1 == p3)  # False
```

---

## 继承

继承让你可以基于已有的类创建新类，**复用代码**。

### 基本继承

```python
# 父类（基类）
class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def speak(self):
        print(f"{self.name} 发出了声音")

    def info(self):
        print(f"{self.name}, {self.age}岁")

# 子类（派生类）
class Dog(Animal):
    def __init__(self, name, age, breed):
        super().__init__(name, age)  # 调用父类的 __init__
        self.breed = breed

    def speak(self):  # 重写父类的方法
        print(f"{self.name} 说: 汪汪汪！")

    def fetch(self):  # 子类独有的方法
        print(f"{self.name} 把球捡回来了！")

class Cat(Animal):
    def speak(self):  # 重写父类的方法
        print(f"{self.name} 说: 喵喵喵～")

# 使用
dog = Dog("旺财", 3, "金毛")
cat = Cat("咪咪", 2)

dog.info()     # 旺财, 3岁（继承自 Animal）
dog.speak()    # 旺财 说: 汪汪汪！（Dog 自己的实现）
dog.fetch()    # 旺财 把球捡回来了！（Dog 独有的）

cat.info()     # 咪咪, 2岁
cat.speak()    # 咪咪 说: 喵喵喵～
```

### super() 的作用

`super()` 用来调用父类的方法，最常见的用法是在 `__init__` 中：

```python
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)   # 让父类帮我初始化 name
        self.breed = breed       # 自己初始化 breed
```

### isinstance() 检查类型

```python
dog = Dog("旺财", 3, "金毛")

print(isinstance(dog, Dog))     # True —— 是 Dog
print(isinstance(dog, Animal))  # True —— 也是 Animal（因为继承）
print(isinstance(dog, Cat))     # False —— 不是 Cat
```

---

## 封装

封装的思想是：**隐藏内部细节，只暴露必要的接口。**

### 私有属性（约定）

Python 没有真正的私有属性，但有**命名约定**：

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self._balance = balance  # 单下划线：约定为"内部使用"

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            print(f"存入 {amount} 元，余额: {self._balance}")

    def withdraw(self, amount):
        if 0 < amount <= self._balance:
            self._balance -= amount
            print(f"取出 {amount} 元，余额: {self._balance}")
        else:
            print("余额不足！")

    def get_balance(self):
        return self._balance

account = BankAccount("作品集账户", 1000)
account.deposit(500)     # 存入 500 元，余额: 1500
account.withdraw(200)    # 取出 200 元，余额: 1300
print(account.get_balance())  # 1300

# 虽然技术上可以直接访问 _balance，但这不是推荐的做法
# print(account._balance)  # 能用，但不应该这么做
```

| 命名约定 | 含义 | 示例 |
|---------|------|------|
| `name` | 公开属性 | `self.name` |
| `_name` | 内部使用（约定） | `self._balance` |
| `__name` | 名称改写（强制隐藏） | `self.__secret` |

---

## 综合案例：AI 模型管理器

```python
class AIModel:
    """AI 模型基类"""
    model_count = 0

    def __init__(self, name, version="1.0"):
        self.name = name
        self.version = version
        self.is_trained = False
        self._accuracy = 0.0
        self._history = []
        AIModel.model_count += 1

    def train(self, epochs=10):
        """训练模型（模拟）"""
        import random
        print(f"开始训练 {self.name} v{self.version}...")
        for epoch in range(1, epochs + 1):
            acc = min(0.5 + epoch * 0.05 + random.uniform(-0.02, 0.02), 1.0)
            self._history.append(acc)
            if epoch % 5 == 0 or epoch == epochs:
                print(f"  Epoch {epoch}/{epochs} - Accuracy: {acc:.2%}")
        self._accuracy = self._history[-1]
        self.is_trained = True
        print(f"训练完成！最终准确率: {self._accuracy:.2%}")

    def predict(self, data):
        """预测"""
        if not self.is_trained:
            print("错误：模型还没有训练！")
            return None
        print(f"{self.name} 正在预测 {len(data)} 条数据...")
        return [f"预测结果_{i}" for i in range(len(data))]

    def __str__(self):
        status = "已训练" if self.is_trained else "未训练"
        return f"Model({self.name} v{self.version}, {status}, acc={self._accuracy:.2%})"


class ImageClassifier(AIModel):
    """图像分类模型"""
    def __init__(self, name, version="1.0", num_classes=10):
        super().__init__(name, version)
        self.num_classes = num_classes

    def predict(self, images):
        if not self.is_trained:
            print("错误：模型还没有训练！")
            return None
        print(f"正在对 {len(images)} 张图片进行分类（{self.num_classes} 个类别）...")
        import random
        return [random.randint(0, self.num_classes - 1) for _ in images]


# 使用
model = ImageClassifier("ResNet-50", "2.0", num_classes=100)
print(model)  # Model(ResNet-50 v2.0, 未训练, acc=0.00%)

model.train(epochs=10)
predictions = model.predict(["img1.jpg", "img2.jpg", "img3.jpg"])
print(f"预测类别: {predictions}")
print(model)
print(f"当前模型总数: {AIModel.model_count}")
```

---

## 动手练习

### 练习 1：图书管理

创建一个 `Book` 类：

```python
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
        self.current_page = 0

    def read(self, pages):
        self.current_page = min(self.current_page + pages, self.pages)

    def progress(self):
        return self.current_page / self.pages * 100

    def __str__(self):
        return f"{self.title}，作者：{self.author}，进度：{self.current_page}/{self.pages} 页"

# 测试
book = Book("Python 入门", "课程团队", 300)
book.read(50)
print(f"{book.progress():.1f}%")  # 16.7%
print(book)
```

### 练习 2：简单的购物车

```python
class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price

class ShoppingCart:
    def __init__(self):
        self.items = {}

    def add(self, product, quantity=1):
        self.items[product.name] = self.items.get(product.name, [product, 0])
        self.items[product.name][1] += quantity

    def remove(self, product_name):
        self.items.pop(product_name, None)

    def total(self):
        return sum(product.price * quantity for product, quantity in self.items.values())

    def __str__(self):
        lines = [f"{product.name} x {quantity}" for product, quantity in self.items.values()]
        return "\n".join(lines) or "购物车为空"

cart = ShoppingCart()
cart.add(Product("键盘", 199), 2)
cart.add(Product("鼠标", 99), 1)
print(cart)
print(f"总价：{cart.total()}")
```

### 练习 3：动物园

用继承实现：

```python
class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def speak(self):
        return "..."

class Dog(Animal):
    def speak(self):
        return "汪汪"

class Cat(Animal):
    def speak(self):
        return "喵喵"

class Duck(Animal):
    def speak(self):
        return "嘎嘎"

animals = [Dog("小黑", 3), Cat("咪咪", 2), Duck("小鸭", 1)]
for animal in animals:
    print(f"{animal.name}: {animal.speak()}")
```

<details>
<summary>参考实现与讲解</summary>

1. `Book` 应该把当前页数作为内部状态保存，并且用 `progress()` 计算阅读进度。读完 50/300 页后，示例输出应当接近 `16.7%`，对象字符串里应显示 `50/300`。
2. `ShoppingCart` 应该把商品对象和数量一起保存，这样 `total()` 才能正确相乘。`remove()` 在商品不存在时应保持安全，`__str__()` 在购物车为空时应返回清楚的提示。
3. `Animal` 提供共享字段和占位版 `speak()`，子类只覆盖自己的叫声。循环输出每个动物的名字和叫声，就能证明继承与多态是正常工作的。

</details>

---

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
模式：类、异常、文件 IO、函数式流水线、生成器或类型提示
代码产物：最小可运行示例和一个真实使用场景
输出：打印的对象状态、捕获的错误、保存的文件、yield 的值，或类型检查备注
失败检查：隐藏变异、吞掉异常、文件路径问题、懒迭代器混淆或误导性标注
期望产出：带调试说明的小型高级 Python 示例
```

## 小结

| 概念 | 说明 | 语法 |
|------|------|------|
| **类** | 对象的模板/蓝图 | `class MyClass:` |
| **对象** | 类的实例 | `obj = MyClass()` |
| **`__init__`** | 构造方法，初始化属性 | `def __init__(self):` |
| **self** | 指向当前对象自身 | `self.name = name` |
| **继承** | 子类复用父类的代码 | `class Dog(Animal):` |
| **super()** | 调用父类的方法 | `super().__init__()` |
| **魔术方法** | 自定义对象行为 | `__str__`, `__len__`, `__eq__` |

:::tip[核心理解]
面向对象的核心思想是**把数据和行为绑在一起**。类是模板，对象是实例。继承让你复用代码，封装让你隐藏细节。在 AI 开发中，你会经常看到类的使用——PyTorch 的模型定义就是一个类继承 `nn.Module`，训练循环中操作的都是对象。
:::