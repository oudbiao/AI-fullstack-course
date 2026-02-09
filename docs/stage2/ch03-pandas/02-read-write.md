---
title: "3.2 数据读写"
sidebar_position: 10
description: "掌握 CSV、Excel、JSON 等格式的数据读写"
---

# 数据读写

## 学习目标

- 掌握 CSV 文件的读取和写入
- 了解 Excel、JSON 等格式的读写
- 学会常用参数配置
- 了解大文件的分块读取技巧

---

## 读取 CSV 文件

CSV（Comma-Separated Values）是数据分析中最常用的文件格式。

### 基本读取

```python
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("titanic_sample.csv")
print(df.head())       # 看前 5 行
print(df.shape)        # 行列数
print(df.info())       # 列信息
```

就这一行，Pandas 自动完成了：
- 识别表头（第一行作为列名）
- 推断每列的数据类型
- 生成行索引（0, 1, 2, ...）

### 常用参数

```python
# 指定分隔符（有些文件用 tab 或分号分隔）
df = pd.read_csv("data.tsv", sep="\t")         # tab 分隔
df = pd.read_csv("data.csv", sep=";")          # 分号分隔

# 指定编码（中文文件常见问题）
df = pd.read_csv("chinese_data.csv", encoding="utf-8")
df = pd.read_csv("chinese_data.csv", encoding="gbk")     # 某些 Windows 导出的文件

# 没有表头的文件
df = pd.read_csv("no_header.csv", header=None)
df = pd.read_csv("no_header.csv", header=None, names=["col1", "col2", "col3"])

# 指定某列为索引
df = pd.read_csv("data.csv", index_col="id")
df = pd.read_csv("data.csv", index_col=0)      # 用第一列做索引

# 只读取部分列
df = pd.read_csv("data.csv", usecols=["Name", "Age", "Fare"])

# 只读取前 100 行
df = pd.read_csv("data.csv", nrows=100)

# 指定缺失值标记
df = pd.read_csv("data.csv", na_values=["NA", "N/A", "missing", "-"])

# 指定数据类型
df = pd.read_csv("data.csv", dtype={"Age": float, "Pclass": str})
```

### 参数速查表

| 参数 | 作用 | 示例 |
|------|------|------|
| `sep` | 分隔符 | `sep="\t"` |
| `encoding` | 编码 | `encoding="utf-8"` |
| `header` | 表头行号 | `header=None` |
| `names` | 自定义列名 | `names=["a","b"]` |
| `index_col` | 索引列 | `index_col="id"` |
| `usecols` | 读取部分列 | `usecols=["Name","Age"]` |
| `nrows` | 读取行数 | `nrows=100` |
| `skiprows` | 跳过行数 | `skiprows=5` |
| `na_values` | 缺失值标记 | `na_values=["NA","-"]` |
| `dtype` | 指定类型 | `dtype={"Age": float}` |
| `parse_dates` | 解析日期列 | `parse_dates=["date"]` |

---

## 写入 CSV 文件

```python
# 基本写入
df.to_csv("output.csv")

# 不保存索引（通常推荐）
df.to_csv("output.csv", index=False)

# 指定编码
df.to_csv("output.csv", index=False, encoding="utf-8-sig")  # Excel 友好的 UTF-8

# 指定分隔符
df.to_csv("output.tsv", index=False, sep="\t")
```

:::tip 中文 CSV 在 Excel 中乱码？
保存时使用 `encoding="utf-8-sig"`（带 BOM 头的 UTF-8），Excel 就能正确显示中文了。
:::

---

## 读写 Excel 文件

```python
# 读取 Excel（需要 openpyxl 库：pip install openpyxl）
df = pd.read_excel("data.xlsx")

# 读取指定工作表
df = pd.read_excel("data.xlsx", sheet_name="Sheet2")
df = pd.read_excel("data.xlsx", sheet_name=1)      # 按索引

# 读取所有工作表（返回字典）
all_sheets = pd.read_excel("data.xlsx", sheet_name=None)
for name, sheet_df in all_sheets.items():
    print(f"工作表 {name}: {sheet_df.shape}")

# 写入 Excel
df.to_excel("output.xlsx", index=False)

# 写入多个工作表
with pd.ExcelWriter("output.xlsx") as writer:
    df1.to_excel(writer, sheet_name="销售数据", index=False)
    df2.to_excel(writer, sheet_name="用户数据", index=False)
```

---

## 读写 JSON 文件

```python
# 读取 JSON
df = pd.read_json("data.json")

# 不同的 JSON 格式
# records 格式：[{"name": "张三", "age": 22}, {...}]
df = pd.read_json("data.json", orient="records")

# 写入 JSON
df.to_json("output.json", orient="records", force_ascii=False, indent=2)
# force_ascii=False：保持中文不转义
# indent=2：格式化输出
```

---

## 处理大文件：分块读取

当文件太大无法一次性载入内存时，可以分块读取：

```python
# 分块读取：每次读 1000 行
chunks = pd.read_csv("huge_file.csv", chunksize=1000)

# 逐块处理
results = []
for chunk in chunks:
    # 对每块做处理
    filtered = chunk[chunk["Age"] > 30]
    results.append(filtered)

# 合并所有块
df_final = pd.concat(results, ignore_index=True)
print(f"总共筛选出 {len(df_final)} 条记录")
```

```python
# 另一个常见用法：统计大文件的总行数
total_rows = sum(len(chunk) for chunk in pd.read_csv("huge_file.csv", chunksize=10000))
print(f"总行数: {total_rows}")
```

---

## 其他格式

```python
# 读取 HTML 表格（需要 lxml 或 html5lib）
# tables = pd.read_html("https://example.com/data.html")

# 读取剪贴板（从 Excel 复制后）
# df = pd.read_clipboard()

# 读取 Parquet（高效的列式存储格式，大数据常用）
# df = pd.read_parquet("data.parquet")

# 读取 SQL 数据库（第 4 章详细讲）
# import sqlite3
# conn = sqlite3.connect("database.db")
# df = pd.read_sql("SELECT * FROM users", conn)
```

---

## 实战：读取并初步查看数据

```python
import pandas as pd

# 假设我们有一份销售数据
# 如果没有文件，先创建一份示例数据
data = {
    "日期": ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19"],
    "商品": ["苹果", "牛奶", "面包", "苹果", "牛奶"],
    "数量": [50, 30, 45, 60, 25],
    "单价": [5.5, 8.0, 3.5, 5.5, 8.0],
    "销售额": [275.0, 240.0, 157.5, 330.0, 200.0]
}
df = pd.DataFrame(data)

# 保存为 CSV
df.to_csv("sales.csv", index=False)

# 重新读取
df = pd.read_csv("sales.csv")

# 标准的"初次见面"流程
print("=== 数据形状 ===")
print(df.shape)

print("\n=== 前几行 ===")
print(df.head())

print("\n=== 数据信息 ===")
print(df.info())

print("\n=== 统计摘要 ===")
print(df.describe())
```

---

## 小结

| 操作 | 读取 | 写入 |
|------|------|------|
| CSV | `pd.read_csv()` | `df.to_csv()` |
| Excel | `pd.read_excel()` | `df.to_excel()` |
| JSON | `pd.read_json()` | `df.to_json()` |
| SQL | `pd.read_sql()` | `df.to_sql()` |
| Parquet | `pd.read_parquet()` | `df.to_parquet()` |

:::tip 拿到数据的第一步
永远先运行这三行：
```python
print(df.shape)
df.info()
df.head()
```
:::

---

## 动手练习

### 练习 1：创建并读写 CSV

```python
# 1. 创建一个包含 10 个学生信息的 DataFrame（姓名、年龄、成绩）
# 2. 保存为 CSV 文件（不带索引）
# 3. 重新读取这个 CSV 文件
# 4. 验证读取前后数据一致
```

### 练习 2：读取真实数据

去 [Kaggle](https://www.kaggle.com/datasets) 下载一个感兴趣的小数据集（CSV 格式），用 `pd.read_csv()` 读取并完成"初次见面"流程。

### 练习 3：处理编码问题

```python
# 创建一份包含中文的数据，分别用 utf-8 和 gbk 编码保存
# 然后尝试用不同编码读取，观察乱码情况
```
