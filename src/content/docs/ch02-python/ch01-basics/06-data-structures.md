---
title: "2.1.6 Data Structures"
description: "Master the four major data structures in Python: lists, tuples, dictionaries, and sets"
sidebar:
  order: 6
---
![Python Data Structures Comparison](/img/course/ch02-data-structures-comparison-en.webp)

## Section Overview

In this section, you will learn how to organize a group of data. Lists, tuples, dictionaries, and sets will appear throughout later topics such as web scraping, data analysis, machine learning sample processing, and parsing API responses. The key is to know what each structure is suitable for storing, and when to use which one.

## Learning Objectives

- Master creating lists and common list operations
- Understand the features and use cases of tuples
- Master key-value operations in dictionaries
- Learn about deduplication and set operations with sets
- Choose the right data structure based on the scenario

---

## Why Do We Need Data Structures?

So far, the variables you have learned can only store one value at a time. But in real-world scenarios, you often need to handle **a collection of data**:

- 100 API latency measurements
- All the parameters of a model
- A user's personal information (name, age, email, ...)

Data structures are containers used to **organize and store multiple pieces of data**.

Python has 4 built-in data structures. A quick way to choose is:

| If you need... | Use this |
|---|---|
| Ordered data that changes often | **List** `[]` |
| Ordered data that should not change | **Tuple** `()` |
| Look up values by name or ID | **Dictionary** `{key: value}` |
| Remove duplicates or compare groups | **Set** `{item}` |

Remember the main properties:

- **List**: ordered, mutable, allows duplicates.
- **Tuple**: ordered, immutable, allows duplicates.
- **Dictionary**: ordered by insertion, mutable, keys cannot be duplicated.
- **Set**: unordered, mutable, duplicates are removed automatically.

---

## List — The Most Common Data Structure

A list is like a **stretchable cabinet**: you can put anything in it, and you can add, remove, or modify items at any time.

### Creating Lists

```python
# Create lists
latencies_ms = [120, 95, 240, 180, 310]
features = ["Login API", "RAG demo", "Chart view"]
mixed = [1, "hello", 3.14, True]   # Mixed types are allowed (but not recommended)
empty = []                         # Empty list

print(type(latencies_ms))  # <class 'list'>
print(len(latencies_ms))   # 5
```

### Accessing Elements (Indexing)

```python
service_queue = ["Login API", "Search API", "Worker", "Dashboard", "Docs site"]
#                 0            1             2         3            4
#                -5           -4            -3        -2           -1

print(service_queue[0])     # Login API (the first service)
print(service_queue[2])     # Worker (the third service)
print(service_queue[-1])    # Docs site (the last service)
print(service_queue[-2])    # Dashboard (the second to last service)
```

### Slicing

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(numbers[2:5])    # [2, 3, 4] (indices 2 to 4)
print(numbers[:3])     # [0, 1, 2] (first 3 items)
print(numbers[7:])     # [7, 8, 9] (from index 7 to the end)
print(numbers[::2])    # [0, 2, 4, 6, 8] (take every other item)
print(numbers[::-1])   # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] (reverse)
```

### Modifying Elements

```python
latencies_ms = [120, 95, 240, 180, 310]

# Modify a single element
latencies_ms[2] = 210
print(latencies_ms)  # [120, 95, 210, 180, 310]

# Modify multiple elements (via slicing)
latencies_ms[1:3] = [100, 180]
print(latencies_ms)  # [120, 100, 180, 180, 310]
```

### Adding Elements

```python
tasks = ["Build login form", "Write API tests"]

# Add to the end
tasks.append("Add error states")
print(tasks)  # ['Build login form', 'Write API tests', 'Add error states']

# Insert at a specific position
tasks.insert(1, "Review auth flow")
print(tasks)  # ['Build login form', 'Review auth flow', 'Write API tests', 'Add error states']

# Add multiple elements
tasks.extend(["Update README", "Record demo"])
print(tasks)  # ['Build login form', 'Review auth flow', 'Write API tests', 'Add error states', 'Update README', 'Record demo']
```

### Removing Elements

```python
tasks = ["Build login form", "Write API tests", "Add error states", "Review auth flow", "Record demo"]

# Remove by value (removes the first matching item)
tasks.remove("Add error states")
print(tasks)  # ['Build login form', 'Write API tests', 'Review auth flow', 'Record demo']

# Remove by index
deleted = tasks.pop(1)     # Remove the element at index 1 and return it
print(deleted)             # Write API tests
print(tasks)               # ['Build login form', 'Review auth flow', 'Record demo']

# Remove the last one
last = tasks.pop()
print(last)    # Record demo

# Remove by index (no return value needed)
del tasks[0]
print(tasks)  # ['Review auth flow']
```

### Common List Operations

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5]

# Sort
numbers.sort()
print(numbers)  # [1, 1, 2, 3, 4, 5, 5, 6, 9]

# Sort in descending order
numbers.sort(reverse=True)
print(numbers)  # [9, 6, 5, 5, 4, 3, 2, 1, 1]

# Sort without modifying the original list
original = [3, 1, 4, 1, 5]
sorted_list = sorted(original)
print(original)    # [3, 1, 4, 1, 5] (original list unchanged)
print(sorted_list) # [1, 1, 3, 4, 5]

# Reverse
numbers = [1, 2, 3, 4, 5]
numbers.reverse()
print(numbers)  # [5, 4, 3, 2, 1]

# Search
print(numbers.index(3))    # 2 (index of element 3)
print(numbers.count(5))    # 1 (number of times element 5 appears)
print(3 in numbers)        # True

# Statistics
latencies_ms = [120, 95, 240, 180, 310]
print(len(latencies_ms))    # 5
print(sum(latencies_ms))    # 945
print(max(latencies_ms))    # 310
print(min(latencies_ms))    # 95
print(sum(latencies_ms) / len(latencies_ms))  # 189.0 (average latency)
```

### List Comprehensions (Very Pythonic!)

List comprehensions are a concise way to create new lists:

```python
# Traditional way
squares = []
for i in range(1, 6):
    squares.append(i ** 2)
print(squares)  # [1, 4, 9, 16, 25]

# List comprehension (done in one line!)
squares = [i ** 2 for i in range(1, 6)]
print(squares)  # [1, 4, 9, 16, 25]

# List comprehension with a condition
even_squares = [i ** 2 for i in range(1, 11) if i % 2 == 0]
print(even_squares)  # [4, 16, 36, 64, 100]

# Real-world use: normalize feature slugs
raw_slugs = ["  Login API  ", "RAG DEMO", "  Chart View "]
clean_slugs = [slug.strip().lower().replace(" ", "-") for slug in raw_slugs]
print(clean_slugs)  # ['login-api', 'rag-demo', 'chart-view']
```

:::tip[List Comprehension Formula]
`[expression for variable in iterable if condition]`

In plain English: for each element that meets the condition, compute the expression and put it into a new list.
:::
---

## Tuple — An Immutable List

Tuples are almost the same as lists, except for one difference: **tuples cannot be modified after they are created**.

### Creating Tuples

```python
# Create with parentheses
point = (3, 4)
colors = ("red", "green", "blue")
single = (42,)          # When there is only one element, you must add a comma!
empty = ()

# In fact, the parentheses can be omitted
coordinates = 3, 4      # This is also a tuple
print(type(coordinates)) # <class 'tuple'>
```

### Tuple Operations

```python
colors = ("red", "green", "blue", "yellow", "purple")

# Access (same as lists)
print(colors[0])     # red
print(colors[-1])    # purple
print(colors[1:3])   # ('green', 'blue')

# Iterate
for color in colors:
    print(color)

# Search
print(len(colors))          # 5
print("red" in colors)      # True
print(colors.count("red"))   # 1
print(colors.index("blue"))  # 2

# But you cannot modify it!
# colors[0] = "black"  # Error! TypeError: 'tuple' object does not support item assignment
```

### Tuple Unpacking

```python
# Assign tuple values to multiple variables
point = (10, 20)
x, y = point
print(f"x={x}, y={y}")  # x=10, y=20

# When a function returns multiple values, it actually returns a tuple
def get_task_and_hours():
    return "Login API", 8

task, hours = get_task_and_hours()
print(f"{task}, {hours} hours")  # Login API, 8 hours

# Use * to collect extra values
first, *rest = [1, 2, 3, 4, 5]
print(first)  # 1
print(rest)   # [2, 3, 4, 5]
```

### When Should You Use a Tuple?

- When the data should not be modified (for example, coordinates, RGB color values)
- As dictionary keys (lists cannot be dictionary keys, but tuples can)
- When a function returns multiple values

---

## Dictionary — Key-Value Storage

A dictionary is one of the **most important data structures in Python**. It uses a **key** to look up a **value**, just like a real dictionary uses a word to find its definition.

### Creating Dictionaries

```python
# Create with curly braces
task = {
    "name": "Login API",
    "owner": "Mina",
    "status": "in_progress",
    "hours": [2, 3, 3]
}

# Empty dictionary
empty = {}

# Create with dict()
config = dict(learning_rate=0.001, epochs=100, batch_size=32)
print(config)  # {'learning_rate': 0.001, 'epochs': 100, 'batch_size': 32}

print(type(task))  # <class 'dict'>
```

### Accessing Values

```python
task = {"name": "Login API", "owner": "Mina", "status": "in_progress"}

# Method 1: Access with []
print(task["name"])   # Login API
# print(task["deadline"])  # Error! KeyError: 'deadline'

# Method 2: Access with .get() (safer)
print(task.get("owner"))    # Mina
print(task.get("deadline"))   # None (returns None if it does not exist, no error)
print(task.get("deadline", "not scheduled"))  # not scheduled (returns default value if it does not exist)
```

:::tip[Recommended: Use .get()]
When you are not sure whether a key exists, `.get()` is safer than `[]` and will not crash the program.
:::
### Adding and Modifying

```python
task = {"name": "Login API", "status": "todo"}

# Add new key-value pairs
task["owner"] = "Mina"
task["repo"] = "portfolio-api"

# Modify an existing value
task["status"] = "in_progress"

print(task)
# {'name': 'Login API', 'status': 'in_progress', 'owner': 'Mina', 'repo': 'portfolio-api'}

# Update in bulk
task.update({"status": "done", "hours": 8})
print(task)
```

### Deleting

```python
task = {"name": "Login API", "status": "done", "owner": "Mina"}

# Delete a specific key
del task["owner"]
print(task)  # {'name': 'Login API', 'status': 'done'}

# pop: delete and return the value
status = task.pop("status")
print(status)  # done
print(task)    # {'name': 'Login API'}
```

### Iterating Through a Dictionary

```python
task_hours = {"Login API": 8, "RAG demo": 12, "Chart view": 5}

# Iterate over keys
for task in task_hours:
    print(task)

# Iterate over values
for hours in task_hours.values():
    print(hours)

# Iterate over key-value pairs (most common)
for task, hours in task_hours.items():
    print(f"{task}: {hours} hours")

# Output:
# Login API: 8 hours
# RAG demo: 12 hours
# Chart view: 5 hours
```

### Dictionary Comprehensions

```python
# Create a mapping from numbers to squares
squares = {x: x**2 for x in range(1, 6)}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Filter a dictionary
task_hours = {"Login API": 8, "Bug fix": 3, "RAG demo": 12, "Docs": 2}
large_tasks = {name: hours for name, hours in task_hours.items() if hours >= 8}
print(large_tasks)  # {'Login API': 8, 'RAG demo': 12}
```

### Real Example: Counting Character Frequency

```python
text = "hello world"
char_count = {}

for char in text:
    if char in char_count:
        char_count[char] += 1
    else:
        char_count[char] = 1

print(char_count)
# {'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1}
```

---

## Set — The Deduplication Tool

A set is an **unordered collection of unique elements**.

### Creating Sets

```python
# Create with curly braces
task_tags = {"api", "ui", "testing", "api"}  # duplicates are removed automatically
print(task_tags)  # {'testing', 'ui', 'api'} (order may differ)

# Create from a list (deduplicates!)
modules = ["api", "api", "ui", "worker", "ui", "db"]
unique_modules = set(modules)
print(unique_modules)  # {'api', 'db', 'ui', 'worker'} (order may differ)

# Note: an empty set must be created with set(), not {}
empty_set = set()     # empty set
empty_dict = {}       # this is an empty dictionary!

print(type(task_tags))   # <class 'set'>
```

### Set Operations

```python
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}

# Intersection (items in both)
print(a & b)          # {4, 5}
print(a.intersection(b))

# Union (combined, duplicates removed)
print(a | b)          # {1, 2, 3, 4, 5, 6, 7, 8}
print(a.union(b))

# Difference (items in a but not in b)
print(a - b)          # {1, 2, 3}
print(a.difference(b))

# Symmetric difference (items unique to each set)
print(a ^ b)          # {1, 2, 3, 6, 7, 8}
```

### Real-World Use

```python
# Scenario: find tasks that touch both frontend and backend work
frontend_tasks = {"Login UI", "Chart view", "Settings page", "Theme switcher"}
backend_tasks = {"Login API", "Chart view", "Audit log", "Settings page"}

both = frontend_tasks & backend_tasks
print(f"Tasks touching both sides: {sorted(both)}")  # ['Chart view', 'Settings page']

only_frontend = frontend_tasks - backend_tasks
print(f"Frontend-only tasks: {sorted(only_frontend)}")  # ['Login UI', 'Theme switcher']

all_tasks = frontend_tasks | backend_tasks
print(f"All related tasks: {sorted(all_tasks)}")
```

---

## Data Structure Selection Guide

| Requirement | Recommended | Reason |
|------|------|------|
| Ordered collection that needs add/remove/modify | **List** | The most versatile container |
| Data should not be modified | **Tuple** | Immutable, safer |
| Find a value by key | **Dictionary** | O(1) lookup speed |
| Deduplication | **Set** | Automatically removes duplicates |
| Count occurrences | **Dictionary** | Keys are elements, values are counts |
| Check whether an element exists | **Set/Dictionary** | Much faster than a list |

---

## Hands-On Practice

### Exercise 1: API Latency Statistics

```python
latencies_ms = [120, 95, 240, 180, 310, 150, 88, 205, 260, 170]

# 1. Calculate the highest latency, lowest latency, and average latency
# 2. Find all latencies above 200 (use a list comprehension)
# 3. Sort the latencies from high to low
```

### Exercise 2: Service Owner Directory

Use a dictionary to implement a simple service owner directory:

```python
owners = {}

# 1. Add 3 services (service name -> owner email)
# 2. Look up a service owner's email
# 3. Modify a service owner's email
# 4. Delete one service
# 5. Print all service owners
```

### Exercise 3: Event Word Frequency Count

```python
text = "api error api timeout worker error api"

# Count how many times each event word appears
# Hint: first use split() to turn it into a list, then count with a dictionary
```

### Exercise 4: Remove Duplicates from a List (Keep Order)

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

# Remove duplicate elements while keeping the original order
# Expected output: [3, 1, 4, 5, 9, 2, 6]
# Hint: use a set to record elements that have already appeared
```

<details>
<summary>Reference implementation and walkthrough</summary>

1. The latency statistics are max `310`, min `88`, and average `181.8`. Latencies above `200` can be `[240, 310, 205, 260]`, and descending order starts with `[310, 260, 240, 205, ...]`.
2. The owner directory should add, update, and delete by key, for example `owners["Login API"] = "api-owner@example.com"`.
3. Event word frequency should count `api: 3`, `error: 2`, `timeout: 1`, and `worker: 1` for the sample text.
4. Ordered deduplication should produce `[3, 1, 4, 5, 9, 2, 6]`. Use a `seen` set plus a result list.
5. Choose a list for order, dictionary for lookup, set for membership or deduplication, and tuple for fixed records.

</details>

---

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
concept: variable, type, operator, input/output, branch, loop, structure, function, or module
code: smallest runnable Python snippet for the concept
output: printed value, type, branch result, loop trace, or returned value
failure_check: type mismatch, indentation, off-by-one, mutable data, or import path issue
Expected_output: code plus printed result that proves the concept works
```

## Summary

| Data Structure | Creation | Features | Common Use |
|---------|---------|------|---------|
| **List** | `[1, 2, 3]` | Ordered, mutable, allows duplicates | Store a group of similar data |
| **Tuple** | `(1, 2, 3)` | Ordered, immutable | Coordinates, return multiple values |
| **Dictionary** | `{"a": 1}` | Key-value pairs, keys cannot be duplicated | Configuration, mapping relationships |
| **Set** | `{1, 2, 3}` | Unordered, unique | Deduplication, set operations |

:::tip[Core Idea]
Choosing a data structure is like choosing a storage tool: a list is like a **drawer** (ordered items), a dictionary is like a **labeled cabinet** (find things by label), a set is like a **sieve** (automatically removes duplicates), and a tuple is like a **sealed bag** (once put in, it cannot be changed). Pick the right tool, and you get twice the result with half the effort.
:::
