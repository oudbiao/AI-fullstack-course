---
title: "1.3 File Operations and Serialization"
sidebar_position: 3
description: "Master file reading and writing, and data serialization"
---

# File Operations and Serialization

![File reading/writing and serialization flowchart](/img/course/ch02-file-io-serialization-flow-en.png)

## Where this section fits

This section shows how program data can be saved and loaded again later. File reading and writing, CSV, JSON, and serialization are the foundation for dataset processing, training logs, configuration files, and saving model results. They are also a key step from temporary code in memory to real projects.

## Learning objectives

- Master basic file reading and writing operations (`open`, `read`, `write`)
- Understand the role and benefits of the `with` statement
- Learn how to handle common data formats such as CSV and JSON
- Understand the concepts of serialization and deserialization

---

## Why do we need file operations?

So far, the data in your programs has lived in **memory** — once the program closes, the data is gone. But in real-world scenarios:

- Trained AI models need to be **saved** to a file so they can be loaded later
- Datasets are stored in CSV files and need to be **read** into the program
- Training logs need to be **written** to files for later analysis
- Configuration parameters are stored in JSON files and need to be **loaded** at startup

File operations let your program **persist data**.

---

## File I/O basics

### Open a file: `open()`

```python
# Basic syntax
file = open("file_path", "mode", encoding="encoding")
```

Common modes:

| Mode | Meaning | When the file does not exist |
|------|------|------------|
| `"r"` | Read (default) | Error |
| `"w"` | Write (overwrite) | Create automatically |
| `"a"` | Append (add to the end) | Create automatically |
| `"x"` | Create (error if file already exists) | Create automatically |
| `"rb"` | Read binary file | Error |
| `"wb"` | Write binary file | Create automatically |

### Write to a file

```python
# Method 1: Manually open and close (not recommended)
file = open("hello.txt", "w", encoding="utf-8")
file.write("Hello, world!\n")
file.write("I am learning Python file operations.\n")
file.close()  # Don't forget to close the file!

# Method 2: Use the with statement (recommended!)
with open("hello.txt", "w", encoding="utf-8") as file:
    file.write("Hello, world!\n")
    file.write("I am learning Python file operations.\n")
# When you leave the with block, the file is closed automatically, so no manual close() is needed
```

:::tip Why is the with statement recommended?
The `with` statement has two benefits:
1. **Automatically closes the file** — you do not need to worry about forgetting `close()`
2. **Exception-safe** — even if an error occurs, the file will still be closed properly

From now on, when writing file operations, **always use `with`**.
:::

### Read a file

```python
# Read the entire content
with open("hello.txt", "r", encoding="utf-8") as file:
    content = file.read()
    print(content)

# Read line by line
with open("hello.txt", "r", encoding="utf-8") as file:
    for line in file:
        print(line.strip())  # strip() removes the newline at the end of the line

# Read all lines into a list
with open("hello.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
    print(lines)  # ['Hello, world!\n', 'I am learning Python file operations.\n']
```

### Append content

```python
# "a" mode: append to the end of the file without overwriting existing content
with open("log.txt", "a", encoding="utf-8") as file:
    file.write("2026-02-09: Started learning\n")
    file.write("2026-02-09: Finished Chapter 1\n")
```

### Write multiple lines

```python
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]

with open("output.txt", "w", encoding="utf-8") as file:
    file.writelines(lines)  # Write a list of strings

# Or use print to write to a file
with open("output.txt", "w", encoding="utf-8") as file:
    print("Line 1", file=file)  # print can direct output to a file
    print("Line 2", file=file)
    print("Line 3", file=file)
```

---

## Real-world examples: working with different file formats

### CSV files

CSV (Comma-Separated Values) is one of the most common data file formats:

```python
import csv

# Write CSV
students = [
    ["Name", "Age", "Score"],
    ["Zhang San", 20, 85],
    ["Li Si", 21, 92],
    ["Wang Wu", 19, 78],
]

with open("students.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(students)

# Read CSV
with open("students.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header row
    print(f"Column names: {header}")

    for row in reader:
        name, age, score = row
        print(f"{name}, {age} years old, score: {score}")

# Read as dictionaries (more convenient)
with open("students.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(f"{row['Name']}'s score is {row['Score']}")
```

### JSON files

JSON is the most common data format in web development and APIs:

```python
import json

# Write JSON
config = {
    "model": "ResNet-50",
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
    "classes": ["cat", "dog", "bird"],
    "use_gpu": True
}

with open("config.json", "w", encoding="utf-8") as file:
    json.dump(config, file, ensure_ascii=False, indent=2)

# Read JSON
with open("config.json", "r", encoding="utf-8") as file:
    loaded_config = json.load(file)

print(f"Model: {loaded_config['model']}")
print(f"Learning rate: {loaded_config['learning_rate']}")
print(f"Classes: {loaded_config['classes']}")
```

Generated `config.json` content:

```json
{
  "model": "ResNet-50",
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 32,
  "classes": ["cat", "dog", "bird"],
  "use_gpu": true
}
```

:::info ensure_ascii=False
By default, `json.dump()` converts Chinese characters into Unicode escapes (such as `\u732b`). Adding `ensure_ascii=False` keeps the Chinese characters as they are, making the file easier to read.
:::

### Text log files

```python
from datetime import datetime

def log(message, filename="app.log"):
    """Write a log entry"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {message}\n")

# Use it
log("Program started")
log("Loaded dataset: train.csv")
log("Started training model")
log("Training complete, accuracy: 92.5%")
```

Generated log file:

```
[2026-02-09 14:30:01] Program started
[2026-02-09 14:30:02] Loaded dataset: train.csv
[2026-02-09 14:30:03] Started training model
[2026-02-09 14:35:15] Training complete, accuracy: 92.5%
```

---

## Path handling: `pathlib`

`pathlib` is the recommended way to handle paths in Python 3. It is more modern and easier to use than `os.path`:

```python
from pathlib import Path

# Create Path objects
data_dir = Path("data")
train_file = data_dir / "train" / "data.csv"  # Use / to join paths!
print(train_file)  # data/train/data.csv

# Check paths
print(train_file.exists())    # Whether the file exists
print(train_file.is_file())   # Whether it is a file
print(data_dir.is_dir())      # Whether it is a directory

# Get file information
path = Path("model.pth")
print(path.name)       # model.pth (file name)
print(path.stem)       # model (without extension)
print(path.suffix)     # .pth (extension)
print(path.parent)     # . (parent directory)

# Create directories
Path("output/results").mkdir(parents=True, exist_ok=True)

# List files in a directory
for file in Path(".").glob("*.py"):
    print(file)

# Recursively find all CSV files
for csv_file in Path("data").rglob("*.csv"):
    print(csv_file)

# Convenient file read/write methods
Path("note.txt").write_text("Hello!", encoding="utf-8")
content = Path("note.txt").read_text(encoding="utf-8")
print(content)  # Hello!
```

---

## Serialization: saving Python objects

### What is serialization?

**Serialization** means converting Python objects (lists, dictionaries, class instances, and so on) into a format that can be saved to a file. **Deserialization** means doing the reverse: restoring Python objects from a file.

| Format | Module | Readability | Speed | Safety | Use case |
|------|------|--------|------|--------|---------|
| JSON | `json` | ✅ Good | Medium | ✅ Safe | Configuration files, API data |
| CSV | `csv` | ✅ Good | Fast | ✅ Safe | Tabular data |
| pickle | `pickle` | ❌ Binary | Fast | ❌ Unsafe | Python objects |

### `pickle`: save any Python object

```python
import pickle

# Save Python object
data = {
    "scores": [85, 92, 78, 95],
    "names": ["Zhang San", "Li Si", "Wang Wu", "Zhao Liu"],
    "metadata": {"class": "Class A", "year": 2026}
}

with open("data.pkl", "wb") as file:  # Note: "wb" (binary write)
    pickle.dump(data, file)

# Load Python object
with open("data.pkl", "rb") as file:  # Note: "rb" (binary read)
    loaded_data = pickle.load(file)

print(loaded_data["names"])  # ['Zhang San', 'Li Si', 'Wang Wu', 'Zhao Liu']
```

:::caution pickle safety warning
**Never load a pickle file from an untrusted source!** pickle can execute arbitrary code, and a maliciously crafted pickle file can run dangerous operations on your computer. Only load pickle files created by yourself or from trusted sources.
:::

---

## Comprehensive example: student grade management system

```python
import json
from pathlib import Path
from datetime import datetime

class GradeBook:
    """Grade management system with file persistence"""

    def __init__(self, filename="gradebook.json"):
        self.filename = Path(filename)
        self.students = {}
        self.load()  # Load data at startup

    def load(self):
        """Load data from a file"""
        if self.filename.exists():
            with open(self.filename, "r", encoding="utf-8") as f:
                self.students = json.load(f)
            print(f"✅ Loaded data for {len(self.students)} students")
        else:
            print("📝 Creating a new gradebook")

    def save(self):
        """Save data to a file"""
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.students, f, ensure_ascii=False, indent=2)

    def add_score(self, name, subject, score):
        """Add a score"""
        if name not in self.students:
            self.students[name] = {}
        self.students[name][subject] = score
        self.save()
        print(f"✅ Saved {name}'s {subject} score ({score})")

    def get_report(self, name):
        """Get a student report"""
        if name not in self.students:
            print(f"❌ Student not found: {name}")
            return

        scores = self.students[name]
        print(f"\n{'='*30}")
        print(f"  {name}'s Grade Report")
        print(f"{'='*30}")
        for subject, score in scores.items():
            print(f"  {subject}: {score}")
        avg = sum(scores.values()) / len(scores)
        print(f"{'─'*30}")
        print(f"  Average score: {avg:.1f}")
        print(f"{'='*30}")

    def export_csv(self, filename="grades.csv"):
        """Export as CSV"""
        import csv
        subjects = set()
        for scores in self.students.values():
            subjects.update(scores.keys())
        subjects = sorted(subjects)

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Name"] + subjects)
            for name, scores in self.students.items():
                row = [name] + [scores.get(s, "") for s in subjects]
                writer.writerow(row)
        print(f"✅ Exported to {filename}")

# Use it
gb = GradeBook()
gb.add_score("Zhang San", "Math", 85)
gb.add_score("Zhang San", "English", 92)
gb.add_score("Zhang San", "Python", 95)
gb.add_score("Li Si", "Math", 78)
gb.add_score("Li Si", "English", 88)
gb.get_report("Zhang San")
gb.export_csv()
```

---

## Hands-on exercises

### Exercise 1: File statistics tool

```python
def file_stats(filename):
    """
    Count file information:
    - Total number of lines
    - Total number of characters (excluding newline characters)
    - Total number of words
    - Longest line and its line number
    """
    pass

# Create a test file and analyze it
```

### Exercise 2: Diary app

Write a simple diary app:
- Support writing new diary entries (automatically add a timestamp)
- Support viewing all diary entries
- Store diary entries in a text file so data is not lost after the program closes

### Exercise 3: Configuration file manager

```python
def load_config(filename="config.json"):
    """Load a configuration file, or create a default config if it does not exist"""
    pass

def save_config(config, filename="config.json"):
    """Save the config to a file"""
    pass

def update_config(key, value, filename="config.json"):
    """Update a configuration item"""
    pass
```

---

## Summary

| Operation | Code | Notes |
|------|------|------|
| Write file | `with open("f.txt", "w") as f:` | `"w"` overwrites, `"a"` appends |
| Read file | `with open("f.txt", "r") as f:` | `.read()`, `.readlines()` |
| Write JSON | `json.dump(data, file)` | Dictionary → JSON file |
| Read JSON | `json.load(file)` | JSON file → Dictionary |
| Write CSV | `csv.writer(file).writerow()` | List → CSV row |
| Read CSV | `csv.reader(file)` | CSV row → List |
| Path handling | `Path("data") / "file.txt"` | `pathlib` is recommended |

:::tip Core idea
File operations give your program a "memory" — data can persist across program runs. In AI development, you will often read and write many kinds of files: datasets (CSV), configurations (JSON/YAML), model weights (`.pth`), and training logs (`.log`). Mastering file operations is a fundamental skill for becoming a developer.
:::
