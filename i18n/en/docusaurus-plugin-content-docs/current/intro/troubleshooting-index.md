---
sidebar_position: 15
title: "Common Errors and Troubleshooting Index"
description: "Use an error-location tree to organize common issues in environments, Python, data, models, RAG, Agent, and deployment, helping learners quickly diagnose problems by symptom."
keywords: [troubleshooting, common errors, Debug, AI learning blockers, RAG errors, Agent errors]
---

# Common Errors and Troubleshooting Index

When learning AI, the things that most often cause learners to get stuck are usually not the concepts, but the environment, dependencies, data formats, APIs, model inputs and outputs, and engineering boundaries. When troubleshooting, don’t rush to copy the error message into a search engine. First identify which layer the problem belongs to: environment, code, data, model, retrieval, tool calling, deployment, or cache.

## One diagram to understand it: don’t skip steps when troubleshooting

```mermaid
flowchart LR
  A["Save the full error message"] --> B["Check the directory and environment"]
  B --> C["Reproduce with minimal input"]
  C --> D["Identify the failing layer"]
  D --> E["Fix it and write it back to the README"]
```

| If you can only do one thing first | What to do |
|---|---|
| Command error | First confirm the current directory and the command being run |
| Python error | First confirm the interpreter and dependency environment |
| RAG gives the wrong answer | First print the retrieval results; don’t rush to change the Prompt |
| Agent goes out of control | First limit the number of steps and save the trace |

## Overall troubleshooting flow

When you encounter an error, check in this order: first save the full error message and the command you ran, then confirm the current directory and runtime environment, then reproduce the issue with minimal input, then locate whether the problem is in dependencies, paths, data, parameters, or business logic, and finally write the solution back into your learning notes or project README.

Don’t look only at the last line of the error. Many Python, Node, Docusaurus, RAG, or Agent errors have key clues in the middle of the message, such as the actual path being read, the Python interpreter being used, the model name being requested, the tool parameter schema, or the retrieved document chunk.

## Environment and tool error-location tree

| Symptom you see | What to check first | Minimal reproduction | Recommended review |
| --- | --- | --- | --- |
| `command not found` or command cannot be found | Current terminal, PATH, whether dependencies are installed, whether you are in the project root | Run `pwd`, `which command-name`, and check package.json or requirements | Developer tools basics, environment setup |
| `ModuleNotFoundError` / `ImportError` | Whether the correct Python environment is activated, and whether pip installed into the same interpreter | Run `python -c "import package_name"` and `python -m pip show package_name` | Python environment, package management |
| `docusaurus: command not found` | Whether `npm install` was run, and whether npm scripts are being run from the project root | Run `npm run start` instead of running `docusaurus` directly | Environment setup, course maintenance process |
| Docusaurus page still shows old content | `.docusaurus`, `build`, browser cache, startup directory | Run `npm run clean` and then restart | This page, course maintenance process |
| Git push failed | Remote URL, permissions, network proxy, branch name, login status | Run `git remote -v` and `git branch --show-current` | Git and version control |

## Python and data error-location tree

| Symptom you see | What to check first | Minimal reproduction | Recommended review |
| --- | --- | --- | --- |
| File cannot be read | Current working directory, relative path, filename case, encoding | Run `pwd`, then test with a minimal script `open("path")` | Python file reading and writing |
| JSON parsing failed | Whether the file is valid JSON, whether it is JSONL with one JSON object per line, whether there are Chinese encoding issues | Copy a small piece of data and test `json.loads` separately | Python files and data structures |
| DataFrame column does not exist | Column name spaces, case, header row, read delimiter | Print `df.columns.tolist()` | Pandas data loading and cleaning |
| Garbled Chinese characters in charts | Font configuration, save format, runtime environment | Test with a minimal chart with a Chinese title | Data visualization |
| SQL query returns empty | Table name, filter conditions, connected database, whether the transaction was committed | Query the first 5 rows first, then add conditions step by step | Database basics |

## Machine learning and deep learning error-location tree

| Symptom you see | What to check first | Minimal reproduction | Recommended review |
| --- | --- | --- | --- |
| Model score is unusually high | Whether there is data leakage, whether training and test sets overlap, whether features include the answer | Retrain a baseline with only a small number of features | Feature engineering, model evaluation |
| Training score is high but validation is poor | Overfitting, data size, regularization, data augmentation, split strategy | Fix the random seed and compare training/validation curves | Deep learning training techniques |
| Model does not converge | Learning rate, data normalization, label format, whether loss matches the task | Overfit on a small dataset as a test | PyTorch training loops |
| PyTorch shape mismatch | Batch dimension, channel dimension, sequence length, loss input format | Print tensor shapes at each layer | PyTorch basics, CNN/Transformer |
| GPU does not work | Driver, CUDA, PyTorch version, device settings | Run `torch.cuda.is_available()` | Deep learning environment |

## LLM and RAG error-location tree

| Symptom you see | What to check first | Minimal reproduction | Recommended review |
| --- | --- | --- | --- |
| API call failed | API Key, model name, quota, network, request format, timeout | Send one shortest Prompt request by itself | LLM API practice |
| Output format is unstable | Whether the Prompt clearly defines the schema, and whether there is validation and retry logic | Make the model return only one minimal JSON object | Structured output |
| RAG answers the wrong question | query, chunk, embedding, top-k, rerank, document scope | Only print retrieval results; do not call the model | RAG basics and retrieval strategies |
| Cannot retrieve relevant documents | Whether documents were imported, whether metadata was preserved, whether chunks are too large or too small | Search directly with keywords from the original text | Document processing and vector databases |
| Citations are unreliable | Whether the answer is truly supported by the matched documents, whether citation paths are recorded | Compare answer sentences with the cited snippets | RAG evaluation |
| Cost suddenly increases | Context too long, too many retries, looping calls, model choice | Record tokens, request count, and time spent at each step | LLM engineering |

## Agent error-location tree

| Symptom you see | What to check first | Minimal reproduction | Recommended review |
| --- | --- | --- | --- |
| Agent loops forever | Stop conditions, maximum steps, whether tool outputs are clear, whether the goal is too broad | Limit it to 3 steps and print the plan at each step | Agent reasoning and planning |
| Tool parameter error | schema field names, types, required fields, whether examples are clear | Manually write one tool parameter call | Tool calling and Function Calling |
| Tool call succeeded but the result is unusable | Tool return format, error codes, empty-result handling | Call the tool with fixed input and print the raw return value | Agent tool strategy |
| Memory pollution | Whether error messages or temporary context are being saved long term | Clear memory and rerun the same task | Agent memory engineering |
| Risky operations go out of control | Whether there is human confirmation, permission limits, audit logs | Change the tool to dry-run mode for testing | Agent safety and Guardrails |
| Failure cannot be reproduced | Whether trace, tool input/output, state, and errors are recorded | Do a replay of one failed task | Agent observability |

## Deployment and engineering error-location tree

| Symptom you see | What to check first | Minimal reproduction | Recommended review |
| --- | --- | --- | --- |
| Works locally but not after deployment | Environment variables, file paths, dependency versions, startup command | Rerun in a clean environment according to the README | Docker and deployment |
| Port access failed | Whether the service has started, port mapping, firewall, host configuration | Curl the health-check endpoint locally | API design and deployment |
| Logs do not show the key error | Log level, exception handling, request ID | Manually trigger an error and inspect the logs | Logging and monitoring |
| Online results differ from local results | Model version, configuration, data index, cache | Print the key configuration and versions | Engineering best practices |

## Troubleshooting record template

Each time you encounter a problem, it is recommended to record the following items. Over time, this will become your own engineering experience library.

```md
## Issue title

### Error symptom
What command I ran and what error I saw.

### Full error message
Paste the key error message; do not paste only the last line.

### Reproduction steps
1. Which directory to enter
2. Which command to run
3. Which input or configuration to use

### Troubleshooting process
What environment, paths, dependencies, data, parameters, or logs I checked.

### Final solution
What exactly was changed, and why it worked.

### How to avoid it next time
What should be added to the README, scripts, validation commands, or test cases.
```

The goal of troubleshooting is not “to suppress this error as quickly as possible,” but to make the next similar error easier to locate. Every time you solve a real error, you should turn it into project documentation, validation scripts, log fields, or test cases.
