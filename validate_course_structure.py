#!/usr/bin/env python3
import os
import re
import sys

root = os.environ.get('COURSE_ROOT') or os.getcwd()
docs = os.path.join(root, 'docs')
errors = []

stage_dirs = [
    name for name in os.listdir(docs)
    if re.fullmatch(r'stage[0-9]+[a-z]?', name)
    and os.path.isdir(os.path.join(docs, name))
]

required_index_sections = [
    '## 阶段交付物',
    '## 阶段通关标准',
]

required_task_sections = [
    '## 本阶段必须完成的任务',
    '## 推荐学习顺序',
    '## 阶段作品集交付物',
    '## 阶段通关问题',
]

for stage in sorted(stage_dirs):
    index_path = os.path.join(docs, stage, 'index.md')
    task_path = os.path.join(docs, stage, 'task-list.md')

    if not os.path.exists(index_path):
        errors.append(f'{stage}/index.md missing')
    else:
        text = open(index_path, encoding='utf-8').read()
        for section in required_index_sections:
            if section not in text:
                errors.append(f'{stage}/index.md missing section: {section}')

    if not os.path.exists(task_path):
        errors.append(f'{stage}/task-list.md missing')
    else:
        text = open(task_path, encoding='utf-8').read()
        for section in required_task_sections:
            if section not in text:
                errors.append(f'{stage}/task-list.md missing section: {section}')

project_roadmaps = []
for dirpath, _, files in os.walk(docs):
    if re.search(r'ch[0-9]+-projects$', dirpath) and '00-roadmap.md' in files:
        project_roadmaps.append(os.path.join(dirpath, '00-roadmap.md'))

for path in project_roadmaps:
    text = open(path, encoding='utf-8').read()
    if '## 项目交付物标准' not in text:
        errors.append(f'{os.path.relpath(path, docs)} missing section: ## 项目交付物标准')

if errors:
    print('FAIL course structure: found missing required structure')
    for error in errors[:40]:
        print(f'  - {error}')
    if len(errors) > 40:
        print(f'  ... and {len(errors) - 40} more')
    sys.exit(1)

print('PASS course structure: stage indexes, task lists, and project roadmaps have required sections')
