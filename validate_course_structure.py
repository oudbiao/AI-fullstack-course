#!/usr/bin/env python3
import os
import re
import sys

root = os.environ.get('COURSE_ROOT') or os.getcwd()
docs = os.path.join(root, 'docs')
errors = []

chapter_dirs = [
    name for name in os.listdir(docs)
    if re.fullmatch(r'ch[0-9]{2}-[a-z0-9-]+', name)
    and os.path.isdir(os.path.join(docs, name))
]

required_index_sections = [
    ('## Stage deliverables', '## Stage Deliverables', '## 阶段交付物'),
    (
        '## Stage completion criteria',
        '## Stage Completion Criteria',
        '## Stage completion standards',
        '## Stage Completion Standards',
        '## 阶段通关标准',
    ),
]

required_task_sections = [
    (
        '## Tasks you must complete in this stage',
        '## Tasks that must be completed in this stage',
        '## Required tasks for this stage',
        '## Required tasks for this phase',
        '## Required Tasks for This Phase',
        '## 本阶段必须完成的任务',
    ),
    (
        '## Recommended learning order',
        '## Recommended Learning Order',
        '## 推荐学习顺序',
    ),
    (
        '## Stage portfolio deliverables',
        '## Portfolio deliverables for this stage',
        '## Phase portfolio deliverables',
        '## Phase Portfolio Deliverables',
        '## 阶段作品集交付物',
    ),
    (
        '## Stage completion questions',
        '## Stage passing questions',
        '## Stage checkpoint questions',
        '## Stage pass questions',
        '## Phase completion questions',
        '## Phase Completion Questions',
        '## 阶段通关问题',
    ),
]

required_project_sections = [
    (
        '## Project Deliverable Standards',
        '## Project deliverable standards',
        '## Project Delivery Standards',
        '## Project Delivery Standard',
        '## 项目交付物标准',
    ),
]


def has_any_section(text, section_options):
    lower_text = text.lower()
    return any(section.lower() in lower_text for section in section_options)


def section_label(section_options):
    return ' / '.join(section_options)

for chapter in sorted(chapter_dirs):
    index_path = os.path.join(docs, chapter, 'index.md')
    task_path = os.path.join(docs, chapter, 'task-list.md')

    if not os.path.exists(index_path):
        errors.append(f'{chapter}/index.md missing')
    else:
        text = open(index_path, encoding='utf-8').read()
        for section in required_index_sections:
            if not has_any_section(text, section):
                errors.append(f'{chapter}/index.md missing section: {section_label(section)}')

    if not os.path.exists(task_path):
        errors.append(f'{chapter}/task-list.md missing')
    else:
        text = open(task_path, encoding='utf-8').read()
        for section in required_task_sections:
            if not has_any_section(text, section):
                errors.append(f'{chapter}/task-list.md missing section: {section_label(section)}')

project_roadmaps = []
for dirpath, _, files in os.walk(docs):
    if re.search(r'ch[0-9]+-projects$', dirpath) and '00-roadmap.md' in files:
        project_roadmaps.append(os.path.join(dirpath, '00-roadmap.md'))

for path in project_roadmaps:
    text = open(path, encoding='utf-8').read()
    for section in required_project_sections:
        if not has_any_section(text, section):
            errors.append(f'{os.path.relpath(path, docs)} missing section: {section_label(section)}')

if errors:
    print('FAIL course structure: found missing required structure')
    for error in errors[:40]:
        print(f'  - {error}')
    if len(errors) > 40:
        print(f'  ... and {len(errors) - 40} more')
    sys.exit(1)

print('PASS course structure: chapter indexes, task lists, and project roadmaps have required sections')
