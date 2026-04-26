#!/usr/bin/env python3
import os
import re
import sys

root = os.environ.get('COURSE_ROOT') or os.getcwd()
docs = os.path.join(root, 'docs')
errors = []

for dirpath, _, files in os.walk(docs):
    for filename in files:
        if not filename.endswith(('.md', '.mdx')):
            continue
        path = os.path.join(dirpath, filename)
        with open(path, encoding='utf-8') as file:
            text = file.read()
        fences = re.findall(r'^```', text, flags=re.M)
        if len(fences) % 2:
            errors.append((os.path.relpath(path, root), len(fences)))

if errors:
    print('FAIL markdown fences: found unbalanced code fences')
    for path, count in errors[:20]:
        print(f'  - {path}: {count} fence markers')
    if len(errors) > 20:
        print(f'  ... and {len(errors) - 20} more')
    sys.exit(1)

print('PASS markdown fences: all code fences are balanced')
