#!/usr/bin/env python3
import os, re, json
root = os.environ.get('COURSE_ROOT', '/Users/carl/Documents/AI-fullstack-course')
docs = os.path.join(root, 'docs')
errors = []
for dirpath, _, files in os.walk(docs):
    for f in files:
        if not f.endswith(('.md', '.mdx')):
            continue
        p = os.path.join(dirpath, f)
        text = open(p, encoding='utf-8').read()
        fences = re.findall(r'^```', text, flags=re.M)
        if len(fences) % 2:
            errors.append(('unbalanced_fence', os.path.relpath(p, root), len(fences)))
print(json.dumps(errors, ensure_ascii=False, indent=2))
print('error_count', len(errors))
