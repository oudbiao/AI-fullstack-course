#!/usr/bin/env python3
import os
import re
import sys

root = os.environ.get('COURSE_ROOT') or os.getcwd()
docs = os.environ.get('COURSE_DOCS') or os.path.join(root, 'src', 'content', 'docs')
config = os.path.join(root, 'astro.config.mjs')
top_level_dirs = set()

for name in os.listdir(docs):
    path = os.path.join(docs, name)
    if os.path.isdir(path) and name not in {'zh-cn', 'ja'}:
        top_level_dirs.add(name)

with open(config, encoding='utf-8') as file:
    text = file.read()

refs = set(re.findall(r'directory:\s*"([A-Za-z0-9_./-]+)"', text))
missing = sorted(ref for ref in refs if not os.path.isdir(os.path.join(docs, ref)))
unlisted = sorted(top_level_dirs - refs)

if missing or unlisted:
    print('FAIL sidebars: found Starlight sidebar directory issues')
    for ref in missing[:20]:
        print(f'  - configured directory missing: {ref}')
    for ref in unlisted[:20]:
        print(f'  - top-level docs directory not listed: {ref}')
    sys.exit(1)

print('PASS sidebars: Starlight sidebar directories resolve')
