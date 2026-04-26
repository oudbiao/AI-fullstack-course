#!/usr/bin/env python3
import os
import re
import sys

root = os.environ.get('COURSE_ROOT') or os.getcwd()
docs = os.path.join(root, 'docs')
sidebars = os.path.join(root, 'sidebars.js')
ids = set()

for dirpath, _, files in os.walk(docs):
    for filename in files:
        if not filename.endswith(('.md', '.mdx')):
            continue
        rel = os.path.relpath(os.path.join(dirpath, filename), docs)
        noext = re.sub(r'\.mdx?$', '', rel).replace(os.sep, '/')
        ids.add(noext)
        stripped = '/'.join(re.sub(r'^\d+-', '', part) for part in noext.split('/'))
        ids.add(stripped)
        if noext.endswith('/index'):
            ids.add(noext[:-6])
            ids.add(stripped[:-6])

with open(sidebars, encoding='utf-8') as file:
    text = file.read()

refs = []
refs.extend(re.findall(r'"([A-Za-z0-9_./-]+)"', text))
refs.extend(re.findall(r"'([A-Za-z0-9_./-]+)'", text))
refs = [ref for ref in refs if '/' in ref or ref == 'index']
missing = sorted({ref for ref in refs if ref not in ids})

if missing:
    print('FAIL sidebars: found sidebar refs without matching docs')
    for ref in missing[:20]:
        print(f'  - {ref}')
    if len(missing) > 20:
        print(f'  ... and {len(missing) - 20} more')
    sys.exit(1)

print('PASS sidebars: all sidebar refs resolve')
