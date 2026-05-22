#!/usr/bin/env python3
import os
import re
import sys

root = os.environ.get('COURSE_ROOT') or os.getcwd()
docs = os.environ.get('COURSE_DOCS') or os.path.join(root, 'src', 'content', 'docs')
public = os.path.join(root, 'public')
existing = set()

for dirpath, _, files in os.walk(docs):
    for filename in files:
        if not filename.endswith(('.md', '.mdx')):
            continue
        rel = os.path.relpath(os.path.join(dirpath, filename), docs)
        noext = re.sub(r'\.mdx?$', '', rel)
        route = '/' + noext.replace(os.sep, '/')
        existing.add(route)
        if noext == 'index':
            existing.add('/')
        if noext.endswith('/index'):
            existing.add('/' + noext[:-6].replace(os.sep, '/'))

if os.path.isdir(public):
    for dirpath, _, files in os.walk(public):
        for filename in files:
            rel = os.path.relpath(os.path.join(dirpath, filename), public)
            existing.add('/' + rel.replace(os.sep, '/'))

missing = []
link_re = re.compile(r'\[[^\]]+\]\((/[^)\s#]+)(?:#[^)]+)?\)')

for dirpath, _, files in os.walk(docs):
    for filename in files:
        if not filename.endswith(('.md', '.mdx')):
            continue
        path = os.path.join(dirpath, filename)
        with open(path, encoding='utf-8') as file:
            text = file.read()
        for match in link_re.finditer(text):
            target = match.group(1).rstrip('/')
            if target and target not in existing:
                missing.append((os.path.relpath(path, root), target))

if missing:
    print('FAIL internal links: found missing absolute doc links')
    for source, target in missing[:20]:
        print(f'  - {source} -> {target}')
    if len(missing) > 20:
        print(f'  ... and {len(missing) - 20} more')
    sys.exit(1)

print('PASS internal links: all absolute doc links resolve')
