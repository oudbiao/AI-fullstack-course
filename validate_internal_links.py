#!/usr/bin/env python3
import os, re, json
root = os.environ.get('COURSE_ROOT', '/Users/carl/Documents/AI-fullstack-course')
docs = os.path.join(root, 'docs')
existing = set()
for dirpath, _, files in os.walk(docs):
    for f in files:
        if f.endswith(('.md', '.mdx')):
            rel = os.path.relpath(os.path.join(dirpath, f), docs)
            noext = re.sub(r'\.mdx?$', '', rel)
            parts = []
            for part in noext.split(os.sep):
                parts.append(re.sub(r'^\d+-', '', part))
            existing.add('/' + '/'.join(parts))
            existing.add('/' + noext.replace(os.sep, '/'))
            if noext.endswith('/index'):
                existing.add('/' + noext[:-6].replace(os.sep, '/'))
                existing.add('/' + '/'.join(parts)[:-6])
missing = []
link_re = re.compile(r'\[[^\]]+\]\((/[^)\s#]+)(?:#[^)]+)?\)')
for dirpath, _, files in os.walk(docs):
    for f in files:
        if not f.endswith(('.md', '.mdx')):
            continue
        p = os.path.join(dirpath, f)
        text = open(p, encoding='utf-8').read()
        for m in link_re.finditer(text):
            target = m.group(1).rstrip('/')
            if target == '':
                continue
            if target not in existing:
                missing.append((os.path.relpath(p, root), target))
print(json.dumps(missing, ensure_ascii=False, indent=2))
print('missing_count', len(missing))
