#!/usr/bin/env python3
import os, re, json
root = os.environ.get('COURSE_ROOT', '/Users/carl/Documents/AI-fullstack-course')
docs = os.path.join(root, 'docs')
sidebars = os.path.join(root, 'sidebars.js')
ids = set()
for dirpath, _, files in os.walk(docs):
    for f in files:
        if f.endswith(('.md', '.mdx')):
            rel = os.path.relpath(os.path.join(dirpath, f), docs)
            noext = re.sub(r'\.mdx?$', '', rel).replace(os.sep, '/')
            ids.add(noext)
            ids.add('/'.join(re.sub(r'^\d+-', '', part) for part in noext.split('/')))
            if noext.endswith('/index'):
                ids.add(noext[:-6])
                stripped = '/'.join(re.sub(r'^\d+-', '', part) for part in noext.split('/'))
                ids.add(stripped[:-6])
text = open(sidebars, encoding='utf-8').read()
refs = []
refs.extend(re.findall(r'"([A-Za-z0-9_./-]+)"', text))
refs.extend(re.findall(r"'([A-Za-z0-9_./-]+)'", text))
refs = [r for r in refs if '/' in r or r == 'index']
missing = []
for ref in refs:
    if ref not in ids:
        missing.append(ref)
print(json.dumps(sorted(set(missing)), ensure_ascii=False, indent=2))
print('missing_count', len(set(missing)))
