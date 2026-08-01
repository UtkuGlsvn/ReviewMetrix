"""Loads the SEO / ASO / AIO content prompts from the prompts/ folder.

Each prompt is a Markdown file with a small front-matter block:

    ---
    title: ...
    description: ...
    ---
    <the prompt body>

Adding a prompt is just dropping a new .md file into the right category folder;
no code change is needed. Files are read once and cached for the process.
"""
import os
import re

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

# Folder slug -> display label, in tab order. AIO = AI Optimization (also known
# as Generative Engine Optimization): optimizing content to be surfaced and
# cited by AI answer engines.
CATEGORIES = [
    ('seo', 'SEO', 'Search Engine Optimization'),
    ('aso', 'ASO', 'App Store Optimization'),
    ('aio', 'AIO', 'AI Optimization / GEO'),
]

_FRONT_MATTER = re.compile(r'^---\s*\n(.*?)\n---\s*\n(.*)$', re.DOTALL)

_cache = None


def _parse(text):
    """Split a prompt file into (title, description, body)."""
    meta = {'title': '', 'description': ''}
    body = text.strip()

    match = _FRONT_MATTER.match(text)
    if match:
        front, body = match.group(1), match.group(2).strip()
        for line in front.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                meta[key.strip().lower()] = value.strip()

    return meta['title'], meta['description'], body


def load_prompts(force=False):
    """Return {slug: {'label', 'subtitle', 'items': [...]}} for every category.

    Each item is {'id', 'title', 'description', 'body'}. Cached after the first
    read; pass force=True to reload from disk.
    """
    global _cache
    if _cache is not None and not force:
        return _cache

    library = {}
    for slug, label, subtitle in CATEGORIES:
        folder = os.path.join(PROMPTS_DIR, slug)
        items = []
        if os.path.isdir(folder):
            for filename in sorted(os.listdir(folder)):
                if not filename.endswith('.md'):
                    continue
                path = os.path.join(folder, filename)
                try:
                    with open(path, encoding='utf-8') as fh:
                        title, description, body = _parse(fh.read())
                except OSError as e:
                    print(f"Could not read prompt {path}: {e}")
                    continue
                items.append({
                    'id': filename[:-3],
                    'title': title or filename[:-3],
                    'description': description,
                    'body': body,
                })
        library[slug] = {'label': label, 'subtitle': subtitle, 'items': items}

    _cache = library
    return library
