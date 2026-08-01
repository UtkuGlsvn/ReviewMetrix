"""Tests for the SEO / ASO / AIO prompt library and page."""
import os

import pytest

from reviewMetrix import prompt_library


# --------------------------------------------------------------------------
# Front-matter parsing
# --------------------------------------------------------------------------

def test_parse_extracts_title_description_and_body():
    text = "---\ntitle: My Title\ndescription: A short desc.\n---\nThe body line.\nSecond line."
    title, desc, body = prompt_library._parse(text)
    assert title == 'My Title'
    assert desc == 'A short desc.'
    assert body == 'The body line.\nSecond line.'


def test_parse_without_front_matter_returns_whole_text_as_body():
    title, desc, body = prompt_library._parse('Just a prompt with no header.')
    assert title == ''
    assert desc == ''
    assert body == 'Just a prompt with no header.'


def test_parse_body_keeps_placeholder_braces():
    """Curly-brace placeholders must survive parsing untouched."""
    text = '---\ntitle: T\n---\nUse {app name} and {keyword} here.'
    _, _, body = prompt_library._parse(text)
    assert '{app name}' in body
    assert '{keyword}' in body


# --------------------------------------------------------------------------
# Loading from disk
# --------------------------------------------------------------------------

def test_load_prompts_returns_all_three_categories():
    lib = prompt_library.load_prompts(force=True)
    assert list(lib.keys()) == ['seo', 'aso', 'aio']
    assert lib['seo']['label'] == 'SEO'
    assert lib['aso']['label'] == 'ASO'
    assert lib['aio']['label'] == 'AIO'


def test_every_category_has_prompts():
    lib = prompt_library.load_prompts(force=True)
    for slug, data in lib.items():
        assert data['items'], f'{slug} has no prompts'
        for item in data['items']:
            assert item['title']
            assert item['body']
            assert item['id']


def test_prompt_files_match_loaded_counts():
    """The loaded count should match the .md files on disk."""
    lib = prompt_library.load_prompts(force=True)
    for slug, data in lib.items():
        folder = os.path.join(prompt_library.PROMPTS_DIR, slug)
        on_disk = [f for f in os.listdir(folder) if f.endswith('.md')]
        assert len(data['items']) == len(on_disk)


def test_prompts_are_sorted_by_filename():
    lib = prompt_library.load_prompts(force=True)
    ids = [i['id'] for i in lib['aso']['items']]
    assert ids == sorted(ids)


def test_load_prompts_is_cached(monkeypatch):
    first = prompt_library.load_prompts(force=True)
    # A second call without force must return the very same cached object
    assert prompt_library.load_prompts() is first


# --------------------------------------------------------------------------
# Route
# --------------------------------------------------------------------------

def test_prompts_page_renders_with_tabs(client):
    resp = client.get('/prompts')
    assert resp.status_code == 200
    body = resp.data.decode()

    for label in ['SEO', 'ASO', 'AIO']:
        assert f'data-tab="{label.lower()}"' in body
    assert 'copyPrompt' in body
    # A known prompt title should appear
    assert 'iOS Keyword Field' in body


def test_prompts_page_escapes_placeholder_braces(client):
    """Placeholders render as literal text, not as broken Jinja."""
    resp = client.get('/prompts')
    body = resp.data.decode()
    assert '{app name}' in body


def test_index_links_to_prompts(client):
    body = client.get('/').data.decode()
    assert '/prompts' in body


def test_prompts_page_not_rate_limited(client, monkeypatch):
    from reviewMetrix.ratelimit import limiter
    monkeypatch.setattr(limiter, 'limit', 1)
    for _ in range(5):
        assert client.get('/prompts').status_code == 200
