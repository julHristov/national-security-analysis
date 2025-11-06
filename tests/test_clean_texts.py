import pytest
from scripts.clean_texts import (
    remove_page_numbers,
    remove_html_tags,
    replace_special_symbols,
    normalize_whitespace,
    clean_text,
    replacements
)


def test_remove_page_numbers():
    text = "Page 12 This is the introduction."
    cleaned = remove_page_numbers(text)
    assert "Page 12" not in cleaned, f"Expected 'Page 12' to be removed, but got {cleaned}"


def test_remove_html_tags():
    text = "<p>Hello<b>world!</b></p>"
    cleaned = remove_html_tags(text)
    assert "<" not in cleaned and ">" not in cleaned, (f"Expected '<' and '>' to be removed,"
                                                       f"but got {cleaned}")


def test_replace_special_symbols():
    text = "".join(replacements.keys())
    cleaned = replace_special_symbols(text)

    for bad_char in replacements.keys():
        assert bad_char in cleaned, f"Character {bad_char} not replaced"
