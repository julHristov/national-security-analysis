import json
from config import RAW_DIR, CLEAN_DIR, REPLACEMENTS_FILE
from pathlib import Path
import re


# ----------------------------
# Cleaning utility functions
# ----------------------------

# Removes page numbers
def remove_page_numbers(text: str) -> str:
    return re.sub(r"(Page\s*\d+|Стр\.?\d+)", "", text)


# Removes html tags
def remove_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


# Removes white spaces, tabs ect.
def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ------------------------
# Replacements from JSON
# ------------------------
# Loads symbols for replacements from JSON file in schema folder
# Returns dictionary key-symbols and their replacements
# Raises RuntimeError if the file is missing, non-valid or is not a dict.
def load_replacements():
    try:
        with open(REPLACEMENTS_FILE, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if not isinstance(data, dict):
                raise ValueError(f"Expects JSON to contain a dict, but it is '{type(data)}'")
            return data
    except FileNotFoundError:
        raise RuntimeError(f"❌ The file '{REPLACEMENTS_FILE}' not found.")
    except json.JSONDecodeError:
        raise RuntimeError(f"❌ The file '{REPLACEMENTS_FILE} does not contain a valid JSON file'")
    except ValueError as ve:
        raise RuntimeError(f"❌ A content error in '{REPLACEMENTS_FILE}': {ve}")


replacements = load_replacements()


# Replaces special symbols
def replace_special_symbols(text: str) -> str:
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# ----------------------
# Main cleaning function
# ----------------------
# Combines all cleaning functions.
def clean_text(text: str) -> str:
    text = remove_page_numbers(text)
    text = remove_html_tags(text)
    text = replace_special_symbols(text)
    text = normalize_whitespace(text)
    return text


# ----------------
# File processing
# ----------------
# Processing all the .txt files from RAW_DIR and writes them in CLEAN_DIR
def process_files():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in RAW_DIR.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                raw_text = file.read()

            cleaned = clean_text(raw_text)

            output_path = CLEAN_DIR / file_path.name
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(cleaned)

            print(f"✅ Cleaned file: {file_path.name}")

        except (FileNotFoundError, UnicodeError) as e:
            print(f"❌ Error in processing of {file_path.name}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in {file_path}")


# -----------
# Entry point
# -----------
if __name__ == "__main__":
    print("=== Start cleaning ===")
    process_files()
    print("=== Cleaning complete ===")
