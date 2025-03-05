import re
from num2words import num2words
from tqdm import tqdm

# Preprocessors adjusted for Swedish

def convert_digits_to_words(data):
    digit_pattern = re.compile(r'(\d+)')
    def replace_with_words(match):
        return num2words(int(match.group()), lang='sv')
    result = digit_pattern.sub(lambda match: ' ' + replace_with_words(match) + ' ', data)
    data = re.sub(r'\s+', ' ', result).strip()
    return data

def remove_special_chars(data):
    # Remove control characters and unwanted symbols,
    # but keep Swedish characters like å, ä, ö intact.
    chars = [
        r'\x13', r'\x10', r'\x01', r'\x02', r'\x10', r'\x12', r'\x15', r'\x1b', r'\x18',
        r'\x1b', r'ø', r'½', r'¹', r'ß', r'Ô', r'¥', r'ë', r'î', r'û', r'ò',
        # Removed r'ö' and r'å' to preserve them for Swedish.
        r'\|n', r'§', r'll', r'¢', r'©', r'£', r'~', r'º', r"\(", r"\)", r"\̀", r"–",
        r"~", r"”", r"»", r"“", r"«", r"˙", r"\\", r"—", r"@", r"´",
    ]
    chars_to_delete = '|'.join(chars)
    data["text"] = re.sub(chars_to_delete, '', data["text"])
    return data

def replace_patterns(data):
    # Remove Portuguese-specific substitutions and update & to "och" (Swedish for "and")
    substitutions = {
        r"'([A-Z])": r' \1',
        r"'([a-z])": r'\1',
        r'&': ' och ',
        r"I’m": 'I am',
        r"'": '',
        r'!': '.',
        r'-': ' ',
        r' {2,}': ' ',
        r' ,': ',',
    }
    for pattern, replacement in substitutions.items():
        data["text"] = re.sub(pattern, replacement, data["text"])
    return data

def convert_to_digit(data):
    data["text"] = convert_digits_to_words(data["text"])
    return data

def remove_double_space(data):
    data["text"] = re.sub(r'\s{2,}', ' ', data["text"])
    return data

def remove_space_punctuation(data):
    data["text"] = re.sub(r'\s+([.,:?!])', r'\1', data["text"])
    return data

def add_uppercase_and_final_punctuation(data):
    # Uppercase first letter and ensure a period or question mark at the end.
    data["text"] = re.sub(r'^(.)', lambda match: match.group(1).upper(), data["text"])
    if not re.search(r'[.?]$', data["text"]):
        data["text"] += '.'
    return data

# List of pre-processing functions for Swedish
PREPROCESSORS = [
    remove_special_chars,
    replace_patterns,
    convert_to_digit,
    add_uppercase_and_final_punctuation,
    remove_double_space,
    remove_space_punctuation,
]

def apply_preprocessors(manifest, preprocessors=PREPROCESSORS):
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])
    print("Finished processing manifest!")
    return manifest
