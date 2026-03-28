"""Post-processing for ASR output — normalizes numbers, abbreviations, formatting."""
import re


# Number words → digits via word2number
try:
    from word2number import w2n
    _HAS_W2N = True
except ImportError:
    _HAS_W2N = False

# Multiplier suffixes that should attach to numbers
_SUFFIXES = {"k": "K", "m": "M", "b": "B", "x": "x"}

# Common number words (for detection)
_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "trillion",
}


def _is_number_word(w):
    # Strip punctuation and handle hyphens (e.g. "twenty-five")
    cleaned = w.lower().rstrip(".,!?;:")
    # Handle hyphenated numbers like "twenty-five"
    parts = cleaned.split("-")
    if all(p in _NUMBER_WORDS for p in parts if p):
        return True
    return cleaned in _NUMBER_WORDS or cleaned == "and"


def _expand_hyphens(words):
    """Split hyphenated number words: ['twenty-five'] → ['twenty', 'five']"""
    result = []
    for w in words:
        cleaned = w.lower().rstrip(".,!?;:")
        parts = cleaned.split("-")
        if len(parts) > 1 and all(p in _NUMBER_WORDS for p in parts if p):
            # Preserve trailing punctuation on the last part
            trailing = w[len(w.rstrip(".,!?;:")):]
            for k, p in enumerate(parts):
                if k == len(parts) - 1 and trailing:
                    result.append(p + trailing)
                else:
                    result.append(p)
        else:
            result.append(w)
    return result


def _convert_number_spans(text):
    """Find spans of number words and convert to digits."""
    if not _HAS_W2N:
        return text

    words = text.split()
    # Expand hyphenated numbers before processing
    words = _expand_hyphens(words)
    result = []
    i = 0

    while i < len(words):
        if _is_number_word(words[i]):
            span = []
            j = i
            while j < len(words) and _is_number_word(words[j]):
                span.append(words[j])
                j += 1

            # Preserve trailing punctuation
            last = span[-1]
            trailing_punct = ""
            stripped = last.rstrip(".,!?;:")
            if len(stripped) < len(last):
                trailing_punct = last[len(stripped):]
                span[-1] = stripped

            # Filter out standalone "and"
            num_words = [w for w in span if w.lower() != "and" or len(span) > 1]

            try:
                num = w2n.word_to_num(" ".join(num_words))
                num_str = f"{num:,}" if num >= 1000 else str(num)

                # Check if next word is a suffix (K, M, etc.)
                if j < len(words) and words[j].lower().rstrip(".,!?;:") in _SUFFIXES:
                    suffix_raw = words[j]
                    suffix_clean = _SUFFIXES[suffix_raw.lower().rstrip(".,!?;:")]
                    suffix_trailing = suffix_raw[len(suffix_raw.rstrip(".,!?;:")):]
                    num_str = str(num) + suffix_clean + suffix_trailing
                    trailing_punct = ""  # suffix has its own punctuation
                    j += 1

                result.append(num_str + trailing_punct)
            except (ValueError, IndexError):
                result.extend(span)
                if trailing_punct:
                    result[-1] += trailing_punct

            i = j
        else:
            result.append(words[i])
            i += 1

    return " ".join(result)


def _normalize_digit_multiplier(text):
    """Convert '3 thousand' → '3,000', '5 million' → '5,000,000' etc."""
    multipliers = [
        ("trillion", 1_000_000_000_000),
        ("billion", 1_000_000_000),
        ("million", 1_000_000),
        ("thousand", 1_000),
    ]
    for word, mult in multipliers:
        pattern = rf'(\d[\d,]*)\s+{word}\b'
        def replacer(m, mult=mult):
            num_str = m.group(1).replace(",", "")
            num = int(num_str) * mult
            return f"{num:,}"
        text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)
    return text


def _normalize_percentages(text):
    """Convert 'X percent' → 'X%'."""
    # Handle both word and digit forms
    text = re.sub(r'(\d+)\s+percent', r'\1%', text, flags=re.IGNORECASE)
    # Handle hyphenated: "Twenty-five percent" (already converted to "25 percent")
    return text


def _normalize_dollars(text):
    """Convert 'X dollars' → '$X'."""
    text = re.sub(r'(\d[\d,]*)\s+dollars?', r'$\1', text, flags=re.IGNORECASE)
    return text


def _capitalize_first(text):
    """Ensure first character is capitalized."""
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def format_transcription(text):
    """Apply all post-processing to ASR output."""
    if not text:
        return text

    text = _normalize_digit_multiplier(text)  # "3 thousand" → "3,000" (before word spans)
    text = _convert_number_spans(text)        # "twenty five" → "25"
    text = _normalize_percentages(text)
    text = _normalize_dollars(text)
    text = _capitalize_first(text)

    return text


if __name__ == "__main__":
    tests = [
        # Word numbers
        "Forget the part where it says that we have three hundred K credits.",
        "We need twenty five percent more bandwidth.",
        "Twenty-five percent of the users.",
        "It costs five hundred dollars per month.",
        "About two hundred and fifty K monthly active users.",
        # Digit + word combos (model sometimes outputs these)
        "3 thousand. 3 thousand.",
        "We have 5 million users.",
        "That's 2 billion dollars.",
        # Already formatted (should pass through)
        "300K.",
        "$500.",
        "1,200 items.",
    ]
    for t in tests:
        out = format_transcription(t)
        if t != out:
            print(f"IN:  {t}")
            print(f"OUT: {out}")
            print()
        else:
            print(f"OK:  {t}")
