"""Post-processing for ASR output — normalizes numbers, abbreviations, formatting."""
import re


# Number words → digits via word2number
try:
    from word2number import w2n
    _HAS_W2N = True
except ImportError:
    _HAS_W2N = False


# Word segmentation for merged tokens like "sidsleeping" → "side sleeping"
try:
    import wordninja
    _HAS_WORDNINJA = True
except ImportError:
    _HAS_WORDNINJA = False


def _wordninja_lm():
    """Return wordninja's internal word-frequency vocabulary (~126k words
    from Google N-grams), or None if wordninja isn't installed. This is
    more modern than /usr/share/dict/words (Webster's 2nd) and covers
    contemporary compounds like 'website', 'smartphone', etc."""
    if not _HAS_WORDNINJA:
        return None
    return wordninja.DEFAULT_LANGUAGE_MODEL._wordcost


# Modern tech/compound words that wordninja's frequency list misses. Without
# these, splits like "workflow" → "work flow" happen. Extend as false
# positives are discovered.
_TECH_COMPOUNDS = frozenset({
    "workflow", "workflows", "keyword", "keywords", "keychain", "keychains",
    "checkbox", "checkboxes", "backend", "frontend", "runtime", "runtimes",
    "codebase", "codebases", "filesystem", "filesystems", "roadmap",
    "pipelines", "datastore", "datasource", "metadata",
    "hotkey", "hotkeys", "webhook", "webhooks", "subprocess", "subprocesses",
    "screenshot", "screenshots", "livestream", "livestreams",
    "changelog", "changelogs", "fallback", "fallbacks", "callback", "callbacks",
    "eslint", "openai", "anthropic", "claude",
})


# Common 1-2 char English words that appear as split pieces but aren't in
# wordninja's frequency list. Allows "havea"→"have a", "totear"→"to tear",
# "upso"→"up so", etc.
_SHORT_VALID_PIECES = frozenset({
    "a", "i",
    "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it",
    "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we",
})


# Contractions, fillers, informal forms. These don't show up in the LM as
# single tokens (apostrophes split them) so we whitelist them explicitly
# to keep "she's", "don't" from being mangled.
_SPEECH_WORDS = frozenset({
    "i'm", "i've", "i'll", "i'd",
    "you're", "you've", "you'll", "you'd",
    "he's", "she's", "it's", "that's", "there's", "here's", "what's",
    "we're", "we've", "we'll", "we'd",
    "they're", "they've", "they'll", "they'd",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
    "shouldn't", "mustn't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't",
    "let's", "who's", "y'all",
    "gonna", "wanna", "gotta", "kinda", "sorta", "lemme", "gimme", "dunno",
    "uh", "um", "hmm", "huh", "yeah", "yep", "nope", "ok", "okay",
    "ya", "nah",
})


def _is_known_word(word):
    """Does this word appear in any of our dictionaries?"""
    if word in _TECH_COMPOUNDS or word in _SPEECH_WORDS:
        return True
    lm = _wordninja_lm()
    if lm is not None and word in lm:
        return True
    return False


def _fix_punctuation_spacing(text):
    """Insert missing space after sentence / clause punctuation when the next
    char is a letter. Catches the common ASR output "shift.should" → "shift. should"
    and "um,she's" → "um, she's". Careful not to break decimals (3.14) or
    ellipses (covered because next char is digit / dot)."""
    # After . , ! ? ; : — add space if directly followed by a letter
    return re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)


def _collapse_duplicate_punctuation(text):
    """Collapse ASR punctuation stutter.

    Granite Speech already emits punctuation, and sometimes produces paired
    marks such as ",,", ",.", "?.", or "..". The rest of the formatter assumes
    punctuation is single-mark, so normalize those pairs before number/brand
    post-processing.
    """
    # Exact repeats: ",," → ",", "??" → "?", ".." / "..." → "."
    text = re.sub(r"([,;:!?])\1+", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)

    # A comma immediately followed by a sentence terminator is really a
    # sentence terminator: "Okay,." → "Okay.", "top,?" → "top?"
    text = re.sub(r",\s*([.!?])", r"\1", text)

    # If a sentence terminator is immediately followed by more punctuation,
    # keep the terminator: "what?." → "what?", "top.," → "top."
    text = re.sub(r"([.!?])\s*[,;:.!?]+", r"\1", text)

    # Mixed clause punctuation stutter: ",;" / ";," etc. Keep the first mark.
    text = re.sub(r"([,;:])\s*[,;:]+", r"\1", text)
    return text


def _is_valid_piece(p):
    """Is this a valid word fragment when validating a split?"""
    if p in _SHORT_VALID_PIECES:
        return True
    return len(p) >= 3 and _is_known_word(p)


def _looks_like_merged_word(word):
    """Heuristic: should we try to split this token?

    Only lowercase alphabetic tokens ≥ 4 chars that aren't a recognized word
    in any of our dictionaries. We'd rather leave an oddity than shred a
    legitimate word.
    """
    if len(word) < 4:
        return False
    if not word.isalpha():
        return False
    if not word.islower():
        return False  # skip proper nouns, acronyms
    if _is_known_word(word):
        return False
    return True


def _split_merged_words(text):
    """Split merged tokens using wordninja, but only when every resulting
    piece is itself a known word. Conservative: we'd rather leave a weird
    token than break a legitimate one."""
    if not _HAS_WORDNINJA:
        return text

    def check_and_split(match):
        raw = match.group(0)
        lower = raw.lower()
        if not _looks_like_merged_word(lower):
            return raw
        pieces = wordninja.split(lower)
        if len(pieces) < 2:
            return raw
        # Every piece must be a known word. Short function words (a, to, up,
        # so, in, …) are validated via _SHORT_VALID_PIECES since wordninja's
        # frequency list doesn't cover them. Longer pieces need ≥ 3 chars AND
        # a dictionary hit to prevent noise splits.
        all_valid = all(_is_valid_piece(p) for p in pieces)
        if not all_valid:
            return raw
        return " ".join(pieces)

    # Match alphabetic runs (apostrophes handled separately by _SPEECH_WORDS)
    return re.sub(r"[A-Za-z]+", check_and_split, text)

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


# Brand/product name corrections — Cohere Transcribe mishears these even with
# decoder prompt biasing because the homophones ("cloud" vs "Claude") have
# strong acoustic priors. Deterministic regex post-pass overrides the model.
# Order matters: domain forms must be collapsed before any standalone
# "Cartha" rule would re-touch the surrounding text.
# Domain separator: punctuation, whitespace, or the literal word "dot"
# (Cohere often spells the period out loud as "Dot" in URL dictation).
_DOM_SEP = r"(?:[\s,.]+(?:dot[\s,.]+)?)+"

_BRAND_REPLACEMENTS = [
    # Homophones — case-normalize "cloud code" / "claude code" → "Claude Code"
    (re.compile(r"\b(?:cloud|claude)\s+code\b", re.IGNORECASE), "Claude Code"),
    # Cartha domain forms — tolerates "dot" word, dropped trailing "e"
    # ("Mobil"), and incorrect plural "websites".
    (re.compile(rf"\bcartha{_DOM_SEP}ai{_DOM_SEP}mobile?s?\b", re.IGNORECASE), "cartha.ai.mobile"),
    (re.compile(rf"\bcartha{_DOM_SEP}websites?\b", re.IGNORECASE), "cartha.website"),
    (re.compile(rf"\bcartha{_DOM_SEP}com\b", re.IGNORECASE), "cartha.com"),
    # Camel-case service names
    (re.compile(r"\bcartha[\s.]+cdk[\s.]+service\b", re.IGNORECASE), "CarthaCdkService"),
    # Common Anthropic/AI brand mishearings
    (re.compile(r"\banthropic\b", re.IGNORECASE), "Anthropic"),
]


def _apply_brand_replacements(text):
    for pattern, replacement in _BRAND_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def _strip_repetition_loop(text, min_phrase_words=3, max_phrase_words=30, min_repeats=3):
    """Detect and truncate autoregressive loop degeneration in ASR output.

    Scans for any phrase of min_phrase_words..max_phrase_words words that
    repeats min_repeats or more times consecutively. When found, the text
    is truncated to just before the first repetition. The earliest (longest)
    loop start point wins, so we preserve as much real transcript as possible.

    Returns (cleaned_text, was_truncated).
    """
    words = text.split()
    if len(words) < min_phrase_words * min_repeats:
        return text, False

    best_cut = len(words)  # index in words[] where we'll cut

    for phrase_len in range(min_phrase_words, min_phrase_words + max_phrase_words):
        if phrase_len * min_repeats > len(words):
            break
        # Slide a window looking for consecutive repeats of this phrase length
        i = 0
        while i + phrase_len * min_repeats <= len(words):
            phrase = words[i : i + phrase_len]
            repeats = 1
            j = i + phrase_len
            while j + phrase_len <= len(words):
                if words[j : j + phrase_len] == phrase:
                    repeats += 1
                    j += phrase_len
                else:
                    break
            if repeats >= min_repeats and i < best_cut:
                best_cut = i
                break  # found earliest loop start for this phrase_len
            i += 1

    if best_cut < len(words):
        cleaned = " ".join(words[:best_cut])
        # Trim back to the last sentence-ending punctuation so we don't leave
        # dangling fragments like "...see her and. The"
        last_sentence_end = max(
            cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?")
        )
        if last_sentence_end > len(cleaned) // 3:
            cleaned = cleaned[: last_sentence_end + 1]
        else:
            # No good sentence boundary — just strip trailing connectors
            cleaned = re.sub(
                r"\s+(?:the|a|an|and|but|or|so|is|are|was|of|in|to|that|for|it|with)\s*$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).rstrip(" ,.")
        return cleaned.strip(), True

    return text, False


def format_transcription(text):
    """Apply all post-processing to ASR output."""
    if not text:
        return text

    text, was_looping = _strip_repetition_loop(text)
    if was_looping:
        print(f"Stripped repetition loop, kept {len(text.split())} words", flush=True)

    # Spacing repairs run BEFORE number conversion, because number spans are
    # detected by splitting on whitespace — "3thousand" needs to become
    # "3 thousand" first.
    text = _collapse_duplicate_punctuation(text)  # "Okay,." → "Okay.", "top.." → "top."
    text = _fix_punctuation_spacing(text)      # "shift.should" → "shift. should"
    text = _split_merged_words(text)           # "carcause" → "car cause"

    text = _normalize_digit_multiplier(text)  # "3 thousand" → "3,000" (before word spans)
    text = _convert_number_spans(text)        # "twenty five" → "25"
    text = _normalize_percentages(text)
    text = _normalize_dollars(text)
    text = _apply_brand_replacements(text)    # "cloud code" → "Claude Code", etc.
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
