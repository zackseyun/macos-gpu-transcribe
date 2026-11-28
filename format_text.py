"""Post-processing for ASR output — normalizes numbers, abbreviations, formatting."""
import re


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
    "eslint", "openai", "anthropic", "claude", "cartha", "kartha",
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
        if not _looks_like_merged_word(raw):
            return raw
        lower = raw.lower()
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

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {
    "thousand": 1_000, "million": 1_000_000,
    "billion": 1_000_000_000, "trillion": 1_000_000_000_000,
}
_NUMBER_WORDS = set(_UNITS) | set(_TEENS) | set(_TENS) | set(_SCALES) | {"hundred"}

_TRAILING_PUNCT = ".,!?;:"

# Words after a lone "one" that make it a quantity worth writing as "1".
_ONE_AS_DIGIT_NEXT = {"percent", "dollar", "dollars", "a.m", "p.m", "o'clock"}

# Words before "one" that make it the pronoun ("this one", "no one").
_ONE_AS_PRONOUN_AFTER = {
    "this", "that", "the", "which", "no", "each", "every", "any", "another",
}

# "twenty-four seven" is an idiom, not a number. Without this the parser would
# see two numbers side by side and leave the words; "24/7" is what people mean.
_TWENTY_FOUR_SEVEN = re.compile(
    r"\b(?:twenty[\s-]+four|24)[\s-]+seven\b", re.IGNORECASE
)


def _number_tokens(word):
    """Number tokens in one word ("Twenty-five," → ["twenty", "five"]),
    ["and"] for the connective, or None when the word isn't a number word."""
    core = word.rstrip(_TRAILING_PUNCT).lower()
    if core == "and":
        return ["and"]
    parts = core.split("-")
    if all(p in _NUMBER_WORDS for p in parts):
        return parts
    return None


def _is_numberish(word):
    if not word:
        return False
    core = word.rstrip(_TRAILING_PUNCT)
    if re.fullmatch(r"\$?\d[\d,.]*%?", core):
        return True
    tokens = _number_tokens(word)
    return tokens is not None and tokens != ["and"]


def _parse_below_hundred(tokens, i):
    """Parse 0–99 at tokens[i]; return (value, next index) or None."""
    if i >= len(tokens):
        return None
    word = tokens[i]
    if word in _TENS:
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt in _UNITS and nxt != "zero":
            return _TENS[word] + _UNITS[nxt], i + 2
        return _TENS[word], i + 1
    if word in _TEENS:
        return _TEENS[word], i + 1
    if word in _UNITS:
        return _UNITS[word], i + 1
    return None


def _parse_group(tokens, i):
    """Parse a group like "seven", "forty two", "three hundred and five",
    or "fifteen hundred"; return (value, next index) or None."""
    first = _parse_below_hundred(tokens, i)
    if first is None:
        return None
    value, j = first
    if j < len(tokens) and tokens[j] == "hundred" and value > 0:
        value *= 100
        j += 1
        k = j + 1 if j < len(tokens) and tokens[j] == "and" else j
        rest = _parse_below_hundred(tokens, k)
        if rest is not None and rest[0] > 0:
            value += rest[0]
            j = rest[1]
    return value, j


def _parse_cardinal(tokens):
    """Return the value when tokens spell exactly ONE number, else None.

    The old word2number path did arithmetic on any run of number words, so
    "twenty four seven" became 20*4+7 = 87 and "nineteen eighty four" became
    1,524. Runs that aren't a single well-formed number now return None and
    the caller keeps the spoken words.
    """
    total = 0
    last_scale = None
    i = 0
    while i < len(tokens):
        group = _parse_group(tokens, i)
        if group is None:
            return None
        value, i = group
        if last_scale is not None and value >= last_scale:
            return None  # "one thousand fifteen hundred"
        if i < len(tokens) and tokens[i] in _SCALES:
            scale = _SCALES[tokens[i]]
            if value == 0 or (last_scale is not None and scale >= last_scale):
                return None  # "zero thousand", "five thousand two thousand"
            total += value * scale
            last_scale = scale
            i += 1
            if i + 1 < len(tokens) and tokens[i] == "and":
                i += 1  # "one thousand and five"
            continue
        if i != len(tokens):
            return None  # another number follows: "twenty four seven"
        return total + value
    return total


def _keep_lone_one(words, i):
    """True when a lone "one" is a word ("this one", "no one", "one of them",
    "one thing"), not a count to write as a digit."""
    nxt = words[i + 1].lower().rstrip(_TRAILING_PUNCT) if i + 1 < len(words) else ""
    if words[i] != words[i].rstrip(_TRAILING_PUNCT):
        nxt = ""  # punctuation ends the phrase: "this one, there's..."
    if nxt in _ONE_AS_DIGIT_NEXT or nxt in _SUFFIXES:
        return False
    # Earlier spacing repair turns "p.m." into "p. m." before numbers run.
    after = words[i + 2].lower().rstrip(_TRAILING_PUNCT) if i + 2 < len(words) else ""
    if nxt in ("a", "p") and after == "m":
        return False

    # Keep ranges and lists consistent with their other end: "one or two" →
    # "1 or 2", "one, two, three" → "1, 2, 3". ("one to one" stays words:
    # that's the phrase, not a range.)
    def other_end_is_number(k):
        return 0 <= k < len(words) and _is_numberish(words[k]) and _number_tokens(words[k]) != ["one"]

    prev = words[i - 1].lower().rstrip(_TRAILING_PUNCT) if i >= 1 else ""
    if prev in _ONE_AS_PRONOUN_AFTER:
        return True  # "this one, two of them" — "one" is the pronoun here
    if words[i].endswith(",") and other_end_is_number(i + 1):
        return False
    if i >= 1 and words[i - 1].endswith(",") and other_end_is_number(i - 1):
        return False
    if nxt in ("or", "to") and other_end_is_number(i + 2):
        return False
    if prev in ("or", "to") and other_end_is_number(i - 2):
        return False
    return True


def _convert_number_spans(text):
    """Find spans of number words and convert to digits."""
    text = _TWENTY_FOUR_SEVEN.sub("24/7", text)

    words = text.split()
    result = []
    i = 0

    while i < len(words):
        tokens = _number_tokens(words[i])
        if tokens is None or tokens == ["and"]:
            result.append(words[i])
            i += 1
            continue

        # Collect the run of number words. A word that carries punctuation
        # ends the run, so "twenty-four, seven" stays two separate numbers.
        j = i
        while j < len(words) and _number_tokens(words[j]) is not None:
            j += 1
            if words[j - 1] != words[j - 1].rstrip(_TRAILING_PUNCT):
                break
        while _number_tokens(words[j - 1]) == ["and"]:
            j -= 1  # "five and then" — the "and" is an ordinary word

        span = words[i:j]
        last = span[-1]
        trailing_punct = last[len(last.rstrip(_TRAILING_PUNCT)):]
        span_tokens = [t for w in span for t in _number_tokens(w)]
        num = _parse_cardinal(span_tokens)

        if num is None or (span_tokens == ["one"] and _keep_lone_one(words, i)):
            # Not one well-formed number ("one two three", "nine one one"),
            # or a lone pronoun-ish "one": keep the words exactly as spoken.
            result.extend(span)
            i = j
            continue

        num_str = f"{num:,}" if num >= 1000 else str(num)

        # Check if next word is a suffix (K, M, etc.)
        if not trailing_punct and j < len(words) and words[j].lower().rstrip(_TRAILING_PUNCT) in _SUFFIXES:
            suffix_raw = words[j]
            suffix_clean = _SUFFIXES[suffix_raw.lower().rstrip(_TRAILING_PUNCT)]
            suffix_trailing = suffix_raw[len(suffix_raw.rstrip(_TRAILING_PUNCT)):]
            num_str = str(num) + suffix_clean + suffix_trailing
            j += 1

        result.append(num_str + trailing_punct)
        i = j

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


def _normalize_ampm_spacing(text):
    """Collapse ASR's spaced-out AM/PM forms.

    The local models sometimes emit "6 p. m." or "6 P M" instead of "6 p.m.".
    Keep this tied to a preceding numeric time so normal words like "I am" are
    never touched.
    """
    def repl(match):
        hour = match.group("hour")
        minute = match.group("minute") or ""
        meridiem = match.group("meridiem").lower()
        return f"{hour}{minute} {meridiem}.m."

    return re.sub(
        r"\b(?P<hour>\d{1,2})(?P<minute>:\d{2})?\s+(?P<meridiem>[ap])\s*\.?\s*m\.?(?=\s|$|[,!?;:])",
        repl,
        text,
        flags=re.IGNORECASE,
    )


# Business shorthand spoken letter-"two"-letter. Without this, number cleanup
# read "B two B" as "B" plus "two B" (two billion) and pasted "B 2B".
_LETTER_TWO_LETTER = frozenset({"B2B", "B2C", "C2B", "C2C", "D2C", "P2P", "B2G", "G2B", "G2C"})
_LETTER_TWO_LETTER_RE = re.compile(
    r"\b([bcdgp])(?:\s+(?:two|to)\s+|\s*-?\s*2\s*-?\s*)([bcdgp])\b", re.IGNORECASE
)


def _join_letter_two_letter(text):
    """"B two B" / "B to B" / "B 2B" → "B2B" for the pairs above only."""
    def repl(match):
        joined = f"{match.group(1)}2{match.group(2)}".upper()
        return joined if joined in _LETTER_TWO_LETTER else match.group(0)
    return _LETTER_TWO_LETTER_RE.sub(repl, text)


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
_CARTHA = r"(?:c|k)artha"
# Nemotron writes "Carthage" for Cartha and, in URLs, also "carter dot com".
# Its "Contextual vocabulary" prompt was tried on 2026-11-27 instead: it fixed
# the names but invented "Cartha" on quiet or noisy audio and turned the name
# Carter into Cartha, so names are corrected here, after transcription.
_CARTHA_DOMAIN = r"(?:(?:c|k)arth(?:a|age)|carter)"

_BRAND_REPLACEMENTS = [
    # Developer/device terms where ASR frequently inserts or drops spaces.
    # These are intentionally narrow and deterministic instead of using a broad
    # compound-word joiner, because "wrong spaces" can also be real words.
    (re.compile(r"\bmy\s*i\s*phone\b", re.IGNORECASE), "my iPhone"),
    (re.compile(r"\bi\s+phone\b", re.IGNORECASE), "iPhone"),
    (re.compile(r"\bswift\s*ui\b", re.IGNORECASE), "SwiftUI"),
    (re.compile(r"\bx\s*code\b", re.IGNORECASE), "Xcode"),
    (re.compile(r"\bout\s+ros\b", re.IGNORECASE), "outros"),
    (re.compile(r"\bout\s+ro\b", re.IGNORECASE), "outro"),
    (re.compile(r"\bcode\s+base\b", re.IGNORECASE), "codebase"),
    (re.compile(r"\bpre\s+launch(?:ed)?\b", re.IGNORECASE), "pre-launch"),
    (re.compile(r"\bner\s+fed\b", re.IGNORECASE), "nerfed"),
    # Homophones — case-normalize "cloud code" / "claude code" → "Claude Code"
    (re.compile(r"\b(?:cloud|claude)\s+code\b", re.IGNORECASE), "Claude Code"),
    # Cartha domain forms — tolerates Cohere's hard-C misspelling ("Kartha"),
    # the spoken "dot" word, dropped trailing "e" ("Mobil"), and incorrect
    # plural "websites".
    (re.compile(rf"\b{_CARTHA_DOMAIN}{_DOM_SEP}ai{_DOM_SEP}mobile?s?\b", re.IGNORECASE), "cartha.ai.mobile"),
    (re.compile(rf"\b{_CARTHA_DOMAIN}{_DOM_SEP}websites?\b", re.IGNORECASE), "cartha.website"),
    (re.compile(rf"\b{_CARTHA_DOMAIN}{_DOM_SEP}com\b", re.IGNORECASE), "cartha.com"),
    # Camel-case service names
    (re.compile(rf"\b{_CARTHA}[\s.]+cdk[\s.]+service\b", re.IGNORECASE), "CarthaCdkService"),
    # Standalone brand: Cohere often chooses "Kartha" for the company/app name.
    # Keep this after the domain/service forms so longer canonical names win.
    (re.compile(r"\bkartha\b", re.IGNORECASE), "Cartha"),
    # Nemotron's spelling of Cartha. This also rewrites the ancient city, which
    # comes up far less often in this dictation than the company does.
    (re.compile(r"\bcarthage\b", re.IGNORECASE), "Cartha"),
    # User name: Cohere tends to choose the more common spelling.
    (re.compile(r"\bzach\b", re.IGNORECASE), "Zack"),
    (re.compile(r"\bmichelle\b", re.IGNORECASE), "Mishaal"),
    # Model name: ASR often hears "Qwen" as common names / syllables like
    # "Quinn", "Quan", or the phonetic spelling "Quen".
    (re.compile(r"\b(?:qin|quan|quen|quinn?|qwen)\s*(?:three|3)[\s-]*asr\b", re.IGNORECASE), "Qwen3-ASR"),
    (re.compile(r"\b(?:qin|quan|quen|quinn?|qwen)\s*(?:three|3)\b", re.IGNORECASE), "Qwen3"),
    (re.compile(r"\b(?:qin|quan|quen|quinn?|qwen)\b", re.IGNORECASE), "Qwen"),
    # AI/video-generation product names that ASR often hears as ordinary words.
    # Keep these deterministic because decoder prompts are not available in every
    # local/GPU runtime path.
    (re.compile(r"\b(?:c|sea|see|seed)\s*[- ]?dance\b", re.IGNORECASE), "Seedance"),
    (re.compile(r"\bseeddance\b", re.IGNORECASE), "Seedance"),
    (re.compile(r"\b(?:cling|clang|kling)\b", re.IGNORECASE), "Kling"),
    (re.compile(r"\b(?:cedence|sedance|seedence|ceedance)\b", re.IGNORECASE), "Seedance"),
    (re.compile(r"\b(?:higgs|hicks|higs)[\s-]*field\b", re.IGNORECASE), "Higgsfield"),
    (re.compile(r"\bopen\s*router\b", re.IGNORECASE), "OpenRouter"),
    (re.compile(r"\bcodex\b", re.IGNORECASE), "Codex"),
    # The video-chat site. Nemotron hears "Omega style"; "Omega three" stays.
    (re.compile(r"\bomega(?=[\s-]+(?:style|like|type)\b)", re.IGNORECASE), "Omegle"),
    (re.compile(r"\bomegle\b", re.IGNORECASE), "Omegle"),
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

    text = _join_letter_two_letter(text)      # "B two B" → "B2B" (before "two B" → "2B")
    text = _normalize_digit_multiplier(text)  # "3 thousand" → "3,000" (before word spans)
    text = _convert_number_spans(text)        # "twenty five" → "25"
    text = _normalize_percentages(text)
    text = _normalize_dollars(text)
    text = _normalize_ampm_spacing(text)      # "6 p. m." → "6 p.m."
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
