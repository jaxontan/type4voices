"""Conservative post-processing for dictated text.

The ASR model is good at words but inconsistent about numbers, spoken
punctuation and fillers.  This module adds a small deterministic cleanup
layer.  It deliberately avoids aggressive rewriting: English text only,
single digits such as "one" are left alone, and a bare "dot" is not treated
as a period when it is clearly being used as a noun ("the red dot").
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Number words
# ---------------------------------------------------------------------------
_DIGIT_WORDS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_TEEN_WORDS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS_WORDS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALE_WORDS = {
    "hundred": 100, "thousand": 1_000, "million": 1_000_000,
    "billion": 1_000_000_000,
}
_ALL_NUMBER_WORDS = (
    set(_DIGIT_WORDS) | set(_TEEN_WORDS) | set(_TENS_WORDS) | set(_SCALE_WORDS)
)
_NUMBER_WORD_PATTERN = "|".join(
    sorted(_ALL_NUMBER_WORDS, key=len, reverse=True)
)

# A number phrase may contain spaces, hyphens, "and" after a scale, and a
# decimal "point"/"dot" followed by digit words.
_NUM_PHRASE_RE = re.compile(
    rf"(?i)(?<![\w'-])"
    rf"(?:(?:minus|negative|a|an)[\s-]+)?"
    rf"(?:{_NUMBER_WORD_PATTERN})"
    rf"(?:"
    rf"[\s-]+(?:and[\s-]+)?(?:{_NUMBER_WORD_PATTERN})"
    rf"|[\s-]+(?:point|dot)[\s-]+(?:{_NUMBER_WORD_PATTERN})"
    rf"(?:[\s-]+(?:{_NUMBER_WORD_PATTERN}))*"
    rf")*"
    rf"(?![\w'-])"
)


def _parse_cardinal(words: list[str]) -> int | None:
    """Parse a cardinal number phrase, or return None if it is not valid."""
    if not words:
        return None

    implicit_one = False
    if words[0] in ("a", "an"):
        implicit_one = True
        words = words[1:]
        if not words or words[0] not in _SCALE_WORDS:
            return None

    cleaned: list[str] = []
    for i, word in enumerate(words):
        if word == "and":
            # "one hundred and five" is fine; "one and two" is not a
            # number phrase and must be left as words.
            if i == 0 or i == len(words) - 1:
                return None
            if words[i - 1] not in _SCALE_WORDS:
                return None
            continue
        if word not in _ALL_NUMBER_WORDS:
            return None
        cleaned.append(word)

    if not cleaned:
        return None
    if implicit_one:
        cleaned.insert(0, "one")

    total = 0
    current = 0
    last: str | None = None
    last_scale: int | None = None

    for word in cleaned:
        if word in _DIGIT_WORDS:
            digit = _DIGIT_WORDS[word]
            if last in ("digit", "teen", "unit"):
                return None
            if last == "tens":
                if digit == 0:
                    return None
                current += digit
                last = "unit"
            else:
                current += digit
                last = "digit"

        elif word in _TEEN_WORDS:
            if last in ("digit", "teen", "tens", "unit"):
                return None
            if current % 100 != 0:
                return None
            current += _TEEN_WORDS[word]
            last = "teen"

        elif word in _TENS_WORDS:
            if last in ("digit", "teen", "tens", "unit"):
                return None
            if current % 100 != 0:
                return None
            current += _TENS_WORDS[word]
            last = "tens"

        elif word == "hundred":
            if last in ("hundred", "scale"):
                return None
            if current >= 100:
                return None
            current = (current or 1) * 100
            last = "hundred"

        elif word in _SCALE_WORDS:
            scale = _SCALE_WORDS[word]
            if last == "scale" and current == 0:
                return None
            if last_scale is not None and scale >= last_scale:
                return None
            total += (current or 1) * scale
            current = 0
            last = "scale"
            last_scale = scale

        else:
            return None

    return total + current


def _parse_decimal(words: list[str], negative: bool) -> str | None:
    """Parse a decimal phrase such as 'three point five'."""
    if sum(word in ("point", "dot") for word in words) != 1:
        return None
    index = next(
        i for i, word in enumerate(words) if word in ("point", "dot")
    )
    int_words = words[:index]
    frac_words = words[index + 1:]
    if not frac_words or any(word not in _DIGIT_WORDS for word in frac_words):
        return None

    if not int_words:
        int_str = "0"
    elif all(word in _DIGIT_WORDS for word in int_words):
        int_str = "".join(str(_DIGIT_WORDS[word]) for word in int_words)
    else:
        value = _parse_cardinal(int_words)
        if value is None:
            return None
        int_str = str(value)

    frac_str = "".join(str(_DIGIT_WORDS[word]) for word in frac_words)
    sign = "-" if negative else ""
    return f"{sign}{int_str}.{frac_str}"


def _parse_number_phrase(words: list[str]) -> str | None:
    if not words:
        return None

    negative = False
    if words[0] in ("minus", "negative"):
        negative = True
        words = words[1:]
        if not words:
            return None

    if any(word in ("point", "dot") for word in words):
        return _parse_decimal(words, negative)

    if all(word in _DIGIT_WORDS for word in words):
        if len(words) >= 2:
            digits = "".join(str(_DIGIT_WORDS[word]) for word in words)
            return ("-" if negative else "") + digits
        # A lone "one"/"two" is too ambiguous to rewrite; a negative one
        # ("minus five") is unambiguous.
        if negative:
            return str(-_DIGIT_WORDS[words[0]])
        return None

    value = _parse_cardinal(words)
    if value is None:
        return None
    if len(words) == 1 and words[0] in _SCALE_WORDS:
        if negative:
            return str(-value)
        return None
    return ("-" if negative else "") + str(value)


def _convert_numbers(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        words = [w.lower() for w in re.findall(r"[A-Za-z]+", raw)]
        result = _parse_number_phrase(words)
        return raw if result is None else result

    return _NUM_PHRASE_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Fillers
# ---------------------------------------------------------------------------
_FILLER_RE = re.compile(
    r"(?i)(?<![\w-])(?:uh+|um+|ah+|erm|hmm+)"
    r"(?![\w-])(?![ \t]*,?[ \t]*(?:oh|ha)\b)"
    r"(?:[ \t]*,)?([ \t]*)([A-Za-z])?"
)


def _remove_fillers(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        before = text[:match.start()].rstrip()
        at_sentence_start = (
            not before or before[-1] in ".?!\n\"'“‘"
        )
        letter = match.group(2)
        if letter:
            if at_sentence_start:
                return letter.upper()
            return " " + letter
        return ""

    return _FILLER_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# Spoken punctuation
# ---------------------------------------------------------------------------
_DETERMINERS = {
    "the", "a", "an", "this", "that", "these", "those", "my", "your",
    "his", "her", "its", "our", "their", "some", "any", "no", "every",
    "each", "either", "neither", "another",
}
_DOT_NOUN_PRECEDERS = _DETERMINERS | {
    # common adjectives used before the noun "dot"
    "red", "blue", "green", "black", "white", "yellow", "orange",
    "purple", "pink", "gray", "grey", "brown", "small", "big", "large",
    "tiny", "little", "round", "single", "double", "bright", "dark",
    "faint", "visible", "invisible", "central", "left", "right", "top",
    "bottom", "middle", "first", "last", "next", "previous", "same",
    "different", "solid", "hollow", "polka", "quantum",
    # words that introduce a mention of the symbol itself
    "word", "symbol", "character", "letter", "mark", "sign",
    "punctuation", "separator", "notation", "product", "pattern",
    "grid", "matrix", "circle", "key", "button",
}
_PUNCT_MENTION_PREV = {
    "word", "symbol", "character", "letter", "mark", "sign",
    "punctuation", "separator", "notation", "called", "name", "named",
    "say", "says", "said", "mean", "means", "meant", "meaning",
}
_PUNCT_MENTION_NEXT = {
    "is", "are", "was", "were", "means", "mean", "symbol", "character",
    "key", "button", "sign", "mark", "punctuation", "separator",
    "notation", "indicates", "separates",
}
_PUNCT_NOUN_PRECEDERS = _DETERMINERS | _PUNCT_MENTION_PREV
_TLDS = {
    "com", "org", "net", "edu", "gov", "mil", "int", "io", "ai", "co",
    "dev", "app", "info", "biz", "xyz", "uk", "de", "cn", "jp", "fr",
    "au", "ca", "eu", "ru", "br", "nl", "se", "ch", "kr", "tw", "hk",
    "sg", "nz", "za", "mx", "ar", "cl", "ie", "il", "tr", "ua", "vn",
    "id", "ph", "th",
}
_PUNCT_WORD_PATTERN = (
    r"(?:comma|period|colon|semicolon|question[ \t]+mark|"
    r"exclamation[ \t]+(?:mark|point)|full[ \t]+stop|dot)"
)


def _prev_word(text: str, position: int) -> str:
    match = re.search(
        r"([A-Za-z]+|\d+(?:[.,]\d+)*)[\s.,!?;:'\"()\[\]{}]*$",
        text[:position],
    )
    return match.group(1).lower() if match else ""


def _next_word(text: str, position: int) -> str:
    match = re.search(
        r"^[\s.,!?;:'\"()\[\]{}]*([A-Za-z]+|\d+(?:[.,]\d+)*)",
        text[position:],
    )
    return match.group(1).lower() if match else ""


def _is_number_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:[.,]\d+)*", token)) or (
        token in _ALL_NUMBER_WORDS
    )


def _dot_at_boundary(text: str, end: int) -> bool:
    tail = text[end:].lstrip(" \t")
    return not tail or tail[0] in "\r\n.,!?;:"


def _punct_guard(text: str, start: int, end: int) -> bool:
    prev = _prev_word(text, start)
    nxt = _next_word(text, end)
    if not prev:
        return False
    if prev in _PUNCT_NOUN_PRECEDERS or prev in _PUNCT_MENTION_PREV:
        return False
    if nxt in _PUNCT_MENTION_NEXT:
        return False
    # Do not rewrite lists that talk about punctuation, e.g.
    # "comma and period" or "Please say comma and period".
    tail = text[:start].rstrip()
    if re.search(
        rf"(?i)\b{_PUNCT_WORD_PATTERN}[ \t]+(?:and|or)$", tail
    ):
        return False
    after = text[end:]
    if re.match(
        rf"(?i)^[\s.,!?;:]*\b(?:and|or)\b[ \t]+{_PUNCT_WORD_PATTERN}\b",
        after,
    ):
        return False
    return True


def _replace_command(
    text: str,
    pattern: str,
    symbol: str,
    *,
    join: bool = False,
    guard=None,
) -> str:
    regex = re.compile(pattern, re.I)
    out: list[str] = []
    last = 0
    for match in regex.finditer(text):
        if guard is not None and not guard(text, match.start(), match.end()):
            continue
        left = match.start()
        while left > last and text[left - 1] in " \t":
            left -= 1
        if symbol in ".,;:!?" and left > last and text[left - 1] == symbol:
            left -= 1
        right = match.end()
        while right < len(text) and text[right] in " \t":
            right += 1
        if symbol in ".,;:!?":
            if right < len(text) and text[right] == symbol:
                right += 1
            while right < len(text) and text[right] in " \t":
                right += 1
        out.append(text[last:left])
        out.append(symbol)
        if not join and right < len(text) and text[right] not in "\n.,!?;:)]}":
            out.append(" ")
        last = right
    out.append(text[last:])
    return "".join(out)


def _apply_dot(text: str) -> str:
    regex = re.compile(r"(?i)(?<![\w'-])dot(?![\w'-])")
    out: list[str] = []
    last = 0
    for match in regex.finditer(text):
        prev = _prev_word(text, match.start())
        nxt = _next_word(text, match.end())
        prev_number = _is_number_token(prev)
        next_number = _is_number_token(nxt)
        at_boundary = _dot_at_boundary(text, match.end())

        convert = False
        if prev_number and next_number:
            # This should normally have been handled by the number parser.
            convert = False
        elif prev_number:
            convert = True
        elif nxt in _TLDS and prev and prev not in _DOT_NOUN_PRECEDERS:
            convert = True
        elif (
            at_boundary
            and prev
            and prev not in _DOT_NOUN_PRECEDERS
            and prev not in _PUNCT_MENTION_PREV
        ):
            convert = True

        if not convert:
            continue

        left = match.start()
        while left > last and text[left - 1] in " \t":
            left -= 1
        right = match.end()
        while right < len(text) and text[right] in " \t":
            right += 1
        out.append(text[last:left])
        out.append(".")
        last = right
    out.append(text[last:])
    return "".join(out)


def _apply_spoken_punctuation(text: str) -> str:
    text = _replace_command(
        text, r"(?<![\w'-])new[ \t]+paragraph(?![\w'-])", "\n\n", join=True
    )
    text = _replace_command(
        text, r"(?<![\w'-])new[ \t]+line(?![\w'-])", "\n", join=True
    )
    text = _replace_command(
        text, r"(?<![\w'-])question[ \t]+mark(?![\w'-])", "?", guard=_punct_guard
    )
    text = _replace_command(
        text,
        r"(?<![\w'-])exclamation[ \t]+(?:mark|point)(?![\w'-])",
        "!",
        guard=_punct_guard,
    )
    text = _replace_command(
        text, r"(?<![\w'-])full[ \t]+stop(?![\w'-])", ".", guard=_punct_guard
    )
    text = _replace_command(
        text, r"(?<![\w'-])period(?![\w'-])", ".", guard=_punct_guard
    )
    text = _replace_command(
        text, r"(?<![\w'-])comma(?![\w'-])", ",", guard=_punct_guard
    )
    text = _replace_command(
        text, r"(?<![\w'-])semicolon(?![\w'-])", ";", guard=_punct_guard
    )
    text = _replace_command(
        text, r"(?<![\w'-])colon(?![\w'-])", ":", guard=_punct_guard
    )
    return _apply_dot(text)


# ---------------------------------------------------------------------------
# Final cleanup
# ---------------------------------------------------------------------------
def _cleanup_punctuation(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])[ \t]+", r"\1", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r";{2,}", ";", text)
    text = re.sub(r":{2,}", ":", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)
    text = re.sub(r"[,;:]\s*([.!?])", r"\1", text)
    text = re.sub(r"([.!?])\s*[,;:]", r"\1", text)
    text = re.sub(r"^[\s,.;:]+", "", text)
    text = re.sub(r"[\s,;:]+$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def postprocess_text(text: str) -> str:
    """Apply the conservative dictation cleanup pipeline."""
    if not text:
        return text
    text = _remove_fillers(text)
    text = _convert_numbers(text)
    text = _apply_spoken_punctuation(text)
    return _cleanup_punctuation(text)
