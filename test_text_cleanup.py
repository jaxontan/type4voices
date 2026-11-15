"""Small offline test harness for text_cleanup.postprocess_text()."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_cleanup import postprocess_text

CASES = [
    # numbers and digit sequences
    ("Call one two three dot.", "Call 123."),
    ("My number is five five five one two three four.", "My number is 5551234."),
    ("I have twenty five apples.", "I have 25 apples."),
    ("There are one hundred and five files.", "There are 105 files."),
    ("It costs three thousand two hundred dollars.", "It costs 3200 dollars."),
    ("Version one point two three.", "Version 1.23."),
    ("Version one dot two.", "Version 1.2."),
    ("One of them is red.", "One of them is red."),
    ("The code is four two zero.", "The code is 420."),
    ("A hundred and five files.", "105 files."),
    ("twenty five", "25"),
    ("nineteen ninety", "nineteen ninety"),
    ("one two three four five six seven eight nine zero", "1234567890"),
    ("minus five", "-5"),
    ("one point zero", "1.0"),
    ("one two three point four five", "123.45"),
    ("The value is three point five.", "The value is 3.5."),
    ("I have two apples.", "I have two apples."),

    # dot: numbers, domains, and the noun "dot"
    ("Open example dot com.", "Open example.com."),
    ("one two three dot com", "123.com"),
    ("example dot co dot uk", "example.co.uk"),
    ("The value is three dot five.", "The value is 3.5."),
    ("one two three dot four five", "123.45"),
    ("The red dot.", "The red dot."),
    ("The dot is red.", "The dot is red."),
    ("The word dot.", "The word dot."),
    ("The symbol dot is small.", "The symbol dot is small."),
    ("The dot com bubble", "The dot com bubble"),
    ("Please send it dot.", "Please send it."),
    ("Hello dot.", "Hello."),

    # spoken punctuation
    ("Hello period.", "Hello."),
    ("The period is important.", "The period is important."),
    ("The comma is missing.", "The comma is missing."),
    ("Hello comma world period.", "Hello, world."),
    ("Hello, comma world. Period.", "Hello, world."),
    ("Hello comma world.", "Hello, world."),
    ("Hello question mark", "Hello?"),
    ("Hello exclamation mark", "Hello!"),
    ("Key colon value", "Key: value"),
    ("First semicolon second", "First; second"),
    ("Line one new line line two", "Line one\nline two"),
    ("Paragraph one new paragraph paragraph two", "Paragraph one\n\nparagraph two"),
    ("world period", "world."),
    ("The period.", "The period."),
    ("A comma.", "A comma."),
    ("Please say comma and period.", "Please say comma and period."),
    ("Hello comma and period.", "Hello comma and period."),
    ("Please say dot.", "Please say dot."),
    ("I mean dot.", "I mean dot."),

    # fillers
    ("Uh, I think, um, this is a test.", "I think, this is a test."),
    ("I think. Um, this is a test.", "I think. This is a test."),
    ("Ah, hello.", "Hello."),
    ("uh-huh", "uh-huh"),
    ("Uh oh.", "Uh oh."),
    ("Uh, i think.", "I think."),
    ("I think, uh, this is a test.", "I think, this is a test."),

    # spacing / duplicate punctuation
    ("Hello,, world..", "Hello, world."),
    ("Hello , world .", "Hello, world."),
    ("Hello... world", "Hello... world"),
    ("  Hello   world  ", "Hello world"),

    # Chinese must pass through untouched
    ("你好，这是一个测试。", "你好，这是一个测试。"),
    ("嗯，你好。", "嗯，你好。"),
    ("uh 你好 one two three", "你好 123"),
]


def main() -> int:
    failed = 0
    for source, expected in CASES:
        actual = postprocess_text(source)
        if actual != expected:
            failed += 1
            print("FAIL")
            print("  in :", source.encode("unicode_escape").decode("ascii"))
            print("  exp:", expected.encode("unicode_escape").decode("ascii"))
            print("  got:", actual.encode("unicode_escape").decode("ascii"))
    print(f"{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
