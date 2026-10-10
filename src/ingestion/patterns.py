"""Every regex the chunkers use to recognise DFSA document structure.

Kept in one place so the formats we support are visible at a glance and can
be unit-tested against real examples (and known false positives).

Numbering in DFSA Rulebook modules:
    chapter      3            "3 CORE RULES - INVESTMENT BUSINESS ..."   (GEN also "2A.")
    section      3.3          "3.3 Key information and Client Agreement"
    rule         3.3.2        "3.3.2 (1) Subject to (2), an Authorised Firm ..."
    appendix     APP2         "APP2 KEY INFORMATION AND CLIENT AGREEMENT"
    app section  A2.1         "A2.1 Core information"
    app rule     A2.1.1       "A2.1.1 The key information ..."
A number part may carry a letter suffix: 3A.1.1, 2.26A.1, 7.12.5A.
"""

import re

# One number part: "3", "3A", "26".
NUM = r"\d+[A-Z]?"

# "3.3.2 ..." or "A2.1.1 ...". The number must be followed by whitespace or
# end of text, so "3.3.2(1)(b)" inside a sentence never matches (and a rule
# reference only matches if an element *starts* with it).
RULE = re.compile(rf"^(?P<app>A)?(?P<num>{NUM}\.{NUM}\.{NUM})(?=\s|$)\s*(?P<rest>.*)$", re.S)

# "3.3 Key information ..." / "A2.1 Core information". The title must start
# with a capital (or a quote/bracket), so "2.5 per cent of ..." doesn't match.
SECTION = re.compile(rf"^(?P<app>A)?(?P<num>{NUM}\.{NUM})\s+(?P<title>[A-Z(\"'‘“].*)$", re.S)

# "3 CORE RULES - ..." / "2A. DEFINITION OF FINANCIAL PRODUCT ...": all-caps title.
CHAPTER = re.compile(rf"^(?P<num>{NUM})\.?\s+(?P<title>[A-Z][A-Z0-9 ,&()'’/:\-–]+)$")

# "APP2 KEY INFORMATION AND CLIENT AGREEMENT" / "APP 3 MARKET DISCLOSURE ...".
APPENDIX = re.compile(r"^APP\s?(?P<num>\d+)\s+(?P<title>.+)$")

GUIDANCE = re.compile(r"^Guidance(\s+notes?)?:?$")

# What follows a rule number when the rule has been removed: "[Deleted]", "Deleted".
DELETED = re.compile(r"^\[?Deleted\]?\.?$", re.I)

# Sub-headings that introduce a section's scope rules ("does/doesn't apply to").
APPLICATION_HEADING = re.compile(r"^(Application|Scope)\b", re.I)

# A scope rule: its subject is the section/chapter/module itself, e.g.
#   "The Rules in this section do not apply to ..."
#   "(1) Subject to (2), this section applies to ..."
# Anchored at the start of the rule text (after the number), so an obligation
# that merely mentions "... to which this section applies" doesn't match.
SCOPE_RULE = re.compile(
    r"^(\(\d+\)\s*)?(Subject to [^,]{1,40},\s*)?"
    r"(this|these|the Rules in this|the Rules in these)\s+"
    r"(section|sections|chapter|module|Rules)\b",
    re.I,
)

# The module's running header when Docling misses it as furniture:
# "CONDUCT OF BUSINESS (COB)".
RUNNING_HEADER = re.compile(r"^[A-Z ,&\-]+\((?P<code>[A-Z]{2,4})\)$")

# Longest section heading we accept from a non-heading element.
MAX_HEADING_CHARS = 160
# Chapter titles can be long lists of activities (COB chapter 3 is ~330 chars);
# an all-caps line after a chapter number is a strong enough signal on its own.
MAX_CHAPTER_CHARS = 500


def number_key(num: str) -> tuple[tuple[int, str], ...]:
    """Sortable key for "3A.12.1" -> ((3, "A"), (12, ""), (1, "")).

    Lets the sequence guard check that rule numbers only move forward.
    """
    key = []
    for part in num.split("."):
        m = re.fullmatch(r"(\d+)([A-Z]?)", part)
        if not m:
            raise ValueError(f"Not a DFSA number part: {part!r} in {num!r}")
        key.append((int(m.group(1)), m.group(2)))
    return tuple(key)
