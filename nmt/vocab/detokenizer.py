"""detokenizer joining tokens back into readable text."""

import re

_JOIN_AFTER = {",", ".", "!", "?", ";", ":", ")", "]", "}", "%"}
_JOIN_BEFORE = {"(", "[", "{"}
_APOSTROPHE = re.compile(r"(\w+)\s+('\w+)")


def detokenize(tokens: list) -> str:
    """join tokens into a line with normal spacing."""
    out = ""
    for i, t in enumerate(tokens):
        if i == 0:
            out = t
        elif t in _JOIN_AFTER or t in _JOIN_BEFORE:
            out += t
        elif tokens[i - 1] in _JOIN_BEFORE:
            out += t
        else:
            out += " " + t
    return _APOSTROPHE.sub(r"\1\2", out)