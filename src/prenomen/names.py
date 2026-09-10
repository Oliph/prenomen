"""Name and country-code string handling.

Pure string work, no third-party imports, so this module stays cheap to import.
``split_full_name`` is the one piece of the 1.x data loader that was worth keeping:
splitting a full name is name logic, not file reading, and the CJK convention it
encodes would otherwise be re-derived by every caller with a single name column.
"""


def _is_cjk_name(s: str) -> bool:
    """True when the string is entirely CJK unified ideographs (so, no spaces)."""
    return all(
        "一" <= ch <= "鿿"  # CJK Unified Ideographs
        or "㐀" <= ch <= "䶿"  # CJK Extension A
        or "\U00020000" <= ch <= "\U0002a6df"  # CJK Extension B
        for ch in s
    )


def split_full_name(full: str) -> tuple[str, str] | None:
    """Split a single full-name string into (first, last), or None if unusable.

    Splits on the last whitespace. Names written without spaces in CJK scripts are
    surname-first by convention, so the leading character is taken as the surname.
    """
    if not isinstance(full, str) or not full.strip():
        return None
    full = full.strip()
    parts = full.rsplit(None, 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    if _is_cjk_name(full):
        return full[1:], full[0]
    return full, ""


def normalize_countries(raw) -> tuple[str, ...]:
    """Normalise country affiliations to a sorted tuple of upper-case codes.

    Accepts the ``"FR;PL"`` string a table column holds, or any sequence of codes,
    because requiring ``.split(";")`` at every call site buys nothing. Anything else
    reads as "no countries".

    The trim/upper/dedupe/sort is load-bearing: the rendered prompt joins this tuple
    in order, and the golden fixtures in tests/fixtures pin that text.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = raw.split(";")
    elif isinstance(raw, (list, tuple, set, frozenset)):
        parts = [p for p in raw if isinstance(p, str)]
    else:
        return ()
    return tuple(sorted({c.strip().upper() for c in parts if c and c.strip()}))
