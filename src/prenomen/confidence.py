"""Reliability-weighted confidence from name-gender dictionary matches.

:func:`weighted_dict_confidence` estimates P(dominant gender | name) from the
dictionaries alone, bounded in [0, 1]. The pipeline uses it for the bypass gate:
when the dictionaries are conclusive enough, the name is settled without a model
call. A single source reduces to its own frequency ratio, so a gate threshold of
0.85 means exactly "frequency ratio >= 0.85".

Each dictionary that matched contributes p = frequency when it agrees with the
gender being scored, else 1 - frequency, where frequency is that source's empirical
P(gender | name). Matches without a frequency carry no evidence and are ignored.
Both genders are scored by that one rule and the higher score wins, so the gender
returned and the confidence attached to it always come from the same arithmetic.

Correlated dictionaries form a *family*: the family carries its reliability weight
once, split evenly across whichever of its members matched, so two near-duplicate
resources for the same language do not compound their weight against a broad one.
Family membership and weight come from each dictionary's spec, so adding a source
needs no code change. See :func:`source_meta_from_specs`.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .dictionaries import DictionarySpec
from .models import DictMatch, Gender

_GENDER_ORDER = {Gender.MALE: 0, Gender.FEMALE: 1}  # deterministic tie-break


@dataclass(frozen=True)
class SourceMeta:
    """Reliability metadata for one dictionary source."""

    family: str
    weight: float


def source_meta_from_specs(
    specs: Sequence[DictionarySpec],
) -> dict[str, SourceMeta]:
    """Build the source metadata map from the dictionary specs being used.

    Each spec carries its own ``family`` (defaulting to its name) and ``weight``
    (defaulting to 1.0), so the calibrated values for the shipped dictionaries live
    in one place -- the registry in :mod:`prenomen.dictionaries` -- rather than being
    restated here where they would drift. Sources sharing a family share one vote.
    """
    return {s.name: SourceMeta(family=s.family, weight=float(s.weight)) for s in specs}


def _resolve(source: str, meta: dict[str, SourceMeta] | None) -> SourceMeta:
    """A source not described by any spec is a standalone family of weight 1."""
    if meta and source in meta:
        return meta[source]
    return SourceMeta(source, 1.0)


def _member_weights(
    found: list[DictMatch], meta: dict[str, SourceMeta] | None
) -> list[float]:
    """Weight per matched source, positionally aligned with ``found``.

    Each family's weight is split evenly across the members of that family that
    actually matched, so near-duplicate resources cast one vote between them.
    """
    resolved = [_resolve(m.source, meta) for m in found]
    fam_counts = Counter(r.family for r in resolved)
    return [r.weight / fam_counts[r.family] for r in resolved]


def weighted_dict_confidence(
    dict_matches: list[DictMatch],
    source_meta: dict[str, SourceMeta] | None = None,
) -> tuple[Gender, float] | None:
    """Dominant gender and the reliability-weighted mean of P(dominant | name).

    Both genders are scored as::

        score(g) = sum(w_s * p_s(g)) / sum(w_s)

    with p_s(g) = frequency_s if source s names g else 1 - frequency_s, and w_s the
    source's reliability weight. The higher-scoring gender is returned with its own
    score as the confidence, so the two can never contradict each other. The result
    is bounded in [0, 1] and never below 0.5; a single source reduces to its own
    frequency ratio. Returns ``None`` when no usable match is present.
    """
    found = [
        m
        for m in dict_matches
        if m.found
        and m.gender not in (None, Gender.NOT_RETRIEVABLE)
        and m.frequency is not None
    ]
    if not found:
        return None

    weights = _member_weights(found, source_meta)
    weight_sum = sum(weights)
    if weight_sum == 0.0:
        return None

    # Score every candidate gender by the same rule that reports the confidence, so
    # the gender returned and the number attached to it can never disagree. Scoring
    # the winner separately from the vote lets two weak agreeing sources out-vote one
    # strong opposing source and then be handed a confidence below 0.5.
    def score(gender: Gender) -> float:
        return (
            sum(
                w * (m.frequency if m.gender == gender else 1.0 - m.frequency)
                for m, w in zip(found, weights)
            )
            / weight_sum
        )

    scores = {g: score(g) for g in (Gender.MALE, Gender.FEMALE)}

    # A genuine tie is unresolvable from the dictionaries alone, so report maximal
    # uncertainty (0.5) -- below any sane threshold, so it never bypasses. The gender
    # is still returned deterministically (MALE-first).
    top = max(scores.values())
    tied = [
        g
        for g, s in scores.items()
        if math.isclose(s, top, rel_tol=1e-9, abs_tol=1e-12)
    ]
    dominant = min(tied, key=lambda g: _GENDER_ORDER.get(g, 99))
    if len(tied) > 1:
        return dominant, 0.5

    return dominant, scores[dominant]
