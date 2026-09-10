"""Load and query name-gender dictionaries.

Two file shapes are supported, selected by each spec's ``format``:

``wgnd``
    One row per (name, country, gender) with a frequency weight. Optionally
    country-aware: at query time the lookup is restricted to the countries the
    author is affiliated with, falling back to the global aggregate when there is
    no overlap. Built for the World Gender Name Dictionary, but any file with
    those four columns works.

``counts``
    One row per name with a male count and a female count, in any units.
    Rows are aggregated by name, so files that repeat a name across regions or
    years (electoral rolls, census extracts) load correctly.

Adding a dictionary is a :class:`~prenomen.dictionaries.DictionarySpec`, not code, as
long as it fits one of these two shapes. Column names are configurable for both.

The two costs of a dictionary are separated deliberately. Checking that a file exists
and that its header holds the configured columns is milliseconds, and happens when
the lookup is constructed, so a mistyped column is reported before anything expensive
starts. Parsing is minutes and gigabytes -- WGND alone is 3.8M names -- and happens on
:meth:`MultiDBLookup.load`, called by the first query or up front by the caller.
"""

import gzip
import logging
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from .dictionaries import DictionarySpec
from .models import DictMatch, Gender

logger = logging.getLogger(__name__)

# Rough cost of parsing each shipped dictionary, for the one log line that warns a
# caller a five-minute pause has started. Deliberately approximate.
_LOAD_COST_GB = {"wgnd": 5.5, "naampy": 1.1, "chinese_cn": 0.2, "chinese_pinyin": 0.02}


def _load_wgnd(source: str, path: str, cfg: dict):
    """Load a (name, country, gender, weight) dictionary.

    Returns a plain ``name -> DictMatch`` dict, or, when ``country_aware`` is set,
    a callable ``(first_key, full_key, countries) -> Optional[DictMatch]``.
    """
    df = pd.read_csv(
        path, sep=cfg.get("separator", ","), low_memory=False,
        keep_default_na=False, na_values=[""],
    )
    name_col = cfg.get("name_col", "name")
    gender_col = cfg.get("gender_col", "gender")
    freq_col = cfg.get("freq_col", "wgt")
    country_col = cfg.get("country_col", "code")
    country_filter = cfg.get("country_filter")
    country_aware = bool(cfg.get("country_aware", False))

    if country_filter and country_col in df.columns:
        df = df[df[country_col].isin(country_filter)]

    # Vectorised preprocessing: WGND has ~26M rows, so row-wise iteration is too slow.
    # Null names are dropped before the string cast: pandas 2 renders them as the
    # literal "nan", which would enter the index and collide with the real name Nan.
    df = df[df[name_col].notna()]
    name_s = df[name_col].astype(str).str.strip().str.lower()
    gender_s = (
        df[gender_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"M": Gender.MALE, "F": Gender.FEMALE})
    )
    weight_s = (
        pd.to_numeric(df[freq_col], errors="coerce").fillna(1.0)
        if freq_col in df.columns
        else pd.Series(1.0, index=df.index)
    )
    code_s = (
        df[country_col].astype(str).str.strip().str.upper()
        if country_col in df.columns
        else pd.Series("", index=df.index)
    )

    mask = gender_s.notna() & name_s.str.len().gt(0)
    clean = pd.DataFrame(
        {
            "name": name_s[mask].values,
            "code": code_s[mask].values,
            "gender": gender_s[mask].values,
            "weight": weight_s[mask].values,
        }
    )

    # Pre-aggregate at (name, code, gender) to shrink the map and speed up queries.
    agg = clean.groupby(["name", "code", "gender"], as_index=False)["weight"].sum()

    per_name: dict[str, list[tuple[str, Gender, float]]] = {}
    if country_aware:
        for name, grp in agg.groupby("name", sort=False):
            per_name[name] = list(
                zip(grp["code"].tolist(), grp["gender"].tolist(), grp["weight"].tolist())
            )

    # Global (all-country) dominant gender per name, used directly and as fallback.
    global_agg = clean.groupby(["name", "gender"], as_index=False)["weight"].sum()
    aggregated: dict[str, DictMatch] = {}
    for name, grp in global_agg.groupby("name", sort=False):
        counts = dict(zip(grp["gender"].tolist(), grp["weight"].tolist()))
        dominant = max(counts, key=counts.__getitem__)
        total = sum(counts.values())
        aggregated[name] = DictMatch(
            source=source,
            gender=dominant,
            frequency=counts[dominant] / total if total > 0 else None,
            found=True,
        )

    logger.info(
        f"{source}: loaded {len(aggregated):,} name entries "
        f"(country_aware={country_aware})"
    )

    if not country_aware:
        return aggregated

    def lookup(
        first_key: str, full_key: str, countries: Optional[list[str]]
    ) -> Optional[DictMatch]:
        # This format is indexed by first name only.
        key = first_key
        if not countries:
            return aggregated.get(key)
        entries = per_name.get(key)
        if not entries:
            return None
        upper = {c.strip().upper() for c in countries if c}
        filtered = [(g, w) for (c, g, w) in entries if c in upper]
        if not filtered:
            # No overlap with the given countries: fall back to the global dominant.
            return aggregated.get(key)
        counts: dict[Gender, float] = {}
        for g, w in filtered:
            counts[g] = counts.get(g, 0.0) + w
        dominant = max(counts, key=counts.__getitem__)
        total = sum(counts.values())
        return DictMatch(
            source=source,
            gender=dominant,
            frequency=counts[dominant] / total if total > 0 else None,
            found=True,
        )

    return lookup


def _load_counts(source: str, path: str, cfg: dict) -> dict[str, DictMatch]:
    """Load a (name, male_count, female_count) dictionary.

    Counts are summed per name, so a file with one row per name-per-region loads
    the same as one with a single row per name. The gender is the larger count and
    the frequency is that count as a share of the total.
    """
    sep = cfg.get("separator", "\t")
    name_col = cfg.get("name_col")
    male_col = cfg.get("male_col", "male")
    female_col = cfg.get("female_col", "female")
    if name_col is None:
        raise ValueError("format 'counts' requires a name_col")

    # Check the header before reading the body, then read only the three columns we
    # use. Electoral-roll extracts carry per-state and per-year columns we ignore,
    # and at tens of millions of rows materialising them costs gigabytes.
    available = list(pd.read_csv(path, sep=sep, nrows=0).columns)
    for col in (name_col, male_col, female_col):
        if col not in available:
            raise ValueError(f"column {col!r} is not in the file header {available}")

    df = pd.read_csv(
        path,
        sep=sep,
        usecols=[name_col, male_col, female_col],
        keep_default_na=False,
        na_values=[""],
    )
    # See _load_wgnd: drop nulls before the cast, or pandas 2 indexes them as "nan".
    # One copy up front, so the column assignments below are not chained onto a slice.
    df = df[df[name_col].notna()].copy()
    df[name_col] = df[name_col].astype(str).str.strip().str.lower()
    df[male_col] = pd.to_numeric(df[male_col], errors="coerce").fillna(0.0)
    df[female_col] = pd.to_numeric(df[female_col], errors="coerce").fillna(0.0)
    df = df[df[name_col].str.len() > 0]

    agg = df.groupby(name_col, as_index=False)[[male_col, female_col]].sum()
    agg["_total"] = agg[male_col] + agg[female_col]
    agg = agg[agg["_total"] > 0]
    agg["_is_male"] = agg[male_col] >= agg[female_col]
    agg["_freq"] = agg["_is_male"] * (agg[male_col] / agg["_total"]) + (
        ~agg["_is_male"]
    ) * (agg[female_col] / agg["_total"])

    index: dict[str, DictMatch] = {
        key: DictMatch(
            source=source,
            gender=Gender.MALE if is_male else Gender.FEMALE,
            frequency=freq,
            found=True,
        )
        for key, is_male, freq in zip(
            agg[name_col].tolist(), agg["_is_male"].tolist(), agg["_freq"].tolist()
        )
    }

    logger.info(f"{source}: loaded {len(index):,} name entries")
    return index


_FORMATS = {
    "wgnd": _load_wgnd,
    "counts": _load_counts,
}


def _header_columns(path: str, separator: str) -> list[str]:
    """The first line's column names, without reading the body.

    Handles gzip by suffix, because naampy ships as .csv.gz. Distinct from
    :func:`prenomen.dictionaries.check_header`, which validates a partial download
    against the registry before it is renamed into place; this validates a file
    already on disk against the spec that will read it.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    return [c.strip().strip('"') for c in first.rstrip("\r\n").split(separator)]


class MultiDBLookup:
    """Query every configured dictionary and return one match per source.

    Construction validates; :meth:`load` parses. Both are safe to call repeatedly.
    """

    def __init__(self, specs: Sequence[DictionarySpec] = (), *, preload: bool = False):
        self._specs: list[DictionarySpec] = list(specs)
        # Values are either dict[str, DictMatch] or a callable
        # (first_key, full_key, countries) -> Optional[DictMatch].
        self._indexes: dict[str, Any] = {}
        self._not_found: dict[str, DictMatch] = {
            s.name: DictMatch(source=s.name, found=False) for s in self._specs
        }
        self._loaded = False
        self._lock = threading.Lock()

        duplicate = [
            s.name for s in self._specs if [x.name for x in self._specs].count(s.name) > 1
        ]
        if duplicate:
            raise ValueError(
                f"Dictionary {duplicate[0]!r} is listed twice; each name must be unique."
            )

        # A dictionary that was asked for and cannot be read stops everything here.
        # Continuing would quietly drop to a slower, less accurate and more expensive
        # LLM-only pass, which on a corpus-scale run is hours and an API bill before
        # anyone notices.
        for spec in self._specs:
            if not Path(spec.path).exists():
                raise ValueError(
                    f"Dictionary {spec.name!r} is missing its file: {spec.path}\n"
                    f"Run `prenomen-dicts get {spec.name}`, correct the path, or "
                    f"leave it out of `dictionaries`."
                )
            try:
                available = _header_columns(spec.path, spec.separator)
            except OSError as exc:
                raise ValueError(
                    f"Dictionary {spec.name!r} could not be read from "
                    f"{spec.path}: {exc}"
                ) from exc
            missing = [c for c in spec.required_columns if c not in available]
            if missing:
                raise ValueError(
                    f"Dictionary {spec.name!r}: column(s) {missing} are not in the "
                    f"header of {spec.path}. Found: {available}"
                )

        if preload:
            self.load()

    def load(self) -> "MultiDBLookup":
        """Parse every dictionary into memory. Idempotent, and safe across threads."""
        if self._loaded:
            return self
        with self._lock:
            if self._loaded:
                return self
            if self._specs:
                cost = sum(_LOAD_COST_GB.get(s.name, 0.2) for s in self._specs)
                logger.info(
                    f"Loading {len(self._specs)} dictionar"
                    f"{'y' if len(self._specs) == 1 else 'ies'} "
                    f"(roughly {cost:.1f} GB); this takes minutes."
                )
            for spec in self._specs:
                try:
                    self._indexes[spec.name] = _FORMATS[spec.format](
                        spec.name, str(spec.path), spec.reader_cfg()
                    )
                except Exception as e:
                    # Including pandas' own ValueError subclasses, whose messages
                    # ("No columns to parse from file") name neither source nor path.
                    raise ValueError(
                        f"Dictionary {spec.name!r} could not be read from "
                        f"{spec.path}: {e}"
                    ) from e
            self._loaded = True
        return self

    @property
    def loaded_sources(self) -> list[str]:
        """Source names, in the order given. Fixes the output column order."""
        return [s.name for s in self._specs]

    @property
    def specs(self) -> list[DictionarySpec]:
        return list(self._specs)

    def query(
        self,
        first_name: str,
        last_name: str = "",
        countries: Optional[Sequence[str]] = None,
    ) -> list[DictMatch]:
        """Return one DictMatch per dictionary (found=True only when matched).

        The full name is tried first, then the first name alone, so dictionaries
        indexed either way work without configuration. Country-aware sources
        receive ``countries`` and filter on it.
        """
        self.load()
        results = []
        key_first = first_name.strip().lower()
        key_full = f"{first_name.strip()} {last_name.strip()}".strip().lower()
        countries = list(countries) if countries else None

        for source, index in self._indexes.items():
            if callable(index):
                match = index(key_first, key_full, countries)
            else:
                match = index.get(key_full) or index.get(key_first)
            results.append(match if match else self._not_found[source])

        return results
