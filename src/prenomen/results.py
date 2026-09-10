"""The input and output records of one inference.

These are plain dataclasses, not pydantic models. Nothing here comes off a wire, a
run can produce millions of them, and ``record`` holds arbitrary user objects that a
pydantic ``model_dump`` would choke on. Pydantic stays where it earns its keep: the
structured-output schema in :mod:`prenomen.models`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .models import DictMatch, Gender, ReasoningCode
from .names import normalize_countries

# The flat column names emitted by GenderResult.to_dict, in order. The first six are
# the 1.x TSV's core columns, spelling included: `llm_confidence` and `dict_bypass`
# are inherited misnomers (on a bypassed row the confidence is the dictionary's, not
# a model's), kept so analysis scripts written against published output still read.
CORE_COLUMNS = (
    "first_name",
    "last_name",
    "gender",
    "llm_confidence",
    "reasoning_code",
    "dict_bypass",
    "cached",
    "error",
)


class Verdict(str, Enum):
    """What settled the gender."""

    DICTIONARY = "dictionary"  # the bypass gate was met; the model was never called
    MODEL = "model"  # the model answered
    ERROR = "error"  # no answer was obtained


@dataclass(frozen=True, slots=True)
class NameQuery:
    """One name to resolve, plus whatever you need to join the answer back.

    ``countries`` accepts a sequence of codes or the ``"FR;PL"`` string a table
    column holds; either way it is normalised to a sorted upper-case tuple.
    ``record`` is opaque: it is echoed on the result and never inspected.
    """

    first_name: str
    last_name: str = ""
    countries: tuple[str, ...] = ()
    record: Any = None

    def __post_init__(self):
        # A blank first name is the caller's mistake and cannot be inferred from, so
        # it is refused here -- synchronously, on their thread, before any request.
        if not isinstance(self.first_name, str) or not self.first_name.strip():
            raise ValueError(
                f"first_name must be a non-empty string, got {self.first_name!r}"
            )
        object.__setattr__(self, "first_name", self.first_name.strip())
        last = self.last_name if isinstance(self.last_name, str) else ""
        object.__setattr__(self, "last_name", last.strip())
        object.__setattr__(self, "countries", normalize_countries(self.countries))


@dataclass(slots=True)
class GenderResult:
    """The outcome of one inference.

    ``decided_by`` says what settled it; ``cached`` says whether a model answer was
    reused rather than requested. They are separate because they are separate facts:
    grouping by ``decided_by`` should not have to remember to fold cache hits in, and
    "how many requests did I pay for?" reads off ``cached`` alone.
    """

    first_name: str
    last_name: str
    gender: Gender
    confidence: float
    # None on an error. ReasoningCode is serialised into the request schema via
    # LLMResponse, so it must not gain an ERROR member: that would change what the
    # model is asked, and the prompt is frozen against the published paper.
    reasoning_code: ReasoningCode | None
    decided_by: Verdict
    cached: bool = False
    countries: tuple[str, ...] = ()
    dict_matches: tuple[DictMatch, ...] = ()
    # The weighted dictionary verdict, reported even when the model overrode it, so
    # model-vs-dictionary disagreement is readable without recomputing it.
    dict_gender: Gender | None = None
    dict_confidence: float | None = None
    record: Any = None
    error: str | None = None
    exception: BaseException | None = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        """False when no answer was obtained."""
        return self.error is None

    def raise_for_error(self) -> "GenderResult":
        """Return self, or re-raise the failure this result stands in for."""
        if self.exception is not None:
            raise self.exception
        if self.error is not None:
            raise RuntimeError(self.error)
        return self

    def to_dict(self, *, columns: Sequence[str] | None = None) -> dict:
        """Flatten to one row, using the 1.x TSV column names.

        A ``Mapping`` record is merged in first so its keys travel with the row; the
        core fields win on a collision, as ``extra_columns`` did. Any other record is
        left out, since there is no column name for it.

        Pass ``columns`` (from ``Prenomen.columns()``) to get exactly those keys,
        missing ones filled with ``""``, so a ``csv.DictWriter`` with fixed
        fieldnames never raises on an unexpected key.
        """
        row: dict[str, Any] = {}
        if isinstance(self.record, Mapping):
            row.update(self.record)
        row["first_name"] = self.first_name
        row["last_name"] = self.last_name
        row["gender"] = self.gender.value
        row["llm_confidence"] = self.confidence
        row["reasoning_code"] = (
            self.reasoning_code.value if self.reasoning_code is not None else ""
        )
        row["dict_bypass"] = self.decided_by is Verdict.DICTIONARY
        row["cached"] = self.cached
        row["error"] = self.error or ""
        for match in self.dict_matches:
            hit = match.found and match.gender is not None
            row[f"{match.source}_gender"] = match.gender.value if hit else ""
            row[f"{match.source}_freq"] = (
                f"{match.frequency:.4f}"
                if hit and match.frequency is not None
                else ""
            )
        if columns is not None:
            return {name: row.get(name, "") for name in columns}
        return row
