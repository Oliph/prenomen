"""Pydantic models for the dictionary matches and the model-facing schema.

``LLMResponse``, ``LLMAuthorResponse``, ``Gender`` and ``ReasoningCode`` are the
**wire schema**: they are serialised into the request as ``response_format``, so they
describe what the model is asked, not merely how the answer is stored. The prompt is
frozen against the published paper (see README, Method and provenance), and this
schema is the same contract one layer down -- adding a ``ReasoningCode`` member or
renaming a field changes the request and breaks comparability with those results.
An error is therefore represented outside this module, by
``GenderResult.reasoning_code is None``, rather than by a new enum member.

Caller-facing input and output records live in :mod:`prenomen.results`.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NOT_RETRIEVABLE = "not retrievable"


class ReasoningCode(str, Enum):
    EMPIRICAL_DATABASE = "empirical_database"
    CULTURAL_ORIGIN = "cultural_origin"
    LINGUISTIC_PATTERN = "linguistic_pattern"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_DATA = "insufficient_data"
    DICT_BYPASS = "dict_bypass"


class DictMatch(BaseModel):
    """Result from a single name-gender database lookup."""
    source: str
    gender: Optional[Gender] = None
    frequency: Optional[float] = None  # ratio of dominant gender among all occurrences (0-1)
    found: bool


class LLMAuthorResponse(BaseModel):
    """Schema sent to the LLM for structured output (the inner 'author' object)."""
    first_name: str
    last_name: str
    gender: Gender
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_code: ReasoningCode


class LLMResponse(BaseModel):
    """Top-level envelope matching the prompt's expected JSON structure."""
    author: LLMAuthorResponse
