"""Prenomen: hybrid dictionary + LLM gender inference from personal names.

    from prenomen import Prenomen

    p = Prenomen(model="gpt-4o-mini", base_url="https://api.openai.com/v1",
                 api_key_env="OPENAI_API_KEY", dictionaries=["wgnd"])
    print(p.infer("Marie", "Curie", countries=["FR"]).gender)

One name in, one result out. Reading tables, iterating them and writing results are
the caller's; see ``examples/batch.py``. Building a Prenomen is expensive, so build
one per process and reuse it.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # One source of truth: the version declared in pyproject.toml, read back from
    # the installed distribution rather than restated here where it would drift.
    __version__ = version("prenomen")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"

# Names resolved on first access rather than imported here, so `import prenomen`
# costs nothing: `Prenomen` pulls in openai, jinja2 and pandas (about a second and a
# half), which a caller who only wants `split_full_name` or the dictionary
# downloader should not pay.
_LAZY = {
    "Prenomen": ".inference",
    "DictionarySpec": ".dict_lookup",
    "ALL_DICTIONARIES": ".dictionaries",
    "dictionary_dir": ".dictionaries",
}

from .models import DictMatch, Gender, ReasoningCode
from .names import split_full_name
from .results import GenderResult, NameQuery, Verdict

__all__ = [
    "ALL_DICTIONARIES",
    "DictMatch",
    "DictionarySpec",
    "Gender",
    "GenderResult",
    "NameQuery",
    "Prenomen",
    "ReasoningCode",
    "Verdict",
    "__version__",
    "dictionary_dir",
    "split_full_name",
]


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)
