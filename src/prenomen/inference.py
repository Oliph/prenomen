"""Infer gender for one name, conditioned on name-gender dictionaries.

A name is resolved in one of three ways:

1. **Dictionary bypass.** When the matched dictionaries alone are conclusive
   (weighted confidence at or above ``dict_bypass_threshold``), the verdict is
   accepted without a model call.
2. **Cache.** When the same name and country context was already answered by the
   model on this object, that answer is reused. Concurrent callers asking for the
   same context share one request rather than making several.
3. **Model call.** Otherwise the dictionary matches are rendered into the prompt and
   the model arbitrates, returning structured JSON.

Everything is synchronous. There is no event loop, which is what lets ``infer`` be
called from a Jupyter cell, from inside ``asyncio.run``, or from any thread without
special handling; ``infer_many`` gets its concurrency from a thread pool, since the
one thing here that waits is a single HTTP request.

The work is deliberately split into *prepare* (dictionary lookup, bypass gate, prompt
render -- pure and fast), *call* (the one blocking request) and *finish* (assembling
the result). An async variant would reuse prepare and finish untouched and differ
only in the middle step.
"""

import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from .client import build_client, resolve_api_key
from .config import config_to_kwargs, load_yaml
from .confidence import source_meta_from_specs, weighted_dict_confidence
from .dict_lookup import MultiDBLookup
from .dictionaries import DictionarySpec, spec as builtin_spec
from .models import DictMatch, Gender, LLMAuthorResponse, LLMResponse, ReasoningCode
from .prompt import build_prompt_template
from .results import CORE_COLUMNS, GenderResult, NameQuery, Verdict

logger = logging.getLogger(__name__)

_QUERY_KEYS = frozenset({"first_name", "last_name", "countries", "record"})


def as_query(item) -> NameQuery:
    """Coerce one ``infer_many`` input into a :class:`NameQuery`.

    A bare string is refused rather than guessed at: whether ``"Marie Curie"`` is a
    first name or a full name is exactly the kind of thing that fails silently on the
    rows where it matters.
    """
    if isinstance(item, NameQuery):
        return item
    if isinstance(item, Mapping):
        unknown = sorted(set(item) - _QUERY_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown key(s) {unknown} in a query; accepted: "
                f"{sorted(_QUERY_KEYS)}."
            )
        return NameQuery(**item)
    if isinstance(item, str):
        raise ValueError(
            f"A bare string is ambiguous: {item!r} could be a first name or a whole "
            f'name. Pass ("first", "last"), or prenomen.split_full_name({item!r}).'
        )
    if isinstance(item, (tuple, list)):
        if not 1 <= len(item) <= 2:
            raise ValueError(
                f"A query tuple must be (first_name,) or (first_name, last_name); "
                f"got {len(item)} items."
            )
        return NameQuery(*item)
    raise ValueError(
        f"Cannot read a query from {type(item).__name__}. Pass a NameQuery, a "
        f'("first", "last") tuple, or a mapping of {sorted(_QUERY_KEYS)}.'
    )


def _try_dict_bypass(
    dict_matches: Sequence[DictMatch],
    query: NameQuery,
    threshold: float,
    source_meta: dict | None = None,
) -> GenderResult | None:
    """Return a finished result when the dictionaries alone settle the name.

    Uses the reliability-weighted dictionary confidence, a bounded estimate of
    P(dominant gender | name). A single source reduces to its own frequency ratio,
    so the threshold reads as "frequency ratio at least this". Genuine conflict
    between sources falls below any sane threshold and is routed to the model.
    Returns None when nothing matched or the confidence is below the threshold.
    """
    result = weighted_dict_confidence(dict_matches, source_meta)
    if result is None:
        return None
    dominant, dict_conf = result
    # Never bypass on an abstention. No shipped dictionary can emit one, so this is
    # a guard for dictionaries added later.
    if dominant not in (Gender.MALE, Gender.FEMALE):
        return None
    dict_conf = round(dict_conf, 4)  # gate on the value we store, avoiding 0.84999
    if dict_conf < threshold:
        return None
    return GenderResult(
        first_name=query.first_name,
        last_name=query.last_name,
        gender=dominant,
        confidence=dict_conf,
        reasoning_code=ReasoningCode.DICT_BYPASS,
        decided_by=Verdict.DICTIONARY,
        countries=query.countries,
        dict_matches=tuple(dict_matches),
        dict_gender=dominant,
        dict_confidence=dict_conf,
        record=query.record,
    )


class Prenomen:
    """A loaded inference engine: dictionaries, endpoint, prompt template and cache.

    Building one is expensive -- all four shipped dictionaries cost roughly 6.8 GB of
    memory and several minutes to parse -- so build one per process and reuse it for
    every name. Safe to share across threads. It cannot be pickled; for Spark or
    joblib, build one per worker process instead.

        p = Prenomen(model="gpt-4o-mini", base_url="https://api.openai.com/v1",
                     api_key_env="OPENAI_API_KEY", dictionaries=["wgnd"])
        p.infer("Marie", "Curie", countries=["FR"]).gender

    Construction validates cheaply and immediately -- every dictionary file must
    exist and carry the configured columns -- but defers parsing until the first
    query, or until :meth:`load` if you want to pay it up front.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        dictionaries: Sequence[str | DictionarySpec] = (),
        dictionary_dir: str | os.PathLike | None = None,
        dict_bypass_threshold: float = 0.85,
        parameters: Mapping[str, Any] | None = None,
        max_workers: int = 16,
        timeout: float = 120.0,
        max_retries: int = 2,
        cache_size: int | None = 100_000,
        template_dir: str | os.PathLike | None = None,
        preload: bool = False,
        client: Any = None,
    ) -> None:
        if not model or not isinstance(model, str):
            raise ValueError("model must be the model name to call, as a string.")
        # isinstance(True, int) is True, and one worker "True" would be a silent
        # misreading of a config value.
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError(
                f"max_workers must be a whole number of threads, at least 1; "
                f"found {max_workers!r}"
            )
        if not isinstance(dict_bypass_threshold, (int, float)) or isinstance(
            dict_bypass_threshold, bool
        ):
            raise ValueError(
                f"dict_bypass_threshold must be a number between 0 and 1; "
                f"found {dict_bypass_threshold!r}"
            )
        if not 0.0 <= float(dict_bypass_threshold) <= 1.0:
            raise ValueError(
                f"dict_bypass_threshold must be between 0 and 1 (0 disables the "
                f"bypass); found {dict_bypass_threshold!r}"
            )
        if cache_size is not None and (
            isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0
        ):
            raise ValueError(
                f"cache_size must be a whole number of entries, or None for "
                f"unbounded; found {cache_size!r}"
            )

        self._model = model
        self._parameters = dict(parameters or {})
        self._max_workers = max_workers
        self._threshold = float(dict_bypass_threshold)
        self._cache_size = cache_size

        self._specs = [self._as_spec(d, dictionary_dir) for d in dictionaries]
        self._lookup = MultiDBLookup(self._specs, preload=preload)
        self._source_meta = source_meta_from_specs(self._specs)
        if self._specs:
            logger.info(f"Dictionaries: {', '.join(self.sources)}")
        else:
            logger.warning("No name-gender dictionaries: every name goes to the model.")
        if self._threshold > 0.0 and self._specs:
            logger.info(f"Dictionary bypass enabled (threshold={self._threshold:.2f})")

        self._template = build_prompt_template(template_dir)
        self._client = client if client is not None else build_client(
            base_url,
            resolve_api_key(api_key, api_key_env),
            timeout=timeout,
            max_retries=max_retries,
        )

        # One lock guards both the cache and the in-flight map, so checking the cache
        # and joining an in-flight request is one atomic step. It is never held across
        # the request itself.
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple, LLMAuthorResponse] = OrderedDict()
        self._in_flight: dict[tuple, Future] = {}

    @classmethod
    def from_config(cls, path, **overrides) -> "Prenomen":
        """Build one from a YAML config; keyword arguments override the file.

        See ``examples/prenomen.yaml``. Every setting is also a plain constructor
        argument, so a config file is a convenience, never a requirement.
        """
        kwargs = config_to_kwargs(load_yaml(path), str(path))
        kwargs.update(overrides)
        return cls(**kwargs)

    @staticmethod
    def _as_spec(item, directory) -> DictionarySpec:
        if isinstance(item, DictionarySpec):
            return item
        if isinstance(item, str):
            return builtin_spec(item, directory)
        raise ValueError(
            f"A dictionary must be the name of a shipped one or a DictionarySpec; "
            f"got {type(item).__name__}."
        )

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> "Prenomen":
        """Parse the dictionaries now, on this thread. Idempotent."""
        self._lookup.load()
        return self

    def close(self) -> None:
        """Close the HTTP client. The object is not usable for requests afterwards."""
        closer = getattr(self._client, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "Prenomen":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __getstate__(self):
        raise TypeError(
            "Prenomen holds an open HTTP client and gigabytes of dictionary indexes, "
            "so it cannot be pickled. Build one per worker process instead: in a "
            "Spark mapPartitions or a joblib worker, construct it inside the "
            "function, or keep one in a module-level global per process."
        )

    # -- introspection -----------------------------------------------------

    @property
    def sources(self) -> list[str]:
        """Dictionary names, in the order given. Fixes the dictionary column order."""
        return self._lookup.loaded_sources

    @property
    def cache_size(self) -> int:
        """How many model answers are currently remembered."""
        with self._lock:
            return len(self._cache)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def columns(self, record_keys: Sequence[str] = ()) -> list[str]:
        """The flat column names ``GenderResult.to_dict`` produces, in order.

        Pass the keys of your ``record`` mappings to get a complete header for a
        ``csv.DictWriter``. The dictionary columns depend on which dictionaries are
        loaded, so this is derived rather than fixed.
        """
        cols = [*record_keys, *CORE_COLUMNS]
        for source in self.sources:
            cols += [f"{source}_gender", f"{source}_freq"]
        seen = set()
        clashing = sorted({c for c in cols if c in seen or seen.add(c)})
        if clashing:
            raise ValueError(
                f"Output columns must be unique; {clashing} appear(s) twice. Check "
                f"record_keys against the result fields and dictionary names."
            )
        return cols

    def lookup(
        self,
        first_name: str,
        last_name: str = "",
        *,
        countries: Sequence[str] | str | None = None,
    ) -> list[DictMatch]:
        """Query the dictionaries only, with no model call. One match per source."""
        query = NameQuery(first_name, last_name, countries)
        return self._lookup.query(query.first_name, query.last_name, query.countries)

    # -- the unit of work --------------------------------------------------

    def infer(
        self,
        first_name: str,
        last_name: str = "",
        *,
        countries: Sequence[str] | str | None = None,
        record: Any = None,
    ) -> GenderResult:
        """Resolve one name.

        ``countries`` takes a sequence of codes or the ``"FR;PL"`` string a table
        column holds. ``record`` is echoed back on the result untouched, so results
        join to your data. A request failure is reported as a result with
        ``decided_by=Verdict.ERROR``, not raised; a mistake you can fix -- a blank
        name, a missing dictionary -- is raised as ``ValueError``.
        """
        return self._resolve(NameQuery(first_name, last_name, countries, record))

    def infer_many(
        self,
        queries: Iterable[NameQuery | Mapping | tuple],
        *,
        ordered: bool = True,
    ) -> Iterator[GenderResult]:
        """Resolve many names concurrently, yielding one result per input.

        Accepts ``NameQuery`` objects, ``(first, last)`` tuples, or mappings. The
        input may be a lazy generator over millions of rows: submission runs in a
        sliding window, so neither the inputs nor the results are all held at once,
        and a slow consumer applies backpressure rather than filling memory.

        ``ordered`` (the default) yields in input order, so
        ``df["gender"] = [r.gender for r in p.infer_many(...)]`` lines up. It costs
        head-of-line blocking: one slow name delays yielding, though not the other
        requests. Pass ``ordered=False`` to yield as answers arrive and join on
        ``record`` instead.
        """
        self.load()
        pool = ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix="prenomen"
        )
        window = self._max_workers * 4
        pending: list[Future] = []
        try:
            for item in queries:
                pending.append(pool.submit(self._resolve, as_query(item)))
                if len(pending) >= window:
                    yield _take(pending, ordered)
            while pending:
                yield _take(pending, ordered)
        finally:
            # A consumer that stops early (a break, an exception, an abandoned
            # generator) must not leave a window of requests in flight.
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)

    # -- resolution --------------------------------------------------------

    def _resolve(self, query: NameQuery) -> GenderResult:
        """Resolve one query. Never raises: a failure becomes an ERROR result."""
        try:
            matches = self._lookup.query(
                query.first_name, query.last_name, query.countries
            )
            scored = weighted_dict_confidence(matches, self._source_meta)
            dict_gender, dict_conf = scored if scored else (None, None)
            if dict_conf is not None:
                dict_conf = round(dict_conf, 4)

            if self._threshold > 0.0:
                bypass = _try_dict_bypass(
                    matches, query, self._threshold, self._source_meta
                )
                if bypass is not None:
                    return bypass

            key = (query.first_name, query.last_name, query.countries)
            answer, cached, error, exc = self._answer(key, query, matches)
            if answer is None:
                return GenderResult(
                    first_name=query.first_name,
                    last_name=query.last_name,
                    gender=Gender.NOT_RETRIEVABLE,
                    confidence=0.0,
                    reasoning_code=None,
                    decided_by=Verdict.ERROR,
                    countries=query.countries,
                    dict_matches=tuple(matches),
                    dict_gender=dict_gender,
                    dict_confidence=dict_conf,
                    record=query.record,
                    error=error,
                    exception=exc,
                )
            # The name columns carry the name as given, not the model's echo of it: a
            # model that transliterates or normalises it would otherwise make the
            # result fail to join back to the caller's row.
            return GenderResult(
                first_name=query.first_name,
                last_name=query.last_name,
                gender=answer.gender,
                confidence=answer.confidence,
                reasoning_code=answer.reasoning_code,
                decided_by=Verdict.MODEL,
                cached=cached,
                countries=query.countries,
                dict_matches=tuple(matches),
                dict_gender=dict_gender,
                dict_confidence=dict_conf,
                record=query.record,
            )
        except Exception as exc:  # pragma: no cover - the catch-all of last resort
            logger.exception(
                f"Unhandled error for {query.first_name} {query.last_name}"
            )
            return GenderResult(
                first_name=query.first_name,
                last_name=query.last_name,
                gender=Gender.NOT_RETRIEVABLE,
                confidence=0.0,
                reasoning_code=None,
                decided_by=Verdict.ERROR,
                countries=query.countries,
                record=query.record,
                error=f"{type(exc).__name__}: {exc}",
                exception=exc,
            )

    def _answer(self, key, query, matches):
        """One model answer for this context, shared with any concurrent caller.

        Returns ``(answer, cached, error, exception)``. Only successes are cached, and
        the shared slot is cleared either way, so a failure is retried next time
        rather than remembered.
        """
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit, True, None, None
            shared = self._in_flight.get(key)
            owner = shared is None
            if owner:
                shared = Future()
                self._in_flight[key] = shared
        if not owner:
            return shared.result()

        try:
            answer, error, exc = self._call_model(query, matches)
            outcome = (answer, False, error, exc)
            with self._lock:
                if answer is not None and self._cache_size != 0:
                    self._cache[key] = answer
                    self._cache.move_to_end(key)
                    if self._cache_size is not None:
                        while len(self._cache) > self._cache_size:
                            self._cache.popitem(last=False)
                if self._in_flight.get(key) is shared:
                    del self._in_flight[key]
            shared.set_result(outcome)
            return outcome
        except BaseException as exc:
            # A waiter blocked on this future must never hang because the owner died.
            with self._lock:
                if self._in_flight.get(key) is shared:
                    del self._in_flight[key]
            outcome = (None, False, f"{type(exc).__name__}: {exc}", exc)
            if not shared.done():
                shared.set_result(outcome)
            raise

    def _call_model(self, query: NameQuery, matches):
        """The one blocking step. Returns ``(answer, error, exception)``."""
        prompt = self._template.render(
            first_name=query.first_name,
            last_name=query.last_name,
            dict_matches=[m for m in matches if m.found],
            countries=query.countries,
        )
        name = f"{query.first_name} {query.last_name}".strip()
        try:
            response = self._client.chat.completions.parse(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format=LLMResponse,
                **self._parameters,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                logger.warning(f"Null response for {name}")
                return None, "the endpoint returned no parsable answer", None
            return parsed.author, None, None
        except Exception as exc:
            raw = None
            completion = getattr(exc, "completion", None)
            if completion is not None:
                try:
                    raw = completion.choices[0].message.content
                except Exception:
                    pass
            if raw:
                preview = (
                    raw if len(raw) <= 500
                    else f"{raw[:250]}...[truncated]...{raw[-250:]}"
                )
                logger.error(
                    f"Error for {name}: {exc} | raw[{len(raw)} chars]: {preview!r}"
                )
            else:
                logger.error(f"Error for {name}: {exc}")
            return None, f"{type(exc).__name__}: {exc}", exc


def _take(pending: list[Future], ordered: bool) -> GenderResult:
    """Pop and return one finished result, in input order or as they arrive."""
    if ordered:
        return pending.pop(0).result()
    done, _ = wait(pending, return_when=FIRST_COMPLETED)
    future = next(f for f in pending if f in done)
    pending.remove(future)
    return future.result()
