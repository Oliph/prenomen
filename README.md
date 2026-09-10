# Prenomen

Infer gender from a personal name, combining name-gender dictionaries with a
language model. One name in, one result out:

```python
from prenomen import Prenomen

p = Prenomen(model="gpt-4o-mini", base_url="https://api.openai.com/v1",
             api_key_env="OPENAI_API_KEY", dictionaries=["wgnd"])

r = p.infer("Marie", "Curie", countries=["FR", "PL"])
r.gender          # <Gender.FEMALE: 'female'>
r.confidence      # 0.97
r.decided_by      # <Verdict.DICTIONARY: 'dictionary'> -- the model was never called
r.dict_matches    # the evidence, one entry per dictionary
```

Reading your data, iterating it and storing the answers are yours. Prenomen does
not know what a CSV is, and that is deliberate: every workflow already has a way to
do that — pandas, polars, Spark, an Airflow task, a notebook cell, a database
cursor — and none of them needs a library's opinion about it.

The method is hybrid on purpose. Dictionary lookups are fast, cheap and
transparent, but they miss unlisted names and cannot resolve a name that is
common to both genders in different regions. A language model handles those, but
is expensive to run over millions of records and works blind on rare names. So:

- Names the dictionaries settle conclusively never reach the model. On a large
  bibliographic corpus this was roughly 45% of records.
- Everything else goes to the model **with the dictionary evidence in the
  prompt**, so it arbitrates between sources rather than guessing unaided.
- Repeated names with the same country context share one request, including
  concurrent ones, and answers are remembered for the life of the object.

Dictionaries are optional. Without them every name goes to the model.

## Install

```bash
pip install prenomen
```

Python 3.10 or newer; tested on 3.10 and 3.14.

## Build it once

Constructing a `Prenomen` parses the dictionaries, which is the expensive part of
the whole library:

| Dictionary | Coverage | Download | RAM to load | Load time |
|---|---|---|---|---|
| [WGND 2.0](https://doi.org/10.7910/DVN/MSEGSJ) | 3.8M names, 195 countries and territories | 87 MB | 5.5 GB | 5 min |
| [PinyinGender](https://doi.org/10.7910/DVN/UAARYY) | 96,797 romanized Chinese given names | 2.6 MB | negligible | 1 s |
| [CnGender](https://doi.org/10.7910/DVN/UAARYY) | 1,051,891 Chinese-character given names | 22 MB | 1 GB | 3 s |
| [naampy](https://doi.org/10.7910/DVN/WZGJBM) | 197,344 Indian given names, 31 states | 117 MB | 2 GB | 9 s |

All four together measured 6.8 GB and a five-and-a-half-minute load, almost all of
it WGND. So **build one object per process and reuse it for every name.** On a
laptop, start with the specialist lists that match your corpus and add WGND when
you have the memory for it.

Construction validates immediately and cheaply — every file must exist and carry
the columns its reader needs, which takes milliseconds — and defers the parse until
the first name. Pass `preload=True` in a script or a worker, so the wait happens at
a predictable moment rather than mid-run.

It cannot be pickled, because it holds an open HTTP client and those gigabytes. For
process-parallel work, build one per worker:

```python
# Spark
def resolve(rows):
    p = Prenomen(model=..., base_url=..., dictionaries=["wgnd"], preload=True)
    return [p.infer(r.first, r.last).to_dict() for r in rows]

df.rdd.mapPartitions(resolve)

# joblib: one per process, built on first use
_engine = None
def engine():
    global _engine
    if _engine is None:
        _engine = Prenomen(model=..., base_url=..., dictionaries=["wgnd"])
    return _engine
```

## Many names

`infer_many` gives you bounded concurrency, request sharing and the cache without
writing any of it:

```python
results = p.infer_many([("Ada", "Lovelace"), ("Alan", "Turing")])
```

It takes `NameQuery` objects, `(first, last)` tuples, or mappings, and yields one
result per input **in input order**, so this lines up:

```python
frame["gender"] = [r.gender.value for r in
                   p.infer_many(zip(frame.first_name, frame.last_name))]
```

The input may be a lazy generator over millions of rows: submission runs in a
sliding window, so neither the inputs nor the results are all held at once, and a
slow consumer applies backpressure instead of filling memory. Ordering costs
head-of-line blocking — one slow name delays yielding, though not the other
in-flight requests — so pass `ordered=False` to take answers as they arrive and
join on `record` instead.

`record` is anything you want echoed back on the result, untouched: a row id, a
namedtuple, the whole source row. It is how results find their way home.

```python
queries = [NameQuery(r.first, r.last, countries=r.countries, record=r.id)
           for r in rows]
for r in p.infer_many(queries):
    store(r.record, r.gender)
```

To build a frame, `to_dict()` flattens a result to one row:

```python
import pandas as pd
pd.DataFrame(r.to_dict() for r in p.infer_many(queries))
```

See `examples/batch.py` for the 1.x pipeline rebuilt on this — CSV in, TSV out,
restartable — in about twenty lines.

### Async

Prenomen is synchronous, which is why `infer` and `infer_many` both work inside a
Jupyter cell, inside `asyncio.run`, and on any thread. From async code, hand it to
a thread:

```python
result = await asyncio.to_thread(p.infer, "Marie", "Curie")
```

That is real concurrency: the call is I/O-bound and releases the GIL.

## Results

```python
r = p.infer("Marie", "Curie", countries=["FR"])
```

| Field | Meaning |
|---|---|
| `first_name`, `last_name` | the name as you gave it, never the model's echo of it |
| `gender` | `Gender.MALE`, `Gender.FEMALE`, or `Gender.NOT_RETRIEVABLE` |
| `confidence` | 0 to 1; the model's own, or the dictionary confidence on a bypass |
| `reasoning_code` | why: `empirical_database`, `cultural_origin`, `linguistic_pattern`, `ambiguous`, `insufficient_data`, `dict_bypass`. `None` on a failure |
| `decided_by` | `Verdict.DICTIONARY`, `Verdict.MODEL`, or `Verdict.ERROR` |
| `cached` | a model answer was reused rather than requested |
| `countries` | as normalised: trimmed, upper-cased, deduplicated, sorted |
| `dict_matches` | one entry per loaded dictionary, matched or not, so you can audit the evidence |
| `dict_gender`, `dict_confidence` | what the dictionaries said, reported **even when the model overruled them** |
| `record` | whatever you passed in |
| `error`, `exception` | set only when `decided_by` is `ERROR` |
| `ok` | `False` when no answer was obtained |

`to_dict()` flattens this to the column names the 1.x TSV used, so analysis scripts
written against published output still read: `gender`, `llm_confidence`,
`reasoning_code`, `dict_bypass`, plus `<source>_gender` and `<source>_freq` per
dictionary, and `cached` and `error`. Two of those names are inherited misnomers —
`llm_confidence` carries the *dictionary* confidence on a bypassed row — and they
are kept anyway, because renaming the data buys nothing and costs comparability.

`p.columns(record_keys=["author_id"])` gives the matching header, including
whichever dictionaries are loaded, and refuses a `record` key that would collide
with a result field. Pass it to `to_dict(columns=...)` and a `csv.DictWriter` never
raises on an unexpected key.

### Failures are results, not exceptions

A timeout, an API error or an unparsable answer yields a result with
`decided_by=Verdict.ERROR`, `gender=NOT_RETRIEVABLE` and an `error` string. It is
not raised, because `infer_many` over a million names must not die on one 503, and
because you need a row to retry. Failures are never cached.

```python
for r in p.infer_many(queries):
    if not r.ok:
        retry_later(r.record, r.error)
```

Mistakes *you* can fix are raised as `ValueError`, synchronously, before any
request: a blank name, an unknown dictionary, a missing dictionary file, an unset
key variable, a bad threshold. `r.raise_for_error()` is there if you want the
exception after all.

### On resuming

Resuming is yours now, and there are two pieces of it. Model answers are remembered
for the life of the object, so within one run a repeated name is free. Across runs,
the output is the record of what is done — `examples/batch.py` reads back its own
TSV in six lines and skips what is already there, retrying only the rows that carry
an `error`.

## Dictionaries

Optional. They are not shipped with this package: they are large, separately
published, and each wants its own citation. A helper downloads them for you.

```bash
prenomen-dicts list          # what is available, and what you already have
prenomen-dicts path          # where they are read from and written to
prenomen-dicts get wgnd      # one of them
prenomen-dicts get all       # all four
```

Then name them, and nothing else:

```python
Prenomen(model=..., base_url=..., dictionaries=["wgnd", "naampy"])
Prenomen(model=..., base_url=..., dictionaries=prenomen.ALL_DICTIONARIES)
```

No config file is involved: the downloader and the library resolve the same
directory, in this order — the `dictionary_dir=` argument, `$PRENOMEN_DICT_DIR`,
`./data/dictionaries` if it already holds one of the files, then
`$XDG_DATA_HOME/prenomen/dictionaries`. `prenomen-dicts path` prints the answer, so
which one won is never a mystery. `--dest` puts files elsewhere, `--force`
refetches, and an interrupted download resumes where it stopped.

A dictionary that is named but missing, unreadable or misconfigured stops
construction, naming the source and the path. Leaving one out is how you turn it
off, never a missing file: a corpus-scale run that silently fell back to LLM-only
would be slower, less accurate and more expensive, with only a log line to show for
it.

All four are public CC0 deposits on Harvard Dataverse, so no account or token is
involved.

**Adding your own** is a `DictionarySpec` and no code, as long as the file is one
of two shapes:

```python
from prenomen import DictionarySpec

mine = DictionarySpec(name="my_list", path="~/names.tsv", format="counts",
                      separator="\t", name_col="name", male_col="m", female_col="f",
                      family="regional", weight=2.0)
Prenomen(model=..., base_url=..., dictionaries=["wgnd", mine])
```

- `format="counts"` — one row per name plus a male count and a female count, in
  any units. Rows repeating a name across regions or years are summed.
- `format="wgnd"` — one row per name, country and gender, plus a frequency weight.
  Set `country_aware=True` to filter by the countries on each record.

Column names are configurable for both. Two further keys control how much a
source is trusted:

- `family` groups near-duplicate sources, so two lists covering the same language
  share one vote instead of compounding against a broad source.
- `weight` sets reliability. A specialist list covering one language or country
  is narrow but precise, so a hit weighs double a global list.

### Confidence and the bypass gate

For each dictionary that matched, `p` is its frequency when it agrees with the
predicted gender and `1 − frequency` when it disagrees, where frequency is that
source's empirical P(gender | name). The dictionary confidence is the
reliability-weighted mean of those, bounded in [0, 1]:

```
confidence = Σ(wₛ · pₛ) / Σ wₛ
```

With one matching source this reduces to its own frequency ratio, so
`dict_bypass_threshold=0.85` reads as "at least 0.85 of occurrences of this name
carry this gender". Sources that disagree pull the score below the gate, and the
record goes to the model. A genuine tie scores 0.5 and never bypasses.

Lower the threshold to spend less on model calls, raise it to send more borderline
names to the model, set it to `0.0` to always call the model.

## Interpreting the output

`not retrievable` is an abstention, not a prediction. Score it as such, keeping
coverage and accuracy-on-answered separate.

A caution if you validate against known-correct labels. Gold labels derived from
names, including any name-gender dictionary, are circular here: the pipeline
already consumes those, so it shares their failure modes and will look better
than it is. A usable gold standard has to come from something independent of the
name, such as a curated biographical record or self-reported data.

**On the gender categories.** The model is constrained to a binary plus an
abstention. This reflects what the training signal and the dictionaries can
actually support, and it means the pipeline cannot represent non-binary people.
Anyone using it should say so in their methods, and should not treat its output
as a statement about identity. It estimates how a name is conventionally
gendered in a region, which is a different thing.

## Configuration

Every setting is a constructor argument. A YAML file is available for the ones you
want to keep and share:

```python
p = Prenomen.from_config("prenomen.yaml", max_workers=4)   # kwargs win
```

See `examples/prenomen.yaml`, which documents every key.

## Migrating from 1.x

1.x was a batch pipeline driven by three YAML files and a `prenomen` command. That
command is gone, along with the data config: there is nothing to map your columns
onto any more, because you do the reading.

| 1.x | 2.x |
|---|---|
| `prenomen --data-config d.yaml` | `p.infer(...)` / `p.infer_many(...)` in your own loop |
| `data.name_columns` | you read the table; `prenomen.split_full_name` if you have one name column |
| `data.extra_columns`, `data.id_column` | `record=` on each query, echoed on the result |
| `data.country_column` | `countries=` per call; takes a list or `"FR;PL"` |
| `output.dir`, the derived TSV path | you choose the path and the writer |
| `output_header()` | `p.columns(record_keys=[...])` |
| `processing.continue`, `--continue` | your own check against your own output; see `examples/batch.py` |
| `processing.max_concurrent` | `max_workers=` |
| `processing.dict_bypass_threshold` | `dict_bypass_threshold=` |
| `dictionaries:` blocks with `enabled`/`path` | `dictionaries=["wgnd", ...]`, or a `DictionarySpec` |
| `prenomen-dicts get --no-enable` | the default; nothing edits a config any more |
| `model.yaml` | `model=`, `base_url=`, `api_key_env=`, `parameters=` |
| a row missing from the output meant failure | a result with `decided_by=ERROR` and an `error` |
| `dict_bypass` column | `decided_by`, plus `cached`; still `dict_bypass` in `to_dict()` |

The prompt is unchanged, byte for byte, so results are comparable across the two.

## Method and provenance

This is the reusable extraction of the pipeline built for a methods paper on
large-scale gender inference in bibliometric data, where it was validated against
an independent gold standard of several hundred thousand authors. The validation
harness, the corpus-scale HPC job scripts and the paper itself are not part of
this repository, which keeps only what transfers to other datasets.

## Development

```bash
uv run --extra dev pytest -q                     # the whole suite, no network
uv run --python 3.10 --extra dev pytest -q       # the oldest supported interpreter
```

Tested on Python 3.10 and 3.14. Those two are what CI runs, and they are not the
same configuration: the lockfile resolves pandas 2 below Python 3.11 and pandas 3
above it, and the two differ in how they treat missing values. Run both before
releasing.

`tests/test_over_http.py` drives the real openai SDK against a loopback stub
endpoint, covering its retry path, timeout handling and connection reuse. It is the
only test that exercises the SDK itself, so it runs in CI rather than being skipped
as slow.

The prompt fixtures under `tests/fixtures/` pin the exact text sent to the model,
and `tests/test_prompt.py` checks them two ways: that the template renders them, and
that a live `infer` call still produces them. When a prompt change is intended,
regenerate them with `python -c "import tests.test_prompt as t; t.regenerate()"` and
review the diff — understanding that results from before and after are no longer
comparable.

## License

MIT. See `LICENSE`.

The dictionaries are **not** covered by it, and are not distributed with it.
The four `prenomen-dicts` knows about are all deposited under CC0 1.0, but they
are research datasets with papers behind them: cite them. `prenomen-dicts get`
prints the citation for everything it downloads. Check the terms yourself for any
dictionary you add, and before redistributing anything derived from one.
