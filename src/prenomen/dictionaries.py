"""The optional name-gender dictionaries: where they live, and how to fetch them.

The dictionaries are not shipped with this package. They are large, separately
published and separately citable datasets, and redistributing them here would put a
stale copy under the wrong licence. They are, however, all public deposits on Harvard
Dataverse under CC0, so one download path covers every one of them with no account,
token or click-through.

``prenomen-dicts list`` shows what is available and what is already on disk,
``prenomen-dicts path`` prints the directory they are read from and written to, and
``prenomen-dicts get wgnd`` (or ``get all``) fetches them there. Nothing edits any
config file: the library finds the files by the same directory convention, so a
download is immediately usable as ``Prenomen(dictionaries=["wgnd"], ...)``.

Each entry records the Dataverse file id rather than a browser URL: file ids are
stable across dataset versions, and the access API serves them without
authentication. ``size`` and ``columns`` are checked against what arrives, so a
truncated transfer or an upstream reshuffle is caught here rather than much later
inside pandas, with a less obvious message.

Each entry's ``reader`` block holds the settings the loaders in
:mod:`prenomen.dict_lookup` need. Keeping them here rather than in a config file
makes this registry the single source of truth for a shipped dictionary: the file id,
the expected columns and the way to read them cannot drift apart.
"""

import argparse
import gzip
import http.client
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from . import __version__

logger = logging.getLogger(__name__)

DATAVERSE_ACCESS = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"

# Where dictionaries live when nothing says otherwise. `data/dictionaries` relative
# to the working directory is where 1.x put them, so it is still honoured when it
# holds something; a fresh install gets the XDG location instead.
ENV_VAR = "PRENOMEN_DICT_DIR"
LEGACY_DIR = Path("data/dictionaries")

DICTIONARIES = {
    "wgnd": {
        "filename": "wgnd_2_0_name-gender-code.csv",
        "file_id": 4750348,
        # Dataverse ingested this one as a .tab; ask for the CSV that was uploaded.
        "original": True,
        "size": 91_229_769,
        "separator": ",",
        "columns": ["name", "code", "gender", "wgt"],
        "coverage": "3.8M distinct names, 195 countries and territories",
        "doi": "10.7910/DVN/MSEGSJ",
        "license": "CC0 1.0",
        "citation": (
            "Raffo, Julio (2021). WGND 2.0. Harvard Dataverse, V1. "
            "https://doi.org/10.7910/DVN/MSEGSJ"
        ),
        "reader": {
            "format": "wgnd",
            "family": "wgnd",
            "weight": 1.0,
            "name_col": "name",
            "gender_col": "gender",  # values M / F / ? / =
            "freq_col": "wgt",
            "country_col": "code",
            "country_aware": True,  # filter matches by the countries on the record
        },
    },
    "chinese_pinyin": {
        "filename": "PinyinGender.txt",
        "file_id": 10803451,
        "size": 2_714_013,
        "separator": "\t",
        "columns": ["PinyinName", "Male", "Female"],
        "coverage": "96,797 romanized Chinese given names",
        "doi": "10.7910/DVN/UAARYY",
        "license": "CC0 1.0",
        "citation": (
            "Shi, Dongbo and Tong, Sherry T. (2025). An Open Dataset of Chinese "
            "Name-to-Gender Associations for Gender Prediction in Broad Scientific "
            "Research. Harvard Dataverse, V2. https://doi.org/10.7910/DVN/UAARYY"
        ),
        "reader": {
            "format": "counts",
            "family": "chinese",  # shares one vote with chinese_cn
            "weight": 2.0,
            "name_col": "PinyinName",
            "male_col": "Male",
            "female_col": "Female",
        },
    },
    "chinese_cn": {
        "filename": "CnGender.txt",
        "file_id": 10803450,
        "size": 22_601_176,
        "separator": "\t",
        "columns": ["CnName", "Male", "Female"],
        "coverage": "1,051,891 Chinese-character given names",
        "doi": "10.7910/DVN/UAARYY",
        "license": "CC0 1.0",
        "citation": (
            "Shi, Dongbo and Tong, Sherry T. (2025). An Open Dataset of Chinese "
            "Name-to-Gender Associations for Gender Prediction in Broad Scientific "
            "Research. Harvard Dataverse, V2. https://doi.org/10.7910/DVN/UAARYY"
        ),
        "reader": {
            "format": "counts",
            "family": "chinese",
            "weight": 2.0,
            "name_col": "CnName",
            "male_col": "Male",
            "female_col": "Female",
        },
    },
    "naampy": {
        # Read straight from the gzip; pandas infers the compression from the suffix.
        "filename": "in_rolls_state_year_fn_naampy_x100.csv.gz",
        "file_id": 4965696,
        "size": 122_818_654,
        "separator": ",",
        "columns": ["first_name", "n_female", "n_male"],
        "coverage": "197,344 Indian given names, 31 states and union territories",
        "doi": "10.7910/DVN/WZGJBM",
        "license": "CC0 1.0",
        "citation": (
            "Sood, Gaurav and Laohaprapanon, Suriyan (2021). naampy. "
            "Harvard Dataverse, V3. https://doi.org/10.7910/DVN/WZGJBM"
        ),
        "reader": {
            "format": "counts",
            "family": "naampy",
            "weight": 2.0,
            "name_col": "first_name",
            "male_col": "n_male",
            "female_col": "n_female",
        },
    },
}

# Every shipped dictionary, in registry order. A tuple rather than a magic "all"
# string, so `dictionaries=ALL_DICTIONARIES` is typed and greppable.
ALL_DICTIONARIES = tuple(DICTIONARIES)


@dataclass(frozen=True)
class DictionarySpec:
    """One name-gender dictionary to load: where it is, and how to read it.

    There is no ``enabled`` flag. Not passing a dictionary is how you leave it out,
    which removes the "enabled but has no path" states the 1.x config allowed.

    ``family`` groups near-duplicate sources so they share one vote in the weighted
    confidence instead of compounding; ``weight`` sets reliability relative to a
    broad source. A specialist list covering one language or country is narrow but
    precise, so a hit there is stronger evidence and weighs double.
    """

    name: str
    path: str | os.PathLike
    format: Literal["wgnd", "counts"] = "counts"
    family: Optional[str] = None  # defaults to name
    weight: float = 1.0
    separator: str = ","
    name_col: Optional[str] = None
    gender_col: str = "gender"
    freq_col: str = "wgt"
    country_col: str = "code"
    country_filter: Optional[Sequence[str]] = None
    country_aware: bool = False
    male_col: str = "male"
    female_col: str = "female"

    def __post_init__(self):
        if self.format not in ("wgnd", "counts"):
            raise ValueError(
                f"Dictionary {self.name!r}: unknown format {self.format!r} "
                f"(supported: counts, wgnd)."
            )
        if self.name_col is None:
            raise ValueError(f"Dictionary {self.name!r}: name_col is required.")
        object.__setattr__(self, "path", str(Path(self.path).expanduser()))
        if self.family is None:
            object.__setattr__(self, "family", self.name)

    @property
    def required_columns(self) -> list[str]:
        """Columns the loader cannot do without, for the pre-load header check.

        ``freq_col`` and ``country_col`` are deliberately absent: the wgnd loader
        falls back to a weight of 1.0 and no country when they are missing.
        """
        if self.format == "wgnd":
            return [self.name_col, self.gender_col]
        return [self.name_col, self.male_col, self.female_col]

    def reader_cfg(self) -> dict[str, Any]:
        """The settings dict the loaders in :mod:`prenomen.dict_lookup` take."""
        return {
            "separator": self.separator,
            "name_col": self.name_col,
            "gender_col": self.gender_col,
            "freq_col": self.freq_col,
            "country_col": self.country_col,
            "country_filter": self.country_filter,
            "country_aware": self.country_aware,
            "male_col": self.male_col,
            "female_col": self.female_col,
        }


def dictionary_dir(explicit: str | os.PathLike | None = None) -> Path:
    """Where dictionaries are read from and downloaded to.

    First hit wins: the argument, then ``$PRENOMEN_DICT_DIR``, then ``./data/
    dictionaries`` when it already holds one of the shipped files (a 1.x checkout),
    then ``$XDG_DATA_HOME/prenomen/dictionaries``, falling back to
    ``~/.local/share``. `prenomen-dicts path` prints the answer, so which branch won
    is never a mystery.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return Path(from_env).expanduser()
    if any((LEGACY_DIR / e["filename"]).exists() for e in DICTIONARIES.values()):
        return LEGACY_DIR
    xdg = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(xdg).expanduser() / "prenomen" / "dictionaries"


def spec(name: str, directory: str | os.PathLike | None = None) -> DictionarySpec:
    """Build the :class:`DictionarySpec` for one shipped dictionary, by name."""
    entry = DICTIONARIES.get(name)
    if entry is None:
        raise ValueError(
            f"Unknown dictionary {name!r}. Shipped dictionaries: "
            f"{', '.join(DICTIONARIES)}. For your own list, pass a DictionarySpec."
        )
    return DictionarySpec(
        name=name,
        path=dictionary_dir(directory) / entry["filename"],
        separator=entry["separator"],
        **entry["reader"],
    )


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _url(entry: dict) -> str:
    url = DATAVERSE_ACCESS.format(file_id=entry["file_id"])
    return url + "?format=original" if entry.get("original") else url


def download(entry: dict, target: Path, force: bool = False) -> Path:
    """Fetch one dictionary to ``target``, resuming a partial download if present.

    The bytes land in a ``.part`` sidecar and are renamed on success, so an
    interrupted download never leaves a half file where the pipeline will read it.
    A ``.part`` left behind by an earlier attempt is continued with a range
    request; servers that ignore the range simply restart it.
    """
    if target.exists() and not force:
        logger.info(f"{target} already present, skipping (use --force to refetch)")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    expected = entry.get("size")

    start = part.stat().st_size if part.exists() else 0
    if expected and start >= expected:
        # A stale or complete-looking leftover; a fresh pass is cheaper than guessing.
        start = 0

    request = urllib.request.Request(
        _url(entry), headers={"User-Agent": f"prenomen/{__version__}"}
    )
    if start:
        request.add_header("Range", f"bytes={start}-")

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if start and response.status != 206:
                start = 0  # range ignored: the body is the whole file again
            length = response.headers.get("Content-Length")
            total = (int(length) + start) if length else expected
            if start:
                logger.info(f"Resuming {target.name} at {_human(start)}")
            else:
                logger.info(f"Downloading {target.name} ({_human(total or 0)})")
            with open(part, "ab" if start else "wb") as fh:
                _copy(response, fh, done=start, total=total)
    except urllib.error.HTTPError as e:
        raise ValueError(
            f"Harvard Dataverse returned HTTP {e.code} for {target.name}. "
            f"Check https://doi.org/{entry['doi']} in a browser, or retry later."
        ) from e
    except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
        raise ValueError(
            f"The download of {target.name} broke off ({e}). What arrived is kept: "
            f"run the same command again to resume from there."
        ) from e

    # Everything below validates the .part file. Only a file that passes is renamed
    # into place, so a bad download is never mistaken for an installed dictionary on
    # the next run, and never reaches the pipeline.
    got = part.stat().st_size
    if total and got != total:
        raise ValueError(
            f"{target.name} is incomplete: {_human(got)} of {_human(total)} arrived. "
            f"Run the same command again to resume from there."
        )
    try:
        check_header(part, entry)
    except ValueError:
        part.unlink(missing_ok=True)
        raise

    part.replace(target)
    logger.info(f"Saved {target} ({_human(target.stat().st_size)})")
    return target


def _copy(response, fh, done: int, total, chunk: int = 1 << 20) -> None:
    """Stream the body to ``fh``, reporting progress against ``total`` if known."""
    tty = sys.stderr.isatty()
    step = max((total or 0) // 20, chunk)  # ~5% between lines on a dumb terminal
    next_report = done + step
    while True:
        block = response.read(chunk)
        if not block:
            break
        fh.write(block)
        done += len(block)
        if tty:
            pct = f" ({done / total:.0%})" if total else ""
            print(f"\r  {_human(done)}{pct}", end="", file=sys.stderr, flush=True)
        elif done >= next_report:
            logger.info(f"  {_human(done)}" + (f" of {_human(total)}" if total else ""))
            next_report = done + step
    if tty:
        print(file=sys.stderr)


def check_header(path: Path, entry: dict) -> None:
    """Confirm the file starts with the columns the pipeline config expects.

    Reads ``path``, but decides on compression from the registry filename: the
    caller checks the ``.part`` sidecar, whose own suffix says nothing.
    """
    opener = gzip.open if entry["filename"].endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    found = [c.strip().strip('"') for c in first.rstrip("\r\n").split(entry["separator"])]
    missing = [c for c in entry["columns"] if c not in found]
    if missing:
        raise ValueError(
            f"{entry['filename']} does not look like the expected file: columns "
            f"{missing} are missing from its header {found}. The download was "
            f"discarded; run the command again. If it keeps happening, the upstream "
            f"dataset at https://doi.org/{entry['doi']} has changed shape."
        )



def _list(dest: Path) -> None:
    print(f"Name-gender dictionaries (all CC0, read from and saved to {dest}/)\n")
    for source, entry in DICTIONARIES.items():
        target = dest / entry["filename"]
        state = "installed" if target.exists() else "not downloaded"
        print(f"  {source:<16}{state:<17}{_human(entry['size']):>9}")
        print(f"    {entry['coverage']}")
        print(f"    https://doi.org/{entry['doi']}\n")
    print("Download one with:  prenomen-dicts get wgnd")
    print("Download them all:  prenomen-dicts get all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prenomen-dicts",
        description="Download the optional name-gender dictionaries.",
    )
    parser.add_argument(
        "command",
        choices=["list", "get", "path"],
        help="'list' shows what is available, 'get' downloads it, "
        "'path' prints the directory in use",
    )
    parser.add_argument(
        "names",
        nargs="*",
        help=f"Which dictionaries to get: 'all', or any of "
        f"{', '.join(DICTIONARIES)}",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=f"Where to put the files (default: the directory `path` prints, "
        f"overridable with ${ENV_VAR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if the file is already there",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dest = dictionary_dir(args.dest)

    if args.command == "path":
        print(dest)
        return
    if args.command == "list":
        _list(dest)
        return

    requested = args.names or ["all"]
    if "all" in requested:
        requested = list(DICTIONARIES)
    unknown = [n for n in requested if n not in DICTIONARIES]
    if unknown:
        raise ValueError(
            f"Unknown dictionary {unknown[0]!r}. Available: "
            f"{', '.join(DICTIONARIES)}, or 'all'."
        )

    installed = []
    for source in requested:
        entry = DICTIONARIES[source]
        download(entry, dest / entry["filename"], force=args.force)
        installed.append((source, entry))

    # No config file is edited: say what to pass instead. `dictionary_dir` resolves
    # to the same place on the next run, so a plain list of names is enough unless
    # --dest sent the files somewhere the default resolution will not look.
    names = ", ".join(f'"{s}"' for s, _ in installed)
    print("\nUse them with:")
    if dest != dictionary_dir(None):
        print(f'  export {ENV_VAR}="{dest}"')
    print(f"  Prenomen(model=..., base_url=..., dictionaries=[{names}])")

    print("\nPlease cite what you use:")
    for citation in dict.fromkeys(e["citation"] for _, e in installed):
        print(f"  {citation}")


def entrypoint() -> None:
    """Console-script wrapper: report the expected failures without a traceback."""
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except ValueError as e:
        logging.getLogger(__name__).error(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    entrypoint()
