"""The 1.x pipeline, rebuilt on top of the library: CSV in, TSV out, restartable.

Prenomen no longer reads or writes your data, and this is what that costs: about
twenty lines you control, in place of a column-mapping config file and a resume
protocol you did not.

    export OPENAI_API_KEY=sk-...
    python examples/batch.py

Kill it and run it again: it picks up where it stopped.
"""

import csv
import os

import pandas as pd

from prenomen import NameQuery, Prenomen

SOURCE = "examples/authors.csv"
OUTPUT = "authors_gender.tsv"

p = Prenomen(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
    dictionaries=[],  # e.g. ["wgnd"] once `prenomen-dicts get wgnd` has run
    max_workers=16,
    preload=True,  # pay the dictionary load now, not on the first name
)

# The output columns, including whichever dictionaries are loaded. Passing this to
# both DictWriter and to_dict keeps the header and the rows in step.
columns = p.columns(record_keys=["author_id", "countries"])

# Resume: the output file is the only record of what is done, so there is no index
# that can fall out of sync with it. Rows that failed are retried, because a failed
# row is written with an `error` and no gender.
done = set()
if os.path.exists(OUTPUT):
    with open(OUTPUT, newline="", encoding="utf-8") as fh:
        done = {
            row["author_id"] for row in csv.DictReader(fh, delimiter="\t")
            if not row["error"]
        }

frame = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
queries = (
    NameQuery(
        row.first_name,
        row.last_name,
        countries=row.countries,
        record={"author_id": row.author_id, "countries": row.countries},
    )
    for row in frame.itertuples()
    if row.author_id not in done
)

with open(OUTPUT, "a", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, columns, delimiter="\t")
    if fh.tell() == 0:
        writer.writeheader()
    for count, result in enumerate(p.infer_many(queries), start=1):
        writer.writerow(result.to_dict(columns=columns))
        if count % 10 == 0:
            fh.flush()
            print(f"  {count} done", flush=True)

print(f"{OUTPUT}: {sum(1 for _ in open(OUTPUT)) - 1} rows")

# In a notebook, where you want a frame rather than a file, it is one line:
#
#   results = pd.DataFrame(r.to_dict() for r in p.infer_many(queries))
#
# And to add a column to a frame you already have, infer_many keeps input order:
#
#   frame["gender"] = [r.gender.value for r in
#                      p.infer_many(zip(frame.first_name, frame.last_name))]
