"""Output validation must reject corruption without modifying the source of truth."""

import csv
import io

import pytest

from prenomen.pipeline import (
    CORE_COLUMNS,
    _record_key,
    check_output_header,
    load_done,
)


def output_text(rows, extras=("id", "countries")):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t")
    writer.writerow([*extras, *CORE_COLUMNS])
    writer.writerows(rows)
    return stream.getvalue()


VALID_ROW = ["0", "NA", "Ada", "", "female", "0.8", "cultural_origin", "False"]


@pytest.mark.parametrize("column,value", [
    (2, ""), (4, "unknown"), (5, ""), (5, "nan"), (5, "inf"),
    (5, "-0.1"), (5, "1.1"), (6, "unknown"), (7, "true"), (7, ""),
])
def test_invalid_result_is_rejected(tmp_path, column, value):
    bad = VALID_ROW.copy()
    bad[column] = value
    path = tmp_path / "results.tsv"
    before = output_text([VALID_ROW, bad, VALID_ROW]).encode()
    path.write_bytes(before)
    with pytest.raises(ValueError, match="line 3"):
        load_done(str(path), "id")
    assert path.read_bytes() == before


@pytest.mark.parametrize("tail", [
    "A1\tGB\tAda\tLovelace\tfemale\t",  # partial result
    'A1\tGB\t"Ada\n',  # unterminated quote even though the file ends in newline
    'A1\tGB\t"Ada"x\tLovelace\tfemale\t0.8\tcultural_origin\tFalse\n',
    "\t".join(VALID_ROW) + "\textra\n",
    "\t".join(VALID_ROW),  # complete fields, missing final record boundary
])
def test_corrupt_tail_is_rejected_without_repair(tmp_path, tail):
    path = tmp_path / "results.tsv"
    before = (output_text([VALID_ROW]) + tail).encode()
    path.write_bytes(before)
    with pytest.raises(ValueError, match="line 3"):
        load_done(str(path), "id")
    assert path.read_bytes() == before


def test_quoted_fields_and_empty_optional_values_resume(tmp_path):
    path = tmp_path / "results.tsv"
    row = ["", "", "Ada\tMarie\nAnn", "", "female", "0.8", "cultural_origin", "False", "", ""]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t")
    writer.writerow(["id", "countries", *CORE_COLUMNS, "test_gender", "test_freq"])
    writer.writerow(row)
    path.write_bytes(stream.getvalue().encode())
    done, cache = load_done(str(path), "id")
    assert done == {_record_key({"id": ""}, row[2], "", "id")}
    assert (row[2], "", ()) in cache


def test_zero_id_and_country_context_survive_resume(tmp_path):
    path = tmp_path / "results.tsv"
    path.write_bytes(output_text([VALID_ROW]).encode())
    done, cache = load_done(str(path), "id")
    assert _record_key({"id": 0}, "Ada", "", "id") in done
    assert ("Ada", "", ("NA",)) in cache
    assert load_done(str(path), "id", collect_state=False) == (set(), {})


def test_output_without_countries_supplies_only_done_keys(tmp_path):
    path = tmp_path / "results.tsv"
    path.write_bytes(output_text([[VALID_ROW[0], *VALID_ROW[2:]]], extras=("id",)).encode())
    done, cache = load_done(str(path), "id")
    assert _record_key({"id": 0}, "Ada", "", "id") in done
    assert cache == {}


def test_bypassed_rows_are_valid_but_not_prediction_cache_entries(tmp_path):
    row = VALID_ROW.copy()
    row[-2:] = ["dict_bypass", "True"]
    path = tmp_path / "results.tsv"
    path.write_bytes(output_text([row]).encode())
    done, cache = load_done(str(path), "id")
    assert len(done) == 1
    assert cache == {}


def test_duplicate_output_columns_are_rejected(tmp_path):
    path = tmp_path / "results.tsv"
    header = ["first_name", *CORE_COLUMNS]
    with pytest.raises(ValueError, match="unique"):
        check_output_header(str(path), header)
    path.write_text("\t".join(header) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_done(str(path), None)
