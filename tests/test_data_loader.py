"""Tests for load_names and the column validation."""

import json
import os
import tempfile
import unittest

import pandas as pd
import pytest

from prenomen.data_loader import load_names, split_full_name, validate_columns


@pytest.mark.parametrize("extension", ["csv", "tsv", "json", "jsonl", "xlsx", "parquet"])
def test_text_fields_survive_every_reader(tmp_path, extension):
    if extension == "xlsx":
        pytest.importorskip("openpyxl")
    if extension == "parquet":
        pytest.importorskip("pyarrow")
    records = [
        {"id": value, "given": "NA", "family": None, "region": "NA"}
        for value in ("001", "1", "0", None)
    ]
    df = pd.DataFrame(records)
    path = tmp_path / f"names.{extension}"
    if extension in ("csv", "tsv"):
        df.to_csv(path, index=False, sep="\t" if extension == "tsv" else ",")
    elif extension == "json":
        path.write_text(json.dumps(records))
    elif extension == "jsonl":
        path.write_text("\n".join(json.dumps(row) for row in records))
    elif extension == "xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_parquet(path, index=False)
    config = {"data": {
        "name_columns": [{"first_name": "given", "last_name": "family"}],
        "id_column": "id", "country_column": "region", "extra_columns": ["id"],
    }}
    validate_columns(path, config)
    authors = list(load_names(path, config))
    assert [a.extra["id"] for a in authors] == ["001", "1", "0", ""]
    assert all(a.first_name == "NA" and a.last_name == "" for a in authors)
    assert all(a.extra["region"] == "NA" for a in authors)


def test_numeric_json_id_with_null_is_normalized(tmp_path):
    path = tmp_path / "names.json"
    path.write_text('[{"id": 0, "name": "Ada"}, {"id": null, "name": "Alan"}]')
    config = {"data": {
        "name_columns": [{"full_name": "name"}],
        "id_column": "id", "extra_columns": ["id"],
    }}
    assert [a.extra["id"] for a in load_names(path, config)] == ["0", ""]


def test_country_is_optional_unless_explicitly_configured(tmp_path):
    path = tmp_path / "names.csv"
    path.write_text("name\nAda Lovelace\n")
    config = {"data": {"name_columns": [{"full_name": "name"}]}}
    validate_columns(path, config)
    config["data"]["country_column"] = "region"
    with pytest.raises(ValueError, match="region"):
        validate_columns(path, config)


def test_default_country_is_loaded_without_extra_columns(tmp_path):
    path = tmp_path / "names.csv"
    path.write_text("name,countries\nAda Lovelace,NA\n")
    config = {"data": {"name_columns": [{"full_name": "name"}]}}
    assert list(load_names(path, config))[0].extra == {"countries": "NA"}


def _write(df: pd.DataFrame, tmpdir: str, name: str = "input.csv") -> str:
    path = os.path.join(tmpdir, name)
    df.to_csv(path, index=False)
    return path


def _load(df: pd.DataFrame, name_columns, extra_columns=None, id_column=None):
    """Write df to a temp CSV and return the AuthorInputs load_names yields."""
    with tempfile.TemporaryDirectory() as d:
        path = _write(df, d)
        config = {
            "data": {
                "name_columns": name_columns,
                "extra_columns": extra_columns or [],
                "id_column": id_column,
            }
        }
        return list(load_names(path, config))


class SplitFullNameTest(unittest.TestCase):
    def test_splits_on_last_whitespace(self):
        self.assertEqual(split_full_name("Ada Lovelace"), ("Ada", "Lovelace"))
        self.assertEqual(
            split_full_name("Santiago Ramon y Cajal"), ("Santiago Ramon y", "Cajal")
        )

    def test_cjk_name_is_surname_first(self):
        # No whitespace to split on; the leading character is the surname.
        self.assertEqual(split_full_name("屠呦呦"), ("呦呦", "屠"))

    def test_single_token_keeps_empty_last_name(self):
        self.assertEqual(split_full_name("Aristotle"), ("Aristotle", ""))

    def test_blank_is_unusable(self):
        self.assertIsNone(split_full_name("   "))
        self.assertIsNone(split_full_name(None))


class SeparateNameColumnsTest(unittest.TestCase):
    name_cols = [{"first_name": "first_name", "last_name": "last_name"}]

    def test_null_last_name_is_kept_as_empty(self):
        # Single-token names carry a null last name; they must still be loaded.
        df = pd.DataFrame(
            {
                "first_name": ["Yang", "Ada"],
                "last_name": [None, "Lovelace"],
                "author_id": ["A1", "A2"],
            }
        )
        authors = _load(df, self.name_cols, ["author_id"])
        self.assertEqual(len(authors), 2)
        by_id = {a.extra["author_id"]: a for a in authors}
        self.assertEqual(by_id["A1"].first_name, "Yang")
        self.assertEqual(by_id["A1"].last_name, "")
        self.assertEqual(by_id["A2"].last_name, "Lovelace")

    def test_missing_first_name_is_skipped(self):
        df = pd.DataFrame(
            {
                "first_name": [None, "  ", "Grace"],
                "last_name": ["X", "Y", "Hopper"],
                "author_id": ["A1", "A2", "A3"],
            }
        )
        authors = _load(df, self.name_cols, ["author_id"])
        self.assertEqual([a.first_name for a in authors], ["Grace"])

    def test_whitespace_is_stripped(self):
        df = pd.DataFrame({"first_name": [" Bai "], "last_name": [None]})
        authors = _load(df, self.name_cols)
        self.assertEqual(authors[0].first_name, "Bai")
        self.assertEqual(authors[0].last_name, "")


class ValidateColumnsTest(unittest.TestCase):
    name_cols = [{"first_name": "first_name", "last_name": "last_name"}]

    def _config(self, **data):
        return {"data": {"name_columns": self.name_cols, **data}}

    def test_missing_column_is_reported(self):
        df = pd.DataFrame({"first_name": ["Ada"], "last_name": ["Lovelace"]})
        with tempfile.TemporaryDirectory() as d:
            path = _write(df, d)
            with self.assertRaises(ValueError) as ctx:
                validate_columns(path, self._config(extra_columns=["countries"]))
            self.assertIn("countries", str(ctx.exception))

    def test_id_column_must_be_carried_through(self):
        # An id_column absent from extra_columns would never reach the output, so
        # resume would find nothing to match on.
        df = pd.DataFrame(
            {"first_name": ["Ada"], "last_name": ["Lovelace"], "author_id": ["A1"]}
        )
        with tempfile.TemporaryDirectory() as d:
            path = _write(df, d)
            with self.assertRaises(ValueError) as ctx:
                validate_columns(path, self._config(id_column="author_id"))
            self.assertIn("extra_columns", str(ctx.exception))

    def test_valid_config_passes(self):
        df = pd.DataFrame(
            {"first_name": ["Ada"], "last_name": ["Lovelace"], "author_id": ["A1"]}
        )
        with tempfile.TemporaryDirectory() as d:
            path = _write(df, d)
            validate_columns(
                path, self._config(id_column="author_id", extra_columns=["author_id"])
            )


if __name__ == "__main__":
    unittest.main()


# -- name_columns shape ----------------------------------------------------
# The one block every new dataset hand-edits, so the likeliest place for a typo.


@pytest.mark.parametrize("mappings,expected", [
    ([{"first_name": "a"}], "must name either"),
    ([{"last_name": "b"}], "must name either"),
    ([{"firstname": "a", "last_name": "b"}], "must name either"),
    ([{"first_name": "a", "last_name": "b"}, {"given": "c"}], r"\[1\]"),
    ({"first_name": "a", "last_name": "b"}, "must be a list"),
    ("first_name", "must be a list"),
    ([["first_name", "a"]], "must be a mapping"),
])
def test_a_malformed_name_mapping_is_reported_with_an_example(mappings, expected):
    from prenomen.data_loader import name_mappings

    with pytest.raises(ValueError, match=expected) as excinfo:
        name_mappings({"name_columns": mappings})
    assert "full_name" in str(excinfo.value)  # the message shows both shapes


@pytest.mark.parametrize("mappings", [
    [{"full_name": "name"}],
    [{"first_name": "a", "last_name": "b"}],
    [{"first_name": "a", "last_name": "b"}, {"full_name": "c"}],
    [{"full_name": "name", "first_name": "a"}],  # full_name wins, as load_names does
])
def test_accepted_name_mapping_shapes_pass(mappings):
    from prenomen.data_loader import name_mappings

    assert name_mappings({"name_columns": mappings}) == mappings


def test_a_malformed_mapping_is_caught_before_the_file_is_read(tmp_path):
    # validate_columns runs first precisely so nothing expensive happens on a typo.
    config = {"data": {"name_columns": [{"first_name": "given"}]}}
    with pytest.raises(ValueError, match="must name either"):
        validate_columns(tmp_path / "does-not-exist.csv", config)


def test_a_missing_input_file_is_reported_not_traced(tmp_path):
    config = {"data": {"name_columns": [{"full_name": "name"}]}}
    with pytest.raises(ValueError, match="Input file not found"):
        validate_columns(tmp_path / "nope.csv", config)
