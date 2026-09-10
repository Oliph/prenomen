"""Tests for the dictionary downloader: the registry, the header check, and the
in-place config edit. Nothing here touches the network."""

import gzip

import pytest
import yaml

from prenomen.dictionaries import (
    DICTIONARIES,
    check_header,
    enable_in_config,
)

CONFIG = """\
# leading comment
dictionaries:
  # World Gender Name Dictionary.
  wgnd:
    enabled: false
    path: "data/dictionaries/wgnd.csv"
    format: wgnd
    weight: 1.0
    gender_col: "gender" # values M / F
  naampy:
    enabled: false
    path: "data/dictionaries/naampy.csv"
    format: counts
"""


def test_registry_matches_the_shipped_pipeline_config():
    """Every registry key names a real block, and the default paths agree."""
    with open("config/pipeline.yaml") as fh:
        blocks = yaml.safe_load(fh)["dictionaries"]
    assert set(DICTIONARIES) == set(blocks)
    for source, entry in DICTIONARIES.items():
        assert blocks[source]["path"].endswith(entry["filename"]), source


def test_registry_columns_are_the_ones_the_config_reads():
    """A download that passes the header check must satisfy the config's columns."""
    with open("config/pipeline.yaml") as fh:
        blocks = yaml.safe_load(fh)["dictionaries"]
    for source, entry in DICTIONARIES.items():
        cfg = blocks[source]
        keys = ("name_col", "gender_col", "freq_col", "country_col", "male_col",
                "female_col")
        for key in keys:
            if cfg.get(key):
                assert cfg[key] in entry["columns"], f"{source}.{key}"


def test_enable_flips_one_block_and_keeps_every_comment(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(CONFIG)

    assert enable_in_config(path, "naampy", tmp_path / "n.csv.gz") is True

    parsed = yaml.safe_load(path.read_text())["dictionaries"]
    assert parsed["naampy"]["enabled"] is True
    assert parsed["naampy"]["path"] == str(tmp_path / "n.csv.gz")
    # The other block and the comments are untouched.
    assert parsed["wgnd"]["enabled"] is False
    assert parsed["wgnd"]["path"] == "data/dictionaries/wgnd.csv"
    assert "# World Gender Name Dictionary." in path.read_text()
    assert "# values M / F" in path.read_text()
    assert "# leading comment" in path.read_text()


def test_enable_refuses_rather_than_mangling_an_unexpected_config(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text("dictionaries:\n  wgnd:\n    format: wgnd\n")  # no enabled/path
    before = path.read_text()

    assert enable_in_config(path, "wgnd", tmp_path / "w.csv") is False
    assert enable_in_config(path, "absent", tmp_path / "w.csv") is False
    assert path.read_text() == before


def test_enable_reports_a_missing_config_instead_of_creating_one(tmp_path):
    path = tmp_path / "nope.yaml"
    assert enable_in_config(path, "wgnd", tmp_path / "w.csv") is False
    assert not path.exists()


@pytest.mark.parametrize(
    "source,header",
    [
        ("wgnd", "name,code,gender,wgt\nmarie,FR,F,1\n"),
        ("chinese_pinyin", "PinyinName\tMale\tFemale\tRatio\njianmin\t18462\t2089\t0.9\n"),
        ("naampy", "state,birth_year,first_name,n_female,n_male\nbihar,1990,priya,7,0\n"),
    ],
)
def test_header_check_accepts_the_real_shapes(tmp_path, source, header):
    """Compression is decided by the registry filename, not the path being read:
    the caller checks a `.part` sidecar whose own suffix says nothing."""
    entry = DICTIONARIES[source]
    path = tmp_path / (entry["filename"] + ".part")
    if entry["filename"].endswith(".gz"):
        path.write_bytes(gzip.compress(header.encode()))
    else:
        path.write_text(header)
    check_header(path, entry)  # does not raise


def test_header_check_names_the_missing_columns(tmp_path):
    path = tmp_path / "wgnd.csv"
    path.write_text("name,country,gender\nmarie,FR,F\n")
    with pytest.raises(ValueError) as excinfo:
        check_header(path, DICTIONARIES["wgnd"])
    assert "'code'" in str(excinfo.value)
    assert "'wgt'" in str(excinfo.value)


def test_header_check_catches_a_truncated_html_error_page(tmp_path):
    """A proxy or outage can hand back HTML with a 200; it must not reach pandas."""
    path = tmp_path / "PinyinGender.txt"
    path.write_text("<!DOCTYPE html><html><head><title>502</title></head>\n")
    with pytest.raises(ValueError, match="does not look like the expected file"):
        check_header(path, DICTIONARIES["chinese_pinyin"])
