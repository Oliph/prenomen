"""Tests for the dictionary loaders and the multi-source lookup.

The two loaders decide what evidence every record is scored against, and the
lookup decides the output column order, so both are pinned here against small
files written to tmp_path. No network, no real dictionary.
"""

import pytest

from prenomen.dict_lookup import MultiDBLookup, _load_counts, _load_wgnd
from prenomen.models import Gender


def counts_cfg(**over):
    cfg = {"separator": ",", "name_col": "name", "male_col": "male",
           "female_col": "female"}
    cfg.update(over)
    return cfg


def wgnd_cfg(**over):
    cfg = {"separator": ",", "name_col": "name", "gender_col": "gender",
           "freq_col": "wgt", "country_col": "code"}
    cfg.update(over)
    return cfg


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# -- counts loader --------------------------------------------------------


def test_rows_repeating_a_name_are_summed_not_overwritten(tmp_path):
    # Electoral-roll and census extracts carry one row per name per region or year.
    # Taking the last row instead of the sum would score a name on one region alone.
    path = write(tmp_path, "c.csv", "name,male,female\n"
                 "priya,1,9\npriya,1,9\npriya,8,2\n")
    index = _load_counts("s", path, counts_cfg())
    assert index["priya"].gender is Gender.FEMALE
    assert index["priya"].frequency == pytest.approx(20 / 30)


def test_frequency_is_the_dominant_share(tmp_path):
    path = write(tmp_path, "c.csv", "name,male,female\nalan,90,10\nmarie,5,95\n")
    index = _load_counts("s", path, counts_cfg())
    assert (index["alan"].gender, index["alan"].frequency) == (Gender.MALE, 0.9)
    assert (index["marie"].gender, index["marie"].frequency) == (Gender.FEMALE, 0.95)


def test_an_even_split_scores_half_and_cannot_bypass(tmp_path):
    path = write(tmp_path, "c.csv", "name,male,female\nandrea,50,50\n")
    index = _load_counts("s", path, counts_cfg())
    assert index["andrea"].frequency == 0.5


def test_names_with_no_occurrences_are_dropped(tmp_path):
    path = write(tmp_path, "c.csv", "name,male,female\nghost,0,0\nreal,1,0\n")
    index = _load_counts("s", path, counts_cfg())
    assert set(index) == {"real"}


def test_unparseable_counts_read_as_zero_rather_than_failing_the_load(tmp_path):
    path = write(tmp_path, "c.csv", "name,male,female\nalan,abc,10\n")
    index = _load_counts("s", path, counts_cfg())
    assert (index["alan"].gender, index["alan"].frequency) == (Gender.FEMALE, 1.0)


def test_blank_name_rows_never_become_the_name_nan(tmp_path):
    # Regression. pandas 2 renders a null name as the literal string "nan", which
    # would enter the index and answer for Nan, a real given name. pandas 3 drops
    # it. The loader must not depend on which pandas is installed.
    path = write(tmp_path, "c.csv", "name,male,female\nalan,90,10\n,5,95\n")
    index = _load_counts("s", path, counts_cfg())
    assert set(index) == {"alan"}
    assert MultiDBLookup(
        {"s": {"enabled": True, "path": path, "format": "counts", **counts_cfg()}}
    ).query("Nan", "")[0].found is False


def test_names_are_indexed_case_and_whitespace_insensitively(tmp_path):
    path = write(tmp_path, "c.csv", "name,male,female\n  ADA  ,0,100\n")
    assert "ada" in _load_counts("s", path, counts_cfg())


def test_missing_column_is_named_before_the_body_is_read(tmp_path):
    path = write(tmp_path, "c.csv", "name,m,f\nalan,90,10\n")
    with pytest.raises(ValueError, match="'male' is not in the file header"):
        _load_counts("s", path, counts_cfg())


def test_counts_format_without_a_name_column_is_refused(tmp_path):
    path = write(tmp_path, "c.csv", "name,male,female\nalan,90,10\n")
    with pytest.raises(ValueError, match="requires a name_col"):
        _load_counts("s", path, counts_cfg(name_col=None))


# -- wgnd loader ----------------------------------------------------------


def test_only_m_and_f_rows_are_kept(tmp_path):
    # WGND marks unknown with ? and unisex with =; neither is a verdict.
    path = write(tmp_path, "w.csv", "name,code,gender,wgt\n"
                 "alan,GB,M,10\nunknown,GB,?,10\nunisex,GB,=,10\n")
    assert set(_load_wgnd("s", path, wgnd_cfg())) == {"alan"}


def test_weights_are_summed_across_countries(tmp_path):
    path = write(tmp_path, "w.csv", "name,code,gender,wgt\n"
                 "andrea,IT,M,90\nandrea,FR,F,5\nandrea,DE,F,5\n")
    match = _load_wgnd("s", path, wgnd_cfg())["andrea"]
    assert (match.gender, match.frequency) == (Gender.MALE, 0.9)


def test_a_missing_weight_column_counts_each_row_once(tmp_path):
    path = write(tmp_path, "w.csv", "name,code,gender\n"
                 "andrea,IT,M\nandrea,ES,M\nandrea,FR,F\n")
    match = _load_wgnd("s", path, wgnd_cfg(freq_col="wgt"))["andrea"]
    assert (match.gender, match.frequency) == (Gender.MALE, pytest.approx(2 / 3))


def test_blank_wgnd_names_never_become_the_name_nan(tmp_path):
    path = write(tmp_path, "w.csv", "name,code,gender,wgt\nalan,GB,M,10\n,FR,F,50\n")
    assert set(_load_wgnd("s", path, wgnd_cfg())) == {"alan"}


def test_country_filter_restricts_the_global_aggregate_too(tmp_path):
    # The filter is the user saying "only these countries exist for me", so the
    # fallback aggregate must not smuggle the excluded ones back in.
    path = write(tmp_path, "w.csv", "name,code,gender,wgt\n"
                 "andrea,IT,M,90\nandrea,FR,F,10\n")
    match = _load_wgnd("s", path, wgnd_cfg(country_filter=["FR"]))["andrea"]
    assert (match.gender, match.frequency) == (Gender.FEMALE, 1.0)


# -- country-aware lookup -------------------------------------------------


@pytest.fixture
def andrea(tmp_path):
    """andrea is male in Italy and female nearly everywhere else."""
    path = write(tmp_path, "w.csv", "name,code,gender,wgt\n"
                 "andrea,IT,M,90\nandrea,FR,F,40\nandrea,GB,F,40\n")
    return _load_wgnd("s", path, wgnd_cfg(country_aware=True))


def test_the_records_own_country_beats_the_global_dominant(andrea):
    assert andrea("andrea", "andrea smith", ["IT"]).gender is Gender.MALE
    assert andrea("andrea", "andrea smith", ["FR"]).gender is Gender.FEMALE


def test_no_overlap_falls_back_to_the_global_dominant(andrea):
    # JP is not in the file at all: better the worldwide verdict than no evidence.
    assert andrea("andrea", "andrea smith", ["JP"]).gender is Gender.MALE


def test_a_record_without_countries_uses_the_global_aggregate(andrea):
    for countries in (None, []):
        match = andrea("andrea", "andrea smith", countries)
        assert (match.gender, match.frequency) == (Gender.MALE, pytest.approx(90 / 170))


def test_country_codes_are_normalised_before_matching(andrea):
    assert andrea("andrea", "andrea smith", [" it ", "it"]).gender is Gender.MALE


def test_an_unknown_name_matches_nothing_in_either_branch(andrea):
    assert andrea("nobody", "nobody smith", ["IT"]) is None
    assert andrea("nobody", "nobody smith", None) is None


# -- MultiDBLookup --------------------------------------------------------


def two_sources(tmp_path):
    first = write(tmp_path, "first.csv", "name,male,female\nada,0,100\n")
    second = write(tmp_path, "second.csv", "name,male,female\nada lovelace,100,0\n")
    return {
        "alpha": {"enabled": True, "path": first, "format": "counts", **counts_cfg()},
        "beta": {"enabled": True, "path": second, "format": "counts", **counts_cfg()},
    }


def test_loaded_sources_keep_config_order_which_fixes_the_columns(tmp_path):
    cfg = two_sources(tmp_path)
    assert MultiDBLookup(cfg).loaded_sources == ["alpha", "beta"]
    reversed_cfg = {k: cfg[k] for k in reversed(list(cfg))}
    assert MultiDBLookup(reversed_cfg).loaded_sources == ["beta", "alpha"]


def test_the_full_name_key_is_preferred_over_the_first_name_key(tmp_path):
    # A dictionary indexed on whole names must win over one indexed on given names,
    # since the whole name is the more specific evidence.
    path = write(tmp_path, "d.csv", "name,male,female\n"
                 "ada,0,100\nada lovelace,100,0\n")
    cfg = {"s": {"enabled": True, "path": path, "format": "counts", **counts_cfg()}}
    lookup = MultiDBLookup(cfg)
    assert lookup.query("Ada", "Lovelace")[0].gender is Gender.MALE
    assert lookup.query("Ada", "")[0].gender is Gender.FEMALE


def test_queries_are_case_and_whitespace_insensitive(tmp_path):
    lookup = MultiDBLookup(two_sources(tmp_path))
    assert lookup.query("  ADA  ", " Lovelace ")[1].gender is Gender.MALE


def test_every_loaded_source_answers_exactly_once(tmp_path):
    results = MultiDBLookup(two_sources(tmp_path)).query("Grace", "Hopper")
    assert [m.source for m in results] == ["alpha", "beta"]
    assert all(m.found is False and m.gender is None for m in results)


def test_an_empty_config_loads_nothing_and_answers_nothing(tmp_path):
    for cfg in ({}, None):
        lookup = MultiDBLookup(cfg)
        assert lookup.loaded_sources == []
        assert lookup.query("Ada", "Lovelace") == []


# -- loading policy -------------------------------------------------------


def test_a_disabled_block_is_skipped_without_touching_its_path():
    lookup = MultiDBLookup({"s": {"enabled": False, "path": "/does/not/exist.csv"}})
    assert lookup.loaded_sources == []


def test_a_block_without_a_path_is_skipped():
    assert MultiDBLookup({"s": {"enabled": True, "format": "counts"}}).loaded_sources == []


def test_an_enabled_dictionary_with_a_missing_file_stops_the_run():
    # Continuing would drop a corpus-scale run to LLM-only: slower, less accurate
    # and more expensive, with nothing but a warning to show for it.
    with pytest.raises(ValueError) as excinfo:
        MultiDBLookup({"wgnd": {"enabled": True, "path": "/does/not/exist.csv"}})
    assert "/does/not/exist.csv" in str(excinfo.value)
    assert "prenomen-dicts get wgnd" in str(excinfo.value)


def test_an_unknown_format_stops_the_run(tmp_path):
    path = write(tmp_path, "d.csv", "name,male,female\nada,0,100\n")
    with pytest.raises(ValueError, match="unknown format"):
        MultiDBLookup({"s": {"enabled": True, "path": path, "format": "bogus"}})


def test_a_dictionary_that_fails_to_parse_stops_the_run(tmp_path):
    path = write(tmp_path, "empty.csv", "")
    with pytest.raises(ValueError) as excinfo:
        MultiDBLookup({"s": {"enabled": True, "path": path, "format": "counts",
                             **counts_cfg()}})
    assert "could not be read" in str(excinfo.value)


def test_a_misconfigured_column_stops_the_run_naming_the_column(tmp_path):
    path = write(tmp_path, "d.csv", "name,male,female\nada,0,100\n")
    with pytest.raises(ValueError, match="Dictionary 's'.*'wrong'"):
        MultiDBLookup({"s": {"enabled": True, "path": path, "format": "counts",
                             **counts_cfg(male_col="wrong")}})


def test_a_home_relative_path_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    write(tmp_path, "d.csv", "name,male,female\nada,0,100\n")
    lookup = MultiDBLookup(
        {"s": {"enabled": True, "path": "~/d.csv", "format": "counts", **counts_cfg()}}
    )
    assert lookup.query("Ada", "")[0].gender is Gender.FEMALE
