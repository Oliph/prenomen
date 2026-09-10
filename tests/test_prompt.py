"""The prompt is the experiment, so its exact text is pinned to fixture files.

A prompt change alters what the model answers, which makes results from before and
after the change incomparable. These tests do not judge the wording; they make an
edit to the template visible in the diff rather than silently changing every run.

To adopt an intentional change, regenerate the fixtures:

    python -c "import tests.test_prompt as t; t.regenerate()"
"""

from pathlib import Path

import pytest

from prenomen.models import DictMatch, Gender
from prenomen.pipeline import build_prompt_template

FIXTURES = Path(__file__).parent / "fixtures"

MATCHES = [
    DictMatch(source="wgnd", gender=Gender.MALE, frequency=0.62, found=True),
    DictMatch(source="chinese_pinyin", gender=Gender.FEMALE, frequency=0.71, found=True),
    DictMatch(source="no_frequency", gender=Gender.MALE, frequency=None, found=True),
]

CASES = {
    "prompt_llm_only.txt": dict(
        first_name="Ada", last_name="Lovelace", dict_matches=[], countries=()
    ),
    "prompt_dicts_and_countries.txt": dict(
        first_name="Andrea",
        last_name="Rossi",
        dict_matches=MATCHES,
        countries=("FR", "IT"),
    ),
    "prompt_dicts_no_country.txt": dict(
        first_name="Andrea", last_name="Rossi", dict_matches=MATCHES, countries=()
    ),
}


def regenerate():
    template = build_prompt_template()
    for name, kwargs in CASES.items():
        (FIXTURES / name).write_text(template.render(**kwargs), encoding="utf-8")


@pytest.mark.parametrize("fixture", CASES)
def test_rendered_prompt_matches_its_fixture(fixture):
    rendered = build_prompt_template().render(**CASES[fixture])
    assert rendered == (FIXTURES / fixture).read_text(encoding="utf-8")


def test_no_branch_leaves_a_run_of_blank_lines():
    # Jinja's default whitespace handling used to leave seven of these per prompt,
    # which is tokens paid for on every single request.
    for kwargs in CASES.values():
        assert "\n\n\n" not in build_prompt_template().render(**kwargs)


def test_a_match_without_a_frequency_still_gets_its_own_line():
    rendered = build_prompt_template().render(**CASES["prompt_dicts_no_country.txt"])
    assert "\n    - no_frequency: male\n" in rendered
    assert "\n    - wgnd: male (frequency ratio: 0.62)\n" in rendered


def test_countries_reach_the_prompt_only_when_present():
    with_countries = build_prompt_template().render(
        **CASES["prompt_dicts_and_countries.txt"]
    )
    without = build_prompt_template().render(**CASES["prompt_dicts_no_country.txt"])
    assert "affiliated with institution(s) in: FR, IT" in with_countries
    assert "affiliated with" not in without


def test_template_ships_inside_the_package():
    # build_prompt_template() resolves the template relative to the package, so a
    # wheel that omitted the .j2 file would fail on the first record, not at import.
    import prenomen

    packaged = Path(prenomen.__file__).parent / "templates" / "gender_prompt.j2"
    assert packaged.is_file()
