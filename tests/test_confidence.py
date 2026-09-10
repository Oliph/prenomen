"""Tests for the reliability-weighted dictionary confidence and the bypass gate."""

import unittest

from prenomen.confidence import (
    SourceMeta,
    source_meta_from_config,
    weighted_dict_confidence,
)
from prenomen.models import AuthorInput, DictMatch, Gender, ReasoningCode
from prenomen.pipeline import _try_dict_bypass

THRESHOLD = 0.85

M = Gender.MALE
F = Gender.FEMALE

# The weights the shipped config declares: a broad source at 1, specialist ones at
# 2, with the two Chinese resources sharing one family.
META = {
    "wgnd": SourceMeta("wgnd", 1.0),
    "chinese_pinyin": SourceMeta("chinese", 2.0),
    "chinese_cn": SourceMeta("chinese", 2.0),
}


def match(source, gender, freq):
    return DictMatch(source=source, gender=gender, frequency=freq, found=True)


def miss(source):
    return DictMatch(source=source, gender=None, frequency=None, found=False)


def conf(matches):
    return weighted_dict_confidence(matches, META)


def bypass_gender(matches, threshold=THRESHOLD):
    """Return the gender a bypass would assign, or None if it routes to the LLM."""
    author = AuthorInput(first_name="Test", last_name="Name")
    resp = _try_dict_bypass(matches, author, threshold, META)
    return None if resp is None else resp.author.gender


class SourceMetaConfigTest(unittest.TestCase):
    def test_config_drives_family_and_weight(self):
        meta = source_meta_from_config(
            {
                "my_list": {"family": "regional", "weight": 3.0},
                "bare": {},
            }
        )
        self.assertEqual(meta["my_list"], SourceMeta("regional", 3.0))
        # An entry that declares neither falls back to its own name at weight 1.
        self.assertEqual(meta["bare"], SourceMeta("bare", 1.0))

    def test_shipped_sources_keep_calibrated_defaults(self):
        # A config block that omits family and weight must not silently change the
        # scoring of the dictionaries the thresholds were calibrated against.
        meta = source_meta_from_config({"wgnd": {}, "chinese_pinyin": {}})
        self.assertEqual(meta["wgnd"], SourceMeta("wgnd", 1.0))
        self.assertEqual(meta["chinese_pinyin"], SourceMeta("chinese", 2.0))

    def test_unknown_source_defaults_to_weight_one(self):
        _, c = weighted_dict_confidence([match("some_new_list", M, 0.77)], {})
        self.assertAlmostEqual(c, 0.77, places=4)


class WeightedDictConfidenceTest(unittest.TestCase):
    def test_no_match_returns_none(self):
        self.assertIsNone(conf([miss("wgnd"), miss("chinese_pinyin")]))

    def test_single_source_reduces_to_frequency(self):
        # A lone source's confidence is exactly its frequency ratio.
        _, c = conf([match("wgnd", M, 0.86)])
        self.assertAlmostEqual(c, 0.86, places=4)
        _, c = conf([match("chinese_pinyin", F, 0.62)])
        self.assertAlmostEqual(c, 0.62, places=4)

    def test_specialist_double_weight_in_mean(self):
        # wgnd (w=1) 0.6 male + chinese (w=2) 0.9 male -> (0.6 + 2*0.9)/3 = 0.8
        dom, c = conf([match("wgnd", M, 0.6), match("chinese_pinyin", M, 0.9)])
        self.assertEqual(dom, M)
        self.assertAlmostEqual(c, (0.6 + 2 * 0.9) / 3, places=4)

    def test_conflict_pulls_score_down(self):
        # wgnd male 0.9 (w=1) vs chinese female 0.9 (w=2): dominant female,
        # p = (1*(1-0.9) + 2*0.9)/3 = 0.6333
        dom, c = conf([match("wgnd", M, 0.9), match("chinese_pinyin", F, 0.9)])
        self.assertEqual(dom, F)
        self.assertAlmostEqual(c, (0.1 + 1.8) / 3, places=4)
        self.assertLess(c, THRESHOLD)

    def test_correlated_sources_share_one_family_weight(self):
        # Both Chinese resources agree: the family carries weight 2 once, split
        # across its two members, so 0.6 + 0.9 + 0.9 -> 0.8, not the 0.84 that an
        # uncapped 2-per-dictionary scheme would give.
        dom, c = conf(
            [
                match("wgnd", M, 0.6),
                match("chinese_pinyin", M, 0.9),
                match("chinese_cn", M, 0.9),
            ]
        )
        self.assertEqual(dom, M)
        self.assertAlmostEqual(c, 0.8, places=4)

    def test_tie_returns_half_and_never_bypasses(self):
        # Equal and opposite: the two chinese sources share one family, so they carry
        # weight 1 each, and 0.9 against 0.9 cancels exactly. Unresolvable, so
        # confidence 0.5 regardless of input order, and below any threshold so it
        # always routes to the model.
        tie = [match("chinese_pinyin", M, 0.9), match("chinese_cn", F, 0.9)]
        self.assertEqual(conf(tie), (M, 0.5))
        self.assertEqual(conf(list(reversed(tie))), (M, 0.5))
        self.assertIsNone(bypass_gender(tie))

    def test_confidence_never_contradicts_the_gender_it_returns(self):
        # Two weak male sources against one strong female one. Summing votes per
        # gender would elect male and then score it at 0.36, i.e. return a gender its
        # own confidence argues against. One rule decides both, so female wins.
        dom, c = conf(
            [
                match("wgnd", M, 0.51),
                match("chinese_pinyin", M, 0.51),
                match("chinese_cn", F, 0.95),
            ]
        )
        self.assertEqual(dom, F)
        self.assertGreaterEqual(c, 0.5)

    def test_frequency_none_match_is_ignored(self):
        # A match without a frequency carries no evidence and must not move the
        # score, nor flip the dominant, which a weight-2 female match otherwise would.
        _, base = conf([match("wgnd", M, 0.9)])
        dom, withnone = conf(
            [match("wgnd", M, 0.9), match("chinese_pinyin", F, None)]
        )
        self.assertEqual(dom, M)
        self.assertAlmostEqual(base, 0.9, places=4)
        self.assertAlmostEqual(withnone, 0.9, places=4)


class BypassGateTest(unittest.TestCase):
    def test_specialist_low_freq_routes_to_llm(self):
        # The gate compares a probability, not a vote magnitude: 0.62 < 0.85.
        self.assertIsNone(bypass_gender([match("chinese_pinyin", F, 0.62)]))

    def test_specialist_high_freq_bypasses(self):
        self.assertEqual(bypass_gender([match("chinese_pinyin", F, 0.95)]), F)

    def test_at_threshold_bypasses(self):
        self.assertEqual(bypass_gender([match("wgnd", M, 0.86)]), M)

    def test_below_threshold_routes_to_llm(self):
        self.assertIsNone(bypass_gender([match("wgnd", M, 0.80)]))

    def test_conflict_routes_to_llm(self):
        self.assertIsNone(
            bypass_gender([match("wgnd", M, 0.9), match("chinese_pinyin", F, 0.9)])
        )

    def test_no_match_returns_none(self):
        self.assertIsNone(bypass_gender([miss("wgnd"), miss("chinese_pinyin")]))

    def test_threshold_zero_bypasses_any_match(self):
        # 0.51 is the weakest a real match gets: both loaders report the frequency of
        # the larger count, so it is never below 0.5.
        self.assertEqual(bypass_gender([match("wgnd", M, 0.51)], threshold=0.0), M)

    def test_bypass_reasoning_code_and_stored_confidence(self):
        author = AuthorInput(first_name="Test", last_name="Name")
        resp = _try_dict_bypass([match("wgnd", M, 0.95)], author, THRESHOLD, META)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.author.reasoning_code, ReasoningCode.DICT_BYPASS)
        self.assertAlmostEqual(resp.author.confidence, 0.95, places=4)

    def test_gate_uses_rounded_confidence(self):
        # The gate and the stored value must agree: 0.84999 rounds to 0.85 and
        # bypasses, 0.849 stays below and routes to the model.
        self.assertEqual(bypass_gender([match("wgnd", M, 0.84999)]), M)
        self.assertIsNone(bypass_gender([match("wgnd", M, 0.849)]))


if __name__ == "__main__":
    unittest.main()
