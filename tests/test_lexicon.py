import math

import pytest

from ecs.lexicon import load_lexicon, score_text

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path("data/raw/LM_MasterDictionary.csv").exists(), reason="LM dictionary not downloaded")


def test_positive_beats_negative():
    assert score_text("Record revenue and strong profitability, we are pleased with the improvement")["lm_net"] > 0
    assert score_text("Revenue declined and we incurred losses, litigation and impairment")["lm_net"] < 0


def test_exact_shares():
    r = score_text("strong loss maybe cat")  # 1 pos, 1 neg, 1 unc ("maybe"), 4 words
    assert r["n_words"] == 4 and r["lm_pos"] == 0.25 and r["lm_neg"] == 0.25 and r["lm_unc"] == 0.25
    assert r["lm_net"] == 0


def test_uncertainty_and_empty():
    assert score_text("results may possibly vary, depending on uncertain conditions")["lm_unc"] > 0.2
    assert math.isnan(score_text("")["lm_net"])


def test_lexicon_sizes():
    lex = load_lexicon()
    assert 2000 < len(lex["negative"]) < 3000 and 300 < len(lex["positive"]) < 500
