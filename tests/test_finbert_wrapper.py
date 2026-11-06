import numpy as np
import pytest

from ecs.finbert import MAX_LEN, FinBert, chunk_ids


def test_chunking_covers_all_tokens():
    for n in [0, 1, 510, 511, 1000, 2345]:
        ids = list(range(n))
        ch = chunk_ids(ids)
        assert [t for c in ch for t in c] == ids
        assert all(len(c) + 2 <= MAX_LEN for c in ch)


@pytest.fixture(scope="module")
def fb():
    return FinBert(batch_size=4)


def test_deterministic_and_sensible(fb):
    t = ["Revenue grew strongly and margins expanded.", "We reported a significant loss and weak demand.",
         "The call will begin shortly."]
    a, b = fb.score(t), fb.score(t)
    assert np.allclose(a, b, atol=1e-5) and np.allclose(a.sum(1), 1, atol=1e-4)
    assert a[0, 0] > a[0, 1] and a[1, 1] > a[1, 0]


def test_long_text_not_truncated(fb):
    long = "Revenue grew strongly. " * 400  # >> 512 tokens in one 'sentence'
    n_tok = len(fb.tok(long, add_special_tokens=False)["input_ids"])
    assert n_tok > MAX_LEN
    p = fb.score([long])  # internal assert checks all tokens were chunked
    assert p.shape == (1, 3) and p[0, 0] > p[0, 1]
