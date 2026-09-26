import random
from itertools import pairwise

import pytest
from transformers import AutoTokenizer

from detokenizer import IncrementalDetokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def stream(tokenizer, ids: list[int]) -> list[str]:
    """Feed ids one token at a time, the way the stream handler does."""
    detokenizer = IncrementalDetokenizer(tokenizer)
    return [detokenizer.decode(ids[:i], finished=i == len(ids)) for i in range(1, len(ids) + 1)]


def test_split_character_is_the_case_under_test(tokenizer):
    # 龘 is two tokens; decoding only the first gives a replacement character.
    ids = tokenizer.encode("龘")
    assert len(ids) == 2
    assert tokenizer.decode(ids[:1]) == "�"


def test_ascii_grows_token_by_token(tokenizer):
    ids = tokenizer.encode("Hello world, how are you")

    texts = stream(tokenizer, ids)

    assert texts == [tokenizer.decode(ids[:i]) for i in range(1, len(ids) + 1)]


def test_holds_back_half_a_character(tokenizer):
    ids = tokenizer.encode("好龘好")  # 好, 龘 part 1, 龘 part 2, 好

    texts = stream(tokenizer, ids)

    assert texts == ["好", "好", "好龘", "好龘好"]


def test_flushes_half_character_when_finished(tokenizer):
    # Stopped by max_new_tokens in the middle of 龘: final text matches the non-stream decode.
    ids = tokenizer.encode("好龘")[:-1]

    texts = stream(tokenizer, ids)

    assert texts[-1] == tokenizer.decode(ids) == "好�"


def test_special_tokens_are_skipped(tokenizer):
    ids = [*tokenizer.encode("Hi"), tokenizer.convert_tokens_to_ids("<|im_end|>")]

    assert stream(tokenizer, ids)[-1] == "Hi"


def test_random_ids_match_non_stream_decode(tokenizer):
    rng = random.Random(0)
    for _ in range(50):
        ids = [rng.randrange(len(tokenizer)) for _ in range(20)]

        texts = stream(tokenizer, ids)

        # Every chunk extends the previous one; nothing already sent is taken back.
        assert all(b.startswith(a) for a, b in pairwise(texts))
        assert not any(t.endswith("�") for t in texts[:-1])
        assert texts[-1] == tokenizer.decode(ids, skip_special_tokens=True)
