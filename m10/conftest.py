import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    # Tiny random Qwen3 with the real tokenizer, same as m2-m4 tests.
    model_dir = tmp_path_factory.mktemp("tiny-qwen3")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return str(model_dir)
