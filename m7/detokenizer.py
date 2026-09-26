"""M7: incremental detokenization. Turns a growing list of token ids into text without splitting a UTF-8 character.

Offset names follow SGLang's DecodeStatus (managers/detokenizer_manager.py).
"""

from transformers import PreTrainedTokenizerBase

# What the tokenizer returns for bytes that are not a complete UTF-8 character yet.
REPLACEMENT_CHAR = "�"


class IncrementalDetokenizer:
    """Decode state for one request's output_ids."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.decoded_text = ""  # confirmed text; never ends in a partial character
        self.surr_offset = 0  # start of the context window decoded again each step
        self.read_offset = 0  # output_ids[:read_offset] are already in decoded_text

    def decode(self, output_ids: list[int], finished: bool) -> str:
        """Returns all text decoded so far. finished=True also returns text still held back."""
        # TODO(M7-4): decode only a window, not all output_ids and not one token alone.
        # 1. new_text: what output_ids[read_offset:] adds on top of output_ids[surr_offset:read_offset].
        #    Decode both slices starting at surr_offset and take the difference.
        # 2. finished: return decoded_text + new_text, even if it ends in REPLACEMENT_CHAR.
        # 3. new_text non-empty and not ending in REPLACEMENT_CHAR: commit it and slide both offsets forward.
        #    Otherwise hold it back; the next token decodes it again.
        raise NotImplementedError

    def _decode(self, ids: list[int]) -> str:
        # Same setting as the non-stream path.
        return self.tokenizer.decode(ids, skip_special_tokens=True)
