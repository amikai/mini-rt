"""M09: same as M08. Turns a growing list of token ids into text without splitting a UTF-8 character.

Offset names follow SGLang's DecodeStatus (managers/detokenizer_manager.py).
"""

from transformers import PreTrainedTokenizerBase

# What the tokenizer returns for bytes that are not a complete UTF-8 character yet.
REPLACEMENT_CHAR = "\N{REPLACEMENT CHARACTER}"  # U+FFFD


class IncrementalDetokenizer:
    """Decode state for one request's output_ids."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.decoded_text = ""  # confirmed text; never ends in a partial character
        self.surr_offset = 0  # start of the context window decoded again each step
        self.read_offset = 0  # output_ids[:read_offset] are already in decoded_text

    def decode(self, output_ids: list[int], finished: bool) -> str:
        """Returns all text decoded so far. finished=True also returns text still held back."""
        # Decode with the previous chunk as context, then keep only what the new tokens added.
        surr_text = self._decode(output_ids[self.surr_offset : self.read_offset])
        new_text = self._decode(output_ids[self.surr_offset :])[len(surr_text) :]
        if finished:
            return self.decoded_text + new_text
        # Commit only at a character boundary; otherwise the next token decodes it again.
        if new_text and not new_text.endswith(REPLACEMENT_CHAR):
            self.decoded_text += new_text
            self.surr_offset = self.read_offset
            self.read_offset = len(output_ids)
        return self.decoded_text

    def _decode(self, ids: list[int]) -> str:
        # Same setting as the non-stream path.
        return self.tokenizer.decode(ids, skip_special_tokens=True)
