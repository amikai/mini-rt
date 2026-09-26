"""M6: Req is a state machine. Status moves WAITING -> RUNNING -> FINISHED or FAILED.

Why it finished is a finish reason object, named after SGLang's FINISH_* classes (managers/schedule_batch.py).
"""

import enum
import threading
from dataclasses import dataclass, field


class ReqStatus(enum.Enum):
    WAITING = enum.auto()  # queued, not yet picked by the scheduler
    RUNNING = enum.auto()  # generating one token per scheduler step
    FINISHED = enum.auto()  # stopped normally: EOS or max_new_tokens
    FAILED = enum.auto()  # stopped by an exception


# Legal moves only; anything else is a scheduler bug.
_TRANSITIONS: dict[ReqStatus, set[ReqStatus]] = {
    ReqStatus.WAITING: {ReqStatus.RUNNING},
    ReqStatus.RUNNING: {ReqStatus.FINISHED, ReqStatus.FAILED},
    ReqStatus.FINISHED: set(),
    ReqStatus.FAILED: set(),
}


class BaseFinishReason:
    def to_json(self) -> dict:
        raise NotImplementedError


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    """Generated one of the EOS token IDs."""

    def __init__(self, matched: int):
        self.matched = matched

    def to_json(self) -> dict:
        return {"type": "stop", "matched": self.matched}


class FINISH_LENGTH(BaseFinishReason):
    """Reached max_new_tokens."""

    def __init__(self, length: int):
        self.length = length

    def to_json(self) -> dict:
        return {"type": "length", "length": self.length}


class FINISH_ERROR(BaseFinishReason):
    """The forward step raised. Not in SGLang, where a model error kills the scheduler process."""

    def __init__(self, error: Exception):
        self.error = error

    def to_json(self) -> dict:
        return {"type": "error", "message": str(self.error)}


@dataclass
class Req:
    """One generation request and everything the runtime knows about it."""

    rid: str
    prompt: str
    input_ids: list[int]
    max_new_tokens: int
    # Any of these ends generation; Qwen3 has two.
    eos_token_ids: set[int] = field(default_factory=set)
    output_ids: list[int] = field(default_factory=list)
    status: ReqStatus = ReqStatus.WAITING
    finished_reason: BaseFinishReason | None = None  # None until the request ends
    # Set by the scheduler when the request ends, FINISHED or FAILED.
    done: threading.Event = field(default_factory=threading.Event, repr=False)

    def finished(self) -> bool:
        return self.finished_reason is not None

    def set_status(self, new: ReqStatus) -> None:
        # TODO(M6-1): move to `new` only if _TRANSITIONS allows it from self.status.
        # Illegal move: raise RuntimeError naming both states.
        raise NotImplementedError

    def update_finish_state(self) -> None:
        """Called after each new token. Sets finished_reason if the request should stop."""
        # TODO(M6-2): look at the last token in output_ids.
        # 1. In eos_token_ids: FINISH_MATCHED_TOKEN. Checked first, so EOS on the last allowed step is "stop".
        # 2. Else output_ids reached max_new_tokens: FINISH_LENGTH.
        # 3. Else leave finished_reason as None.
        raise NotImplementedError
