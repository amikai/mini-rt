import threading
from dataclasses import dataclass, field


@dataclass
class Req:
    """One generation request and everything the runtime knows about it."""

    rid: str
    prompt: str
    input_ids: list[int]
    max_new_tokens: int
    output_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None  # None while running, then "stop" or "length"
    error: Exception | None = None  # set if the run failed
    # Set by the scheduler when the request finishes, success or error.
    done: threading.Event = field(default_factory=threading.Event, repr=False)
