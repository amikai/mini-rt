import pytest

from req import FINISH_LENGTH, FINISH_MATCHED_TOKEN, Req, ReqStatus

EOS_A, EOS_B = 98, 99


def make_req(max_new_tokens: int = 4) -> Req:
    return Req(rid="r", prompt="", input_ids=[1], max_new_tokens=max_new_tokens, eos_token_ids={EOS_A, EOS_B})


def test_new_request_is_waiting():
    req = make_req()

    assert req.status is ReqStatus.WAITING
    assert not req.finished()


@pytest.mark.parametrize("end", [ReqStatus.FINISHED, ReqStatus.FAILED])
def test_legal_path(end):
    req = make_req()

    req.set_status(ReqStatus.RUNNING)
    req.set_status(end)

    assert req.status is end


@pytest.mark.parametrize(
    ("path", "bad"),
    [
        ([], ReqStatus.FINISHED),  # WAITING cannot skip RUNNING
        ([], ReqStatus.WAITING),  # no self-loop
        ([ReqStatus.RUNNING], ReqStatus.WAITING),  # no going back
        ([ReqStatus.RUNNING, ReqStatus.FINISHED], ReqStatus.RUNNING),  # terminal
        ([ReqStatus.RUNNING, ReqStatus.FAILED], ReqStatus.FINISHED),  # terminal
    ],
)
def test_illegal_transition_raises_and_keeps_status(path, bad):
    req = make_req()
    for status in path:
        req.set_status(status)
    before = req.status

    with pytest.raises(RuntimeError):
        req.set_status(bad)
    assert req.status is before


def test_not_finished_mid_generation():
    req = make_req(max_new_tokens=4)
    req.output_ids.append(5)

    req.update_finish_state()

    assert req.finished_reason is None


@pytest.mark.parametrize("eos", [EOS_A, EOS_B])
def test_any_eos_id_stops(eos):
    req = make_req()
    req.output_ids += [5, eos]

    req.update_finish_state()

    assert isinstance(req.finished_reason, FINISH_MATCHED_TOKEN)
    assert req.finished_reason.to_json() == {"type": "stop", "matched": eos}


def test_length_stops():
    req = make_req(max_new_tokens=2)
    req.output_ids += [5, 6]

    req.update_finish_state()

    assert isinstance(req.finished_reason, FINISH_LENGTH)
    assert req.finished_reason.to_json() == {"type": "length", "length": 2}


def test_eos_on_last_step_is_stop_not_length():
    req = make_req(max_new_tokens=2)
    req.output_ids += [5, EOS_A]

    req.update_finish_state()

    assert isinstance(req.finished_reason, FINISH_MATCHED_TOKEN)
