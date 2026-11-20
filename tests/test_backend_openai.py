"""Tests for `OpenAIBackend` (predicate/choice/score conversion, positional mapping, refusals).

Offline by design: the backend is exercised against a fake client injected into it, returning real
`openai.types.decision.Decision` objects -- the SDK's own validation, no server. The backend never
constructs a client and never reads an API key, so no factory or environment is patched anywhere.
The base contract these tests stand on is pinned in `test_backend_base.py`.
"""

from __future__ import annotations

import httpx2
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from openai.types.decision import (
    AnswerAnswerResourceChoice,
    AnswerAnswerResourceChoiceProbability,
    AnswerAnswerResourcePredicate,
    AnswerAnswerResourceRefusal,
    AnswerAnswerResourceScore,
    AnswerAnswerResourceScoreProbability,
    Decision,
)
from typesafe_sdk import Noul

from decision_as_reranker.backend.base import ChoiceAnswer, DecisionResponse, NoulAnswer, ScoreAnswer
from decision_as_reranker.backend.openai import DEFAULT_MODEL, OpenAIBackend

STATE = {"query": "what matters"}


def predicate_answer(probability=0.75, name=None):
    return AnswerAnswerResourcePredicate(type="predicate", name=name, probability=probability)


def choice_answer(entries, choice="d0", confidence=0.8, **overrides):
    probabilities = [
        AnswerAnswerResourceChoiceProbability(value=value, probability=probability) for value, probability in entries
    ]
    return AnswerAnswerResourceChoice(
        type="choice", name=None, choice=choice, probabilities=probabilities, confidence=confidence, **overrides
    )


def score_answer(entries, score=1.5):
    probabilities = [
        AnswerAnswerResourceScoreProbability(value=index, label=f"L{index}", probability=probability)
        for index, probability in entries
    ]
    return AnswerAnswerResourceScore(type="score", name=None, score=score, probabilities=probabilities, confidence=0.5)


def refusal_answer():
    return AnswerAnswerResourceRefusal(type="refusal", name=None)


def decision(answers, model="gpt-6-luna"):
    """A real `Decision`, built through the SDK's own validation -- no server involved."""
    return Decision(
        answers=answers,
        model=model,
        usage={
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 103,
        },
    )


class FakeDecisions:
    def __init__(self, response=None, error=None):
        self.response = response if response is not None else decision([predicate_answer()])
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeOpenAIClient:
    """Stands in for an `OpenAI` client: records calls, returns a canned decision or raises."""

    def __init__(self, response=None, error=None):
        self.decisions = FakeDecisions(response=response, error=error)
        self.closed = False

    def close(self):
        self.closed = True


def status_error(cls=BadRequestError, status=400):
    request = httpx2.Request("POST", "https://api.example/v1/decisions")
    response = httpx2.Response(status, request=request)
    return cls("boom", response=response, body=None)


# --------------------------------------------------------------------------- answer conversion


def test_predicate_answers_convert_to_the_neutral_noul_not_a_provider_type():
    # The side note this file exists to pin: the wire says `predicate`, the framework says `noul`.
    # Nothing above the seam may learn the word "predicate".
    backend = OpenAIBackend(FakeOpenAIClient(response=decision([predicate_answer(0.9)])))

    response = backend.ask(STATE, {"q0": {"type": "noul", "instructions": "relevant?"}})

    assert response.answers == {"q0": NoulAnswer(noul=0.9)}
    assert not hasattr(response.answers["q0"], "confidence"), "confidence is dropped at the boundary"


def test_choice_and_score_answers_convert_to_the_neutral_contract():
    answers = [
        choice_answer([("d0", 0.2), ("d1", 0.8)], choice="d1"),
        score_answer([(0, 0.5), (1, 0.5)], score=0.5),
    ]
    backend = OpenAIBackend(FakeOpenAIClient(response=decision(answers)))
    questions = {
        "c": {"type": "choice", "instructions": "which?", "criteria": {"d0": "a", "d1": "b"}},
        "s": {"type": "score", "instructions": "how much?", "criteria": ["lo", "hi"]},
    }

    response = backend.ask(STATE, questions)

    assert response.answers["c"] == ChoiceAnswer(choice="d1", probabilities={"d0": 0.2, "d1": 0.8})
    assert response.answers["s"] == ScoreAnswer(score=0.5, probabilities={0: 0.5, 1: 0.5}, legend={0: "L0", 1: "L1"})


def test_answers_map_by_position_not_by_name():
    # Ids carry no meaning to the model; the array order is the contract, echoed names are a
    # cross-check only. A backend that keyed on `name` would misfile every answer here.
    backend = OpenAIBackend(
        FakeOpenAIClient(response=decision([predicate_answer(0.1, name="zzz"), predicate_answer(0.9, name="aaa")]))
    )

    response = backend.ask(
        STATE, {"candidate-a": {"type": "noul", "instructions": "x"}, "zid": {"type": "noul", "instructions": "y"}}
    )

    assert response.answers["candidate-a"] == NoulAnswer(noul=0.1)
    assert response.answers["zid"] == NoulAnswer(noul=0.9)


def test_refusal_leaves_the_question_unanswered_rather_than_zero():
    # A decline is not evidence of irrelevance: the id is absent, and the reranker's on_error
    # policy -- not this layer -- decides raise-vs-drop.
    backend = OpenAIBackend(FakeOpenAIClient(response=decision([predicate_answer(0.4), refusal_answer()])))

    response = backend.ask(
        STATE, {"q0": {"type": "noul", "instructions": "x"}, "q1": {"type": "noul", "instructions": "y"}}
    )

    assert set(response.answers) == {"q0"}
    assert response.answers["q0"] == NoulAnswer(noul=0.4)


def test_an_answer_count_disagreement_raises_instead_of_truncating():
    backend = OpenAIBackend(FakeOpenAIClient(response=decision([predicate_answer()])))

    with pytest.raises(ValueError, match="1 answers for 2 questions"):
        backend.ask(STATE, {"q0": {"type": "noul", "instructions": "x"}, "q1": {"type": "noul", "instructions": "y"}})


def test_bool_choice_values_are_stringified_at_the_boundary():
    # Legal on the wire, but this repo's ChoiceAnswer is str-keyed. model_construct skips the
    # strict-str validation the real endpoint's parse would apply -- the conversion must not care.
    raw = Decision.model_construct(
        answers=[
            AnswerAnswerResourceChoice.model_construct(
                type="choice",
                name=None,
                choice=True,
                probabilities=[
                    AnswerAnswerResourceChoiceProbability.model_construct(value=True, probability=0.7),
                    AnswerAnswerResourceChoiceProbability.model_construct(value=False, probability=0.3),
                ],
                confidence=0.7,
            )
        ],
        model="gpt-6-luna",
        usage=None,
    )
    backend = OpenAIBackend(FakeOpenAIClient(response=raw))

    response = backend.ask(
        STATE, {"q0": {"type": "choice", "instructions": "which?", "criteria": {"True": "a", "False": "b"}}}
    )

    assert response.answers["q0"] == ChoiceAnswer(choice="True", probabilities={"True": 0.7, "False": 0.3})


def test_a_provider_answer_object_never_reaches_the_envelope():
    # Boundary enforcement lives in DecisionResponse; this pins that the *provider's* class is what
    # it rejects -- i.e. conversion is mandatory, not cosmetic.
    with pytest.raises(ValueError, match="not a backend Answer"):
        DecisionResponse(answers={"q0": predicate_answer()}, model="gpt-6-luna")


# --------------------------------------------------------------------------- question conversion


def test_noul_criteria_fold_into_the_predicate_instructions():
    # This wire has no criteria field for a predicate; the wording must survive the trip.
    backend = OpenAIBackend(FakeOpenAIClient())

    backend.ask(
        STATE,
        {
            "q0": {
                "type": "noul",
                "instructions": "relevant?",
                "criteria": {"true": "it states it", "false": "topic only"},
            }
        },
    )

    question = backend._client.decisions.calls[0]["questions"][0]
    assert question["type"] == "predicate"
    assert question["name"] == "q0"
    assert question["instructions"] == "relevant? it states it answers yes; topic only answers no."


def test_typesafe_question_objects_cross_the_same_conversion():
    # The rerankers build dict questions; SDK objects with the same fields must convert identically.
    backend = OpenAIBackend(FakeOpenAIClient())

    backend.ask(STATE, {"q0": Noul(instructions="relevant?", criteria={"true": "yes-it", "false": "no-not"})})

    question = backend._client.decisions.calls[0]["questions"][0]
    assert question["instructions"] == "relevant? yes-it answers yes; no-not answers no."


def test_choice_criteria_keep_dict_order_and_omit_empty_descriptions():
    # Position bias is vendor-universal: order is preserved, never sorted. The None description is
    # the label-prior probe's contentless form -- the option must stay present.
    backend = OpenAIBackend(FakeOpenAIClient())

    backend.ask(
        STATE, {"q0": {"type": "choice", "instructions": "which?", "criteria": {"d2": "c", "d0": None, "d1": "a"}}}
    )

    assert backend._client.decisions.calls[0]["questions"][0]["choices"] == [
        {"value": "d2", "description": "c"},
        {"value": "d0"},
        {"value": "d1", "description": "a"},
    ]


def test_score_criteria_become_positionally_labelled_levels():
    backend = OpenAIBackend(FakeOpenAIClient())

    backend.ask(STATE, {"q0": {"type": "score", "instructions": "how much?", "criteria": ["lo", "mid", "hi"]}})

    assert backend._client.decisions.calls[0]["questions"][0]["levels"] == [
        {"label": "L0", "description": "lo"},
        {"label": "L1", "description": "mid"},
        {"label": "L2", "description": "hi"},
    ]


def test_structured_instructions_arrive_as_compact_json_text():
    import json

    backend = OpenAIBackend(FakeOpenAIClient())

    backend.ask(STATE, {"q0": {"type": "noul", "instructions": {"question": "rel?", "candidate": "doc"}}})

    rendered = backend._client.decisions.calls[0]["questions"][0]["instructions"]
    assert json.loads(rendered) == {"candidate": "doc", "question": "rel?"}


def test_state_travels_as_one_user_input_json_item():
    backend = OpenAIBackend(FakeOpenAIClient())

    backend.ask({"query": "q", "extra": [1, 2]}, {"q0": {"type": "noul", "instructions": "x"}})

    assert backend._client.decisions.calls[0]["input"] == [
        {"role": "user", "content": [{"type": "input_json", "data": {"query": "q", "extra": [1, 2]}}]}
    ]


def test_questions_this_wire_cannot_ask_reject_before_any_call():
    client = FakeOpenAIClient()
    backend = OpenAIBackend(client)

    with pytest.raises(ValueError, match="speaks noul, choice and score"):
        backend.ask(STATE, {"q0": {"type": "box", "instructions": "where?"}})
    with pytest.raises(ValueError, match="non-empty list of level"):
        backend.ask(STATE, {"q0": {"type": "score", "instructions": "x", "criteria": []}})
    with pytest.raises(ValueError, match="not str"):
        backend.ask(STATE, {"q0": {"type": "choice", "instructions": "x", "criteria": {1: "one"}}})
    assert client.decisions.calls == [], "a wiring bug must not spend a request"


# --------------------------------------------------------------------------- envelope and provenance


def test_usage_flattens_to_int_or_none_with_no_nested_dicts():
    backend = OpenAIBackend(FakeOpenAIClient())

    response = backend.ask(STATE, {"q0": {"type": "noul", "instructions": "x"}})

    assert response.usage == {
        "input_tokens": 100,
        "cached_tokens": 64,
        "cache_write_tokens": 0,
        "output_tokens": 3,
        "total_tokens": 103,
    }
    assert all(value is None or isinstance(value, int) for value in response.usage.values())


def test_provenance_reports_the_model_that_answered():
    backend = OpenAIBackend(FakeOpenAIClient(response=decision([predicate_answer()], model="gpt-6-luna-2026-11")))

    response = backend.ask(STATE, {"q0": {"type": "noul", "instructions": "x"}})

    assert response.model == "gpt-6-luna-2026-11"
    assert response.request_id is None, "a client that attaches no headers must not break the call"


# --------------------------------------------------------------------------- transport errors


@pytest.mark.parametrize(
    "error",
    [
        APIConnectionError(request=httpx2.Request("POST", "https://api.example/v1/decisions")),
        APITimeoutError(request=httpx2.Request("POST", "https://api.example/v1/decisions")),
        status_error(RateLimitError, 429),
        status_error(InternalServerError, 500),
    ],
    ids=["connection", "timeout", "rate-limit", "server-error"],
)
def test_unreachable_or_refused_is_classified_as_transport(error):
    backend = OpenAIBackend(FakeOpenAIClient(error=error))

    assert isinstance(error, backend.transport_errors)


@pytest.mark.parametrize(
    "error", [status_error(BadRequestError, 400), status_error(APIStatusError, 401)], ids=["400", "401"]
)
def test_a_bad_request_is_a_wiring_bug_not_a_transport_failure(error):
    assert not isinstance(error, OpenAIBackend.transport_errors)


# --------------------------------------------------------------------------- the injected client


def test_never_builds_a_client_of_its_own():
    # The API key belongs to the caller's client; this class must not read the environment.
    with pytest.raises(TypeError):
        OpenAIBackend()  # type: ignore[call-arg]


def test_reuses_one_client_across_calls():
    client = FakeOpenAIClient()
    backend = OpenAIBackend(client)

    backend.ask(STATE, {"q0": {"type": "noul", "instructions": "x"}})
    backend.ask(STATE, {"q0": {"type": "noul", "instructions": "x"}})

    assert len(client.decisions.calls) == 2


def test_close_leaves_an_injected_client_open():
    client = FakeOpenAIClient()

    with OpenAIBackend(client) as backend:
        backend.ask(STATE, {"q0": {"type": "noul", "instructions": "x"}})

    assert client.closed is False


# --------------------------------------------------------------------------- model configuration


def test_sends_the_default_model_when_none_is_configured():
    client = FakeOpenAIClient()

    OpenAIBackend(client).ask(STATE, {"q0": {"type": "noul", "instructions": "x"}})

    assert client.decisions.calls[0]["model"] == DEFAULT_MODEL == "gpt-6-luna"


def test_forwards_its_configured_model_to_every_call():
    client = FakeOpenAIClient()
    backend = OpenAIBackend(client, model="gpt-6-luna-next")

    backend.ask(STATE, {"q0": {"type": "noul", "instructions": "x"}})
    backend.ask(STATE, {"q0": {"type": "noul", "instructions": "x"}})

    assert [call["model"] for call in client.decisions.calls] == ["gpt-6-luna-next"] * 2
