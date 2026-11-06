"""Tests for `BaseDecisionBackend` (request contract, model configuration, lifecycle, envelope).

Offline by design: the base is exercised against a stub subclass that records what reached `_ask`;
no server and no provider behaviour is involved anywhere. `OpenAIBackend` does not exist yet; these
tests pin the contract it will have to satisfy, not any provider's specifics. The concrete
`TypeSafeBackend` tests live in `test_backend_typesafe.py`.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import sys

import pytest
from typesafe_sdk import Noul
from typesafe_sdk import NoulAnswer as SDKNoulAnswer

from decision_as_reranker.backend import base as backend_module
from decision_as_reranker.backend.base import BaseDecisionBackend, DecisionResponse, NoulAnswer

QUESTIONS = {"q0": Noul(instructions="relevant?")}
STATE = {"query": "what matters"}


def make_response(count=1, model="stub-1"):
    answers = {f"q{i}": NoulAnswer(noul=0.5) for i in range(count)}
    return DecisionResponse(answers=answers, model=model, usage={"input_tokens": 10, "output_tokens": 1})


class StubBackend(BaseDecisionBackend):
    """Minimal concrete backend: the base does all the work, the stub only stands in for a transport."""

    transport_errors = (ConnectionError,)

    def __init__(self, *, model=None, response=None, error=None):
        super().__init__(model=model)
        self.response = response if response is not None else make_response()
        self.error = error
        self.calls: list[dict] = []
        self.closed = False

    def _ask(self, state, questions):
        self.calls.append({"state": state, "questions": questions, "model": self.model})
        if self.error is not None:
            raise self.error
        return self.response

    def close(self):
        self.closed = True


class BareBackend(BaseDecisionBackend):
    """Backend that inherits both defaults: empty `transport_errors` and the no-op `close`."""

    def _ask(self, state, questions):
        return make_response()


# --------------------------------------------------------------------------- construction


def test_base_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseDecisionBackend()  # type: ignore[abstract]


def test_transport_errors_default_to_the_empty_tuple():
    # `except backend.transport_errors` must be valid syntax/semantics even before a subclass
    # names anything -- an empty tuple classifies nothing as a transport failure.
    assert BaseDecisionBackend.transport_errors == ()
    try:
        raise ValueError("a wiring bug")
    except BareBackend.transport_errors:
        pytest.fail("the default classification caught a non-transport error")
    except ValueError:
        pass


# --------------------------------------------------------------------------- delegation


def test_ask_forwards_state_and_questions_to_the_transport_once():
    backend = StubBackend()

    response = backend.ask(STATE, QUESTIONS)

    assert response is backend.response
    assert len(backend.calls) == 1
    assert backend.calls[0]["state"] == STATE
    assert backend.calls[0]["questions"] == QUESTIONS


def test_backend_failure_propagates_unmodified():
    # The adapter never converts a failure into a response; classification is by exception
    # identity (`transport_errors`), so wrapping here would break the reranker's on_error policy.
    backend = StubBackend(error=ConnectionError("down"))

    with pytest.raises(ConnectionError):
        backend.ask(STATE, QUESTIONS)


# --------------------------------------------------------------------------- model configuration


def test_ask_takes_no_per_call_model():
    # The model is constructor config: a different model is a different serving instance, hence a
    # different backend. Pinned as an absence so the kwarg cannot creep back into one side of the
    # seam alone -- the split that produced this bug twice.
    backend = StubBackend(model="backend-default")

    with pytest.raises(TypeError):
        backend.ask(STATE, QUESTIONS, model="call-override")
    assert backend.calls == [], "the rejection must happen before any transport call"


def test_the_constructor_model_reaches_the_transport():
    backend = StubBackend(model="backend-default")
    backend.ask(STATE, QUESTIONS)

    assert backend.calls[0]["model"] == "backend-default"


def test_no_model_configured_defers_to_the_transport_with_none():
    backend = StubBackend()
    backend.ask(STATE, QUESTIONS)

    assert backend.calls[0]["model"] is None, "the base must not invent a model name"


# --------------------------------------------------------------------------- request contract


def test_stateless_request_is_rejected_before_any_call():
    backend = StubBackend()

    with pytest.raises(ValueError, match="state"):
        backend.ask(None, QUESTIONS)
    assert backend.calls == [], "a wiring bug must not reach the network"


def test_empty_questions_is_rejected_before_any_call():
    backend = StubBackend()

    with pytest.raises(ValueError, match="non-empty"):
        backend.ask(STATE, {})
    assert backend.calls == []


@pytest.mark.parametrize("question_id", ["", 0, None])
def test_question_ids_must_be_non_empty_strings(question_id):
    backend = StubBackend()

    with pytest.raises(ValueError, match="question id"):
        backend.ask(STATE, {question_id: Noul(instructions="?")})
    assert backend.calls == []


# --------------------------------------------------------------------------- lifecycle


def test_context_manager_closes_a_backend_that_overrides_close():
    backend = StubBackend()
    with backend as entered:
        assert entered is backend
    assert backend.closed


def test_default_close_is_a_no_op():
    class DefaultCloseBackend(BaseDecisionBackend):
        def _ask(self, state, questions):
            return make_response()

    with DefaultCloseBackend() as backend:
        assert backend.ask(STATE, QUESTIONS).model == "stub-1"


# --------------------------------------------------------------------------- the envelope


def test_response_carries_provenance_alongside_the_answers():
    response = make_response(count=2, model="kev-latest")

    assert sorted(response.answers) == ["q0", "q1"]
    assert response.model == "kev-latest"
    assert response.usage == {"input_tokens": 10, "output_tokens": 1}
    assert response.request_id is None, "optional fields default to None, never to a fake value"


def test_response_is_frozen():
    response = make_response()
    with pytest.raises(dataclasses.FrozenInstanceError):
        response.model = "other"  # type: ignore[misc]


def test_response_without_a_model_name_is_rejected():
    # Provenance is an invariant, not a nicety: every backend must say what answered.
    with pytest.raises(ValueError, match="model"):
        DecisionResponse(answers={"q0": NoulAnswer(noul=0.5)}, model="")


@pytest.mark.parametrize("bad_key", ["", 7])
def test_response_answer_keys_must_be_non_empty_strings(bad_key):
    with pytest.raises(ValueError, match="answer key"):
        DecisionResponse(answers={bad_key: NoulAnswer(noul=0.5)}, model="stub-1")


def test_provider_answer_objects_are_rejected_at_the_envelope():
    # The SDK's identically-named class is transport output, not the contract: a backend that
    # forgets to convert fails here, loudly, instead of leaking the provider type upstream.
    with pytest.raises(ValueError, match="not a backend Answer"):
        DecisionResponse(answers={"q0": SDKNoulAnswer(noul=0.5)}, model="stub-1")


def test_an_empty_answers_mapping_is_allowed():
    # Completeness is the reranker's `on_error` policy, not the envelope's: a backend that
    # answered nothing must fail through the policy path, not as a construction-time ValueError.
    assert DecisionResponse(answers={}, model="stub-1").answers == {}


# --------------------------------------------------------------------------- the optional-dependency seam
#
# `typesafe_sdk` is an extra, not a hard dependency, and `base.py` names it nowhere -- not at
# runtime, not under TYPE_CHECKING. These tests pin what an app that never installs the extra may
# rely on: importing this module, subclassing the base, and naming `Question` in an annotation all
# work without it, and the alias is the same dict form whether or not it is present. What the
# concrete `TypeSafeBackend` does when the extra is missing is pinned in test_backend_typesafe.py.


@pytest.fixture
def without_sdk(monkeypatch):
    """Make `import typesafe_sdk` fail, as if the extra were absent."""
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)


@pytest.mark.parametrize("sdk_present", [True, False])
def test_question_is_the_neutral_dict_form_whatever_is_installed(sdk_present, request):
    # No lazy union, no SDK members: widening to the SDK's question classes is TypeSafeBackend's
    # business in its own module, so the base alias must not move when the extra comes and goes.
    if not sdk_present:
        request.getfixturevalue("without_sdk")

    assert backend_module.Question == collections.abc.Mapping[str, object]


def test_question_stays_usable_in_a_runtime_evaluated_annotation_without_the_sdk(without_sdk):
    # The consumer pattern that broke before: a module without `from __future__ import annotations`
    # evaluates its signatures at def time, so `Mapping[str, Question]` must not demand the SDK.
    question = backend_module.Question

    assert collections.abc.Mapping[str, question] is not None
