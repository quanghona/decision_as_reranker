"""Tests for `TypeSafeBackend` (SDK -> neutral conversion, client injection, transport errors).

Offline by design: the backend is exercised against a fake `TypeSafeClient` injected into it and
returning real `SystemOneResponse` objects -- the SDK's own validation, no server. The backend
never constructs a client, so no factory or environment is patched anywhere. The base contract
these tests stand on is pinned in `test_backend_base.py`.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from typesafe_sdk import (
    Noul,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
    TypeSafeError,
)

from decision_as_reranker.backend.base import ChoiceAnswer, NoulAnswer, ScoreAnswer
from decision_as_reranker.backend.typesafe import TypeSafeBackend

QUESTIONS = {"q0": Noul(instructions="relevant?")}
STATE = {"query": "what matters"}


def sdk_response(answers=None, model="jev-1.14.0"):
    """A real `SystemOneResponse`, built through the SDK's own validation -- no server involved."""
    return SystemOneResponse.model_validate(
        {
            "model": model,
            "usage": {"input_tokens": 12, "output_tokens": 4},
            "answers": answers if answers is not None else {"q0": {"type": "noul", "noul": 0.75}},
        }
    )


class FakeTypeSafeClient:
    """Stands in for a `TypeSafeClient`: records calls, returns a canned response or raises.

    The canned response carries no HTTP headers, so its `request_id` raises `TypeSafeError` --
    exactly the situation on self-hosted endpoints that omit `x-typesafe-request-id`.
    """

    def __init__(self, response=None, error=None):
        self.response = response if response is not None else sdk_response()
        self.error = error
        self.calls: list[dict] = []
        self.closed = False

    def system_one(self, state, questions, *, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        if self.error is not None:
            raise self.error
        return self.response

    def close(self):
        self.closed = True


# --------------------------------------------------------------------------- conversion


def test_typesafe_backend_converts_every_answer_kind_to_the_neutral_contract():
    answers = {
        "n": {"type": "noul", "noul": 0.75},
        "c": {"type": "choice", "choice": "b", "confidence": 0.6, "probabilities": {"a": 0.2, "b": 0.8}},
        "s": {
            "type": "score",
            "score": 1.5,
            "confidence": 0.5,
            "probabilities": {0: 0.5, 1: 0.5},
            "legend": {0: "lo", 1: "hi"},
        },
    }
    backend = TypeSafeBackend(FakeTypeSafeClient(response=sdk_response(answers)))

    response = backend.ask(STATE, QUESTIONS)

    assert response.answers["n"] == NoulAnswer(noul=0.75)
    assert response.answers["c"] == ChoiceAnswer(choice="b", probabilities={"a": 0.2, "b": 0.8})
    assert response.answers["s"] == ScoreAnswer(score=1.5, probabilities={0: 0.5, 1: 0.5}, legend={0: "lo", 1: "hi"})


def test_typesafe_backend_keeps_provenance_and_survives_a_missing_request_id():
    # `confidence` is absent from the neutral answers by design; `model`/`usage` are logging-only.
    # The fake response has no headers, so reading `.request_id` would raise -- the backend maps
    # that to None instead of breaking calls to compatible endpoints that omit the header.
    backend = TypeSafeBackend(FakeTypeSafeClient(response=sdk_response(model="kev-latest")))

    response = backend.ask(STATE, QUESTIONS)

    assert not hasattr(response.answers["q0"], "confidence")
    assert response.model == "kev-latest"
    assert response.usage == {"input_tokens": 12, "output_tokens": 4}
    assert response.request_id is None


def test_typesafe_backend_rejects_answers_it_cannot_represent():
    answers = {
        "b": {"type": "box", "box": {"top": 0.1, "bottom": 0.9, "left": 0.2, "right": 0.8}, "confidence": 0.9},
    }
    backend = TypeSafeBackend(FakeTypeSafeClient(response=sdk_response(answers)))

    with pytest.raises(ValueError, match="cannot convert"):
        backend.ask(STATE, QUESTIONS)


# --------------------------------------------------------------------------- the injected client


def test_typesafe_backend_reuses_one_client_across_calls():
    # A client per call would pay a fresh TLS handshake on every (query, candidate) pair.
    client = FakeTypeSafeClient()
    backend = TypeSafeBackend(client)

    backend.ask(STATE, QUESTIONS)
    backend.ask(STATE, QUESTIONS)

    assert len(client.calls) == 2


def test_typesafe_backend_never_builds_a_client_of_its_own():
    # The client is a required argument: endpoint configuration is made once, by the caller, and a
    # backend that cannot construct a client cannot misconfigure or secretly reconfigure one.
    # Pinned as a missing-argument TypeError so the removed env-var factory cannot creep back as a
    # default.
    with pytest.raises(TypeError):
        TypeSafeBackend()  # type: ignore[call-arg]


def test_close_leaves_an_injected_client_open():
    # The endpoint config belongs to the caller; the backend may use it, not end it.
    client = FakeTypeSafeClient()

    with TypeSafeBackend(client) as backend:
        backend.ask(STATE, QUESTIONS)

    assert client.closed is False


# --------------------------------------------------------------------------- model configuration


def test_typesafe_backend_forwards_its_configured_model_to_every_call():
    client = FakeTypeSafeClient()
    backend = TypeSafeBackend(client, model="backend-default")

    backend.ask(STATE, QUESTIONS)
    backend.ask(STATE, QUESTIONS)

    assert [call["model"] for call in client.calls] == ["backend-default", "backend-default"]


def test_typesafe_backend_defers_to_the_client_default_when_no_model_is_configured():
    client = FakeTypeSafeClient()

    TypeSafeBackend(client).ask(STATE, QUESTIONS)

    assert client.calls[0]["model"] is None, "None must reach system_one as None, not an invented name"


def test_typesafe_backend_validates_before_touching_the_client():
    client = FakeTypeSafeClient()

    with pytest.raises(ValueError, match="non-empty"):
        TypeSafeBackend(client).ask(STATE, {})
    assert client.calls == []


# --------------------------------------------------------------------------- transport errors


def test_typesafe_transport_errors_cover_refusal_and_unreachability():
    # TypeSafeAPIError is the status family, TypeSafeAPIConnectionError the no-response one; they
    # are siblings under TypeSafeError, so naming only one would misclassify the other.
    assert TypeSafeBackend.transport_errors == (TypeSafeAPIError, TypeSafeAPIConnectionError)
    assert issubclass(TypeSafeAPITimeoutError, TypeSafeBackend.transport_errors)
    assert not issubclass(TypeSafeError, TypeSafeBackend.transport_errors)


def test_typesafe_api_failure_propagates_and_is_classified_as_transport():
    backend = TypeSafeBackend(FakeTypeSafeClient(error=TypeSafeAPIConnectionError("unreachable")))

    with pytest.raises(TypeSafeAPIConnectionError):
        backend.ask(STATE, QUESTIONS)
    try:
        backend.ask(STATE, QUESTIONS)
    except backend.transport_errors:
        pass  # the reranker's on_error policy must be able to classify it without SDK imports


# --------------------------------------------------------------------------- the optional-dependency seam
#
# `typesafe_sdk` is an extra, not a hard dependency, and `typesafe.py` is the one module that names
# it -- at module level. So the loud failure for a missing install is the import of that module
# itself, not anything inside `TypeSafeBackend`: the client is injected, so construction reaches for
# nothing. The base contract stays importable either way -- pinned in test_backend_base.py.


def test_importing_the_typesafe_module_without_the_sdk_fails_at_the_import(monkeypatch):
    # Poisoning `sys.modules` simulates the missing extra without uninstalling the dev-group copy;
    # `typesafe.py` names the SDK's question classes (the widened alias) and error classes
    # (transport_errors) at module level, so the re-import must raise. monkeypatch restores the
    # real modules on teardown.
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    monkeypatch.delitem(sys.modules, "decision_as_reranker.backend.typesafe")

    with pytest.raises(ImportError):
        importlib.import_module("decision_as_reranker.backend.typesafe")
