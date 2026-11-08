"""The TypeSafe-compatible backend: the System One contract over `typesafe_sdk`.

One concrete adapter in its own file, per the layout `backend.base` describes: the base owns the
request contract, this module owns the transport call and the SDK -> neutral conversions. On the
request side by widening the base's `Question` alias with the SDK's question classes (checkers
only -- the SDK accepts its objects and the dict form alike), on the answer side by field-copying
SDK answers into the neutral dataclasses. `typesafe_sdk` is an optional dependency (the `typesafe`
extra).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, TypeAlias

from typesafe_sdk import (
    Choice,
    Noul,
    Score,
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeClient,
)

from decision_as_reranker.backend.base import (
    Answer,
    BaseDecisionBackend,
    ChoiceAnswer,
    DecisionResponse,
    NoulAnswer,
    Question,
    ScoreAnswer,
)

#: The base contract plus the SDK's own question objects, which `system_one` accepts directly.
#: Widened here, not in `base`: this module is the only place `typesafe_sdk` may be named.
TypeSafeQuestion: TypeAlias = Noul | Choice | Score | Question


def _to_backend_answer(answer: object) -> Answer:
    """Convert one SDK answer into its neutral counterpart.

    `DecisionResponse` rejects the provider's identically-named classes, so every answer crosses
    the boundary by field copy, not by identity. `confidence` is deliberately not carried over --
    it is never a ranking input (see the module docstring).

    Args:
        answer: The SDK's `NoulAnswer`, `ChoiceAnswer`, or `ScoreAnswer`.

    Returns:
        The neutral dataclass holding the same values.

    Raises:
        ValueError: The answer is a kind this repo cannot represent (e.g. `box`).
    """
    from typesafe_sdk import ChoiceAnswer as SDKChoiceAnswer
    from typesafe_sdk import NoulAnswer as SDKNoulAnswer
    from typesafe_sdk import ScoreAnswer as SDKScoreAnswer

    if isinstance(answer, SDKNoulAnswer):
        return NoulAnswer(noul=answer.noul)
    if isinstance(answer, SDKChoiceAnswer):
        return ChoiceAnswer(choice=answer.choice, probabilities=answer.probabilities)
    if isinstance(answer, SDKScoreAnswer):
        return ScoreAnswer(score=answer.score, probabilities=answer.probabilities, legend=answer.legend)
    raise ValueError(f"cannot convert a {type(answer).__name__}: this repo ranks on noul, choice and score only")


def _request_id_or_none(response: SystemOneResponse) -> str | None:
    """Read the request id without letting its absence break a compatible endpoint.

    The SDK raises `TypeSafeError` when the `x-typesafe-request-id` header is missing -- normal for
    self-hosted endpoints -- and the field is logging-only, so `None` is the honest value.

    Args:
        response: The SDK's parsed response.

    Returns:
        The request id, or `None` when the endpoint did not send one.
    """
    from typesafe_sdk import TypeSafeError

    try:
        return response.request_id
    except TypeSafeError:
        return None


class TypeSafeBackend(BaseDecisionBackend):
    """A backend for any endpoint speaking the TypeSafe System One contract.

    Hosted Jev and every compatible server -- self-hosted Kev, Microsoft Decision, Cloudflare
    Clef, or a gateway in front of any of them -- are the same call (`POST {base_url}/v1/systemone`)
    behind a different `base_url` and `model`, so the swap is configuration, never code. The SDK's
    answers are converted to this repo's neutral dataclasses here, at the boundary.

    The client is injected, never constructed here: endpoint configuration (base url, credentials,
    HTTP options) is one caller-side decision, and a backend that cannot build a client cannot
    misconfigure one. It is held for the backend's whole life and reused by every call: reranking
    costs `queries x candidates` calls, and a client per call pays a fresh TLS handshake on each.

    Args:
        client: A configured `TypeSafeClient` to call. Required.
        model: The model every call uses. `None` defers to the client's own default (an env var,
            then the SDK's `jev-latest`).
    """

    # `TypeSafeAPIError` (status family) and `TypeSafeAPIConnectionError` (no-response family) are
    # siblings under `TypeSafeError`, so both must be named; `TypeSafeAPITimeoutError` subclasses
    # the connection error and is covered.
    transport_errors: ClassVar[tuple[type[BaseException], ...]] = (TypeSafeAPIError, TypeSafeAPIConnectionError)

    def __init__(self, client: TypeSafeClient, *, model: str | None = None) -> None:
        super().__init__(model=model)
        self._client = client

    def _ask(self, state: Mapping[str, object], questions: Mapping[str, TypeSafeQuestion]) -> DecisionResponse:
        response = self._client.system_one(state, questions, model=self.model)
        return DecisionResponse(
            answers={question_id: _to_backend_answer(a) for question_id, a in response.answers.items()},
            model=response.model,
            usage=response.usage.model_dump() if response.usage is not None else None,
            request_id=_request_id_or_none(response),
        )
