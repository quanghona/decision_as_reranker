"""The OpenAI Decisions backend: the same seam over `POST /v1/decisions`.

One concrete adapter in its own file, per the layout `backend.base` describes: the base owns the
request contract, this module owns the transport call and the OpenAI -> neutral conversions.
`openai` is an optional dependency (the `openai` extra); it is imported at module scope here and
nowhere else in the package, so the core stays importable without it.

Two structural differences from the System One contract, both handled here rather than upstream:

* Answers come back as an **ordered array**, not keyed by question id, and every answer's `name`
  may be `null`. The mapping is positional -- `zip(question ids, answers)` in the order the
  questions were built -- and `name` is only ever a cross-check, never a key.
* The yes/no primitive is called `predicate` there and `noul` here. It converts to the neutral
  `NoulAnswer` at this boundary, so nothing above the seam learns the word "predicate": the
  reranker ranks `noul`/`choice`/`score` whichever backend answered. A `refusal` converts to *no
  answer* -- the id is simply absent from `DecisionResponse.answers` -- never to `NoulAnswer(0.0)`,
  because a decline is not evidence of irrelevance; the reranker's `on_error` policy judges the gap.

Questions cross the boundary the other way: the repo's dict forms (`noul`/`choice`/`score` with
`instructions` and `criteria`) are rendered into OpenAI's `predicate`/`choice`/`score` dicts. A
`noul` has no criteria field on this wire, so the criterion wording is folded into `instructions`
from the module template below. `confidence` is dropped like everywhere else in this repo -- never
a ranking input.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError
from openai.types import decision as decision_models

from decision_as_reranker.backend.base import (
    Answer,
    BaseDecisionBackend,
    ChoiceAnswer,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
)

#: The only model the endpoint serves (as of the docs this file was written against). `None` on the
#: constructor defers to it; a different model is a different serving instance, hence a different
#: backend -- see `BaseDecisionBackend`.
DEFAULT_MODEL = "gpt-6-luna"

#: A `noul`'s `criteria` has no field on this wire, so the wording is folded into `instructions`.
#: Module constant so the phrasing is one auditable decision, not an inline string.
_NOUL_CRITERIA_TEMPLATE = " {true_text} answers yes; {false_text} answers no."
_NOUL_CRITERIA_TRUE_ONLY = " {text} answers yes."
_NOUL_CRITERIA_FALSE_ONLY = " {text} answers no."


def _render(value: object) -> str:
    """Render one instruction/criterion as the text this wire accepts.

    Args:
        value: A string, or JSON structure (`Mapping`/`Sequence`) the serving model reads as data.

    Returns:
        The string unchanged, or its compact JSON encoding.

    Raises:
        ValueError: `value` is not JSON-serialisable -- the question cannot cross this boundary.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"question text must be a str or JSON-serialisable, got {type(value).__name__}") from exc


def _to_openai_question(question_id: str, question: object) -> dict[str, Any]:
    """Convert one repo question into its OpenAI Decisions dict.

    Accepts the neutral dict form and anything carrying the same fields as a pydantic model (e.g. a
    `typesafe_sdk` `Noul`/`Choice`/`Score`), which is normalised by dump first. The repo id becomes
    the question `name`: it is echoed back and tests can assert order/name agreement, but the
    answer mapping stays positional -- names are optional on this API and may return `null`.

    Args:
        question_id: The repo-side id, used as the cross-check name only.
        question: The question to convert.

    Returns:
        A `{"type": ..., "name": ..., ...}` dict for the `questions` array.

    Raises:
        ValueError: The question is not a mapping, has an unknown `type`, or carries an option/level
            set this wire cannot express (empty, or a non-string choice option name).
    """
    if hasattr(question, "model_dump"):
        question = question.model_dump()  # type: ignore[attr-defined]
    if not isinstance(question, Mapping):
        raise ValueError(  # noqa: TRY004
            f"question {question_id!r} is {type(question).__name__}, not a mapping"
        )
    kind = question.get("type")
    instructions = _render(question.get("instructions", ""))
    criteria = question.get("criteria")

    if kind == "noul":
        return {"type": "predicate", "name": question_id, "instructions": instructions + _fold_criteria(criteria)}
    if kind == "choice":
        return {"type": "choice", "name": question_id, "instructions": instructions, "choices": _to_choices(criteria)}
    if kind == "score":
        return {"type": "score", "name": question_id, "instructions": instructions, "levels": _to_levels(criteria)}
    raise ValueError(f"question {question_id!r} has type {kind!r}; this backend speaks noul, choice and score")


def _fold_criteria(criteria: object) -> str:
    """Fold a `noul`'s true/false criteria into the instructions text this wire has room for.

    Args:
        criteria: The question's `criteria` mapping (`{"true": ..., "false": ...}`), or `None`.

    Returns:
        The appended sentence, or `""` when there is nothing to fold.
    """
    if not isinstance(criteria, Mapping) or not criteria:
        return ""
    true_text = criteria.get("true")
    false_text = criteria.get("false")
    if true_text is not None and false_text is not None:
        return _NOUL_CRITERIA_TEMPLATE.format(true_text=_render(true_text), false_text=_render(false_text))
    if true_text is not None:
        return _NOUL_CRITERIA_TRUE_ONLY.format(text=_render(true_text))
    if false_text is not None:
        return _NOUL_CRITERIA_FALSE_ONLY.format(text=_render(false_text))
    return ""


def _to_choices(criteria: object) -> list[dict[str, Any]]:
    """Turn a `choice` question's criteria into the ordered `choices` array.

    Dict order *is* the option order -- position bias is vendor-universal, so it is preserved, not
    sorted. A `None` description is omitted: that is the label-prior probe's contentless form.

    Args:
        criteria: Option name to description, or `None`.

    Returns:
        1+ `{"value": str, "description"?: str}` entries, in criteria order.

    Raises:
        ValueError: Criteria empty/not a mapping, or an option name is not a string (this repo's
            `ChoiceAnswer` is str-keyed; a non-string name cannot come back readable).
    """
    if not isinstance(criteria, Mapping) or not criteria:
        raise ValueError("a choice question needs non-empty criteria mapping option names to descriptions")
    choices: list[dict[str, Any]] = []
    for name, description in criteria.items():
        if not isinstance(name, str):
            raise ValueError(  # noqa: TRY004
                f"choice option name {name!r} is {type(name).__name__}, not str"
            )
        choice: dict[str, Any] = {"value": name}
        if description is not None:
            choice["description"] = _render(description)
        choices.append(choice)
    return choices


def _to_levels(criteria: object) -> list[dict[str, Any]]:
    """Turn a `score` question's criteria into ordered `levels`.

    Args:
        criteria: The level descriptions, lowest to highest.

    Returns:
        `{"label": "L{i}", "description": ...}` entries; labels are positional (`L0`, `L1`, ...)
        because the answer's probability array is keyed by index, not by any caller-side name.

    Raises:
        ValueError: Criteria empty or not a list of descriptions.
    """
    if isinstance(criteria, (str, bytes)) or not isinstance(criteria, (list, tuple)) or not criteria:
        raise ValueError("a score question needs criteria: a non-empty list of level descriptions")
    return [{"label": f"L{index}", "description": _render(description)} for index, description in enumerate(criteria)]


@dataclass(frozen=True)
class _Refused:
    """Marker for a question the model declined; the caller omits the id from the answers map."""


def _to_backend_answer(answer: object) -> Answer | type(_Refused):
    """Convert one OpenAI answer into its neutral counterpart.

    `DecisionResponse` rejects the provider's answer objects, so every answer crosses by field copy.
    `confidence` is deliberately not carried over -- it is never a ranking input (see the module
    docstring). A refusal is not an answer: the `_Refused` marker tells `_ask` to leave the id out.

    Args:
        answer: One element of `decision.answers`, in question order.

    Returns:
        The neutral dataclass, or the `_Refused` marker.

    Raises:
        ValueError: The answer is a kind this repo cannot represent.
    """
    if isinstance(answer, decision_models.AnswerAnswerResourcePredicate):
        return NoulAnswer(noul=answer.probability)
    if isinstance(answer, decision_models.AnswerAnswerResourceChoice):
        return ChoiceAnswer(
            choice=str(answer.choice),
            probabilities={str(entry.value): entry.probability for entry in answer.probabilities},
        )
    if isinstance(answer, decision_models.AnswerAnswerResourceScore):
        return ScoreAnswer(
            score=answer.score,
            probabilities={entry.value: entry.probability for entry in answer.probabilities},
            legend={entry.value: entry.label for entry in answer.probabilities},
        )
    if isinstance(answer, decision_models.AnswerAnswerResourceRefusal):
        return _Refused
    raise ValueError(f"cannot convert a {type(answer).__name__}: this repo ranks on noul, choice and score only")


def _flatten_usage(usage: object) -> dict[str, int | None]:
    """Flatten the SDK's nested usage object into the envelope's `Mapping[str, int | None]`.

    `cached_tokens`/`cache_write_tokens` are components of `input_tokens`, never extras -- they are
    reported side by side, not summed into it.

    Args:
        usage: The response's `usage` object, or `None`.

    Returns:
        Flat int-valued counts, or `{}` when the response carried no usage.
    """
    if usage is None:
        return {}
    details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "cached_tokens": getattr(details, "cached_tokens", None),
        "cache_write_tokens": getattr(details, "cache_write_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _request_id_or_none(response: object) -> str | None:
    """Read the transport's `x-request-id` without breaking on clients that expose no headers.

    Args:
        response: The parsed SDK response object.

    Returns:
        The request id, or `None` when the client did not attach headers to the object.
    """
    headers = getattr(response, "_response_headers", None)
    return headers.get("x-request-id") if headers is not None else None


class OpenAIBackend(BaseDecisionBackend):
    """A backend for OpenAI's Decisions endpoint (`POST {base_url}/decisions`, `gpt-6-luna`).

    The client is injected, never constructed here: the API key, base url and timeout belong to the
    caller's client configuration, and a backend that cannot build a client cannot misconfigure one
    -- this class never reads `OPENAI_API_KEY` or any other environment variable. The client is
    held for the backend's whole life and reused by every call: reranking costs
    `queries x candidates` questions, and a client per call pays a fresh TLS handshake on each.

    Args:
        client: A configured `OpenAI` client to call. Required. Its `base_url`, if set, must
            include `/v1` (the SDK appends only `/decisions`), and its default 600 s timeout
            belongs nowhere in a rerank path -- set it on the client.
        model: The model every call uses. `None` sends `DEFAULT_MODEL` (`gpt-6-luna`), the only
            model the endpoint serves.
    """

    # Connection/timeout (APITimeoutError subclasses APIConnectionError), 429 and 5xx are the
    # serving instance failing us. 400/401/403/404/422 are deliberately absent: they mean the
    # request or the key is wrong -- a wiring bug that must propagate regardless of `on_error`.
    transport_errors: ClassVar[tuple[type[BaseException], ...]] = (
        APIConnectionError,
        RateLimitError,
        InternalServerError,
    )

    def __init__(self, client: OpenAI, *, model: str | None = None) -> None:
        super().__init__(model=model)
        self._client = client

    def _ask(self, state: Mapping[str, object], questions: Mapping[str, object]) -> DecisionResponse:
        ids = list(questions)
        decision = self._client.decisions.create(
            model=self.model or DEFAULT_MODEL,
            input=[{"role": "user", "content": [{"type": "input_json", "data": dict(state)}]}],
            questions=[_to_openai_question(question_id, questions[question_id]) for question_id in ids],
        )
        if len(decision.answers) != len(ids):
            raise ValueError(
                f"backend returned {len(decision.answers)} answers for {len(ids)} questions; "
                "answers map positionally, so a disagreement cannot be resolved"
            )
        answers: dict[str, Answer] = {}
        for question_id, answer in zip(ids, decision.answers):
            converted = _to_backend_answer(answer)
            if converted is not _Refused:
                answers[question_id] = converted
        return DecisionResponse(
            answers=answers,
            model=decision.model,
            usage=_flatten_usage(decision.usage),
            request_id=_request_id_or_none(decision),
        )


__all__ = ["DEFAULT_MODEL", "OpenAIBackend"]
