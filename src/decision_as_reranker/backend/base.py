"""Backend adapters: one interface, many decision-serving instances.

`BaseDecisionBackend` is the seam between this repo and *whatever* answers decision questions. The
reranker calls `ask(state, questions)` and gets back a `DecisionResponse`; it never touches an SDK
client, an HTTP stack, or a provider's exception classes directly. Concrete backends
(`TypeSafeBackend`, `OpenAIBackend`, ...) implement `_ask` alone -- the transport call and the
mapping of the provider's reply onto `DecisionResponse`. Everything mechanical (request validation,
provenance logging, lifecycle) lives in the base and is not reimplemented per backend.

The answer types are defined *here*, not borrowed from any SDK: noul, choice and score are this
repo's domain vocabulary, and the classes below are that vocabulary in its neutral form. A concrete
backend converts whatever its transport returns -- a provider SDK's identically-named answer models,
an OpenAI-compatible server's JSON, anything -- into these dataclasses before building a
`DecisionResponse`, which rejects any other value at construction. Nothing above this layer imports
a provider's answer classes; `http://localhost:8000/v1/chat/completions` producing a `ChoiceAnswer`
is a valid backend.

The neutral answers carry no `confidence` field. It is deliberately absent, not deferred: for a
`choice` it is a transform of the top probability (no independent signal), for a `noul` it does not
exist, and a field that is never a legitimate ranking input should not sit in the contract waiting
to be misused. Bounds are *not* validated here either -- an off-contract value must surface as the
reranker's "backend is off-contract" error, so these classes are dumb containers.

Two invariants carry over from the reranker:

* A failed call is never turned into a score. `ask` propagates whatever the transport raised;
  `transport_errors` names which of those exceptions mean "the serving instance could not be
  reached or refused", so the reranker's `on_error` policy can classify them without importing
  any provider's exception hierarchy.
* Completeness is *not* validated here. A backend that answers none of the questions it was asked
  is broken, but "broken backend" is exactly what `on_error` decides how to treat -- raising at
  the envelope would bypass that policy and surface a transport problem as a programming error.
  The reranker's per-question handling stays the single place where a missing answer is judged.
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import ClassVar, TypeAlias

from typing_extensions import Self

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class NoulAnswer:
    """P(yes) for one yes/no question. The value itself is the relevance; no option names."""

    noul: float


@dataclass(frozen=True)
class ChoiceAnswer:
    """One probability per option. `choice` (the argmax name) is carried for logging only."""

    choice: str
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class ScoreAnswer:
    """The probability-weighted mean level index, plus the distribution that produced it.

    `probabilities` is keyed by level index; `legend` is the optional level descriptions, used only
    to count levels when `probabilities` is empty. Normalisation onto `[0, 1]` is the reranker's job.
    """

    score: float
    probabilities: Mapping[int, float]
    legend: Mapping[int, object] = field(default_factory=dict)


#: Anything a backend accepts as a question: the neutral dict form. Part of the request contract,
#: so it is defined here and imported by the reranker, not the other way round. Provider-native
#: question objects are deliberately *not* named here -- the backend that speaks that protocol
#: accepts them too and widens the alias in its own module. Nothing dispatches on the alias at
#: runtime; it documents the shape `ask` validates and `_ask` receives.
Question: TypeAlias = Mapping[str, object]


#: The answers a backend must produce. See the module docstring: these are the domain primitives.
Answer: TypeAlias = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class DecisionResponse:
    """One backend's answer to one request, in a form the reranker can parse without knowing
    which backend produced it.

    `model` is required, not optional: provenance is an invariant (a threshold tuned against an
    alias drifts when the alias moves), so every backend must be able to say what answered.
    `usage` and `request_id` are for logging only -- no ranking decision may read them.
    """

    answers: Mapping[str, Answer]
    model: str
    usage: Mapping[str, int | None] | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        # ValueError, not TypeError, by contract: these checks police the *request/response*
        # boundary (this repo's convention -- every off-contract value raises ValueError, tested
        # as such), not a mis-typed call site.
        if not isinstance(self.answers, Mapping):
            raise ValueError(f"answers must be a mapping, got {type(self.answers).__name__}")  # noqa: TRY004
        for question_id, answer in self.answers.items():
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"answer key {question_id!r} is not a non-empty str")
            if not isinstance(answer, (NoulAnswer, ChoiceAnswer, ScoreAnswer)):
                raise ValueError(  # noqa: TRY004
                    f"answer for {question_id!r} is {type(answer).__name__}, not a backend Answer; "
                    "convert the provider's reply in the backend, not upstream of it"
                )
        if not isinstance(self.model, str) or not self.model:
            raise ValueError(f"model must be a non-empty str, got {self.model!r}")


class BaseDecisionBackend(abc.ABC):
    """Request contract and lifecycle layer over a decision-serving instance.

    Concrete about the call shape (`ask`), abstract about the transport (`_ask`). A subclass
    answers one `POST`-equivalent and maps the reply to `DecisionResponse`; it should not
    re-validate the request or write its own context-manager plumbing.

    Args:
        model: The model every call uses; `None` defers to the transport's own default. Configured
            once here, never per call -- a different model is a different serving instance, so it
            is a different backend (construct one sharing the same client to make the swap free).
    """

    #: Exception classes that mean "the serving instance could not be reached or refused the
    #: request". The reranker's `on_error` policy catches exactly these; anything else escaping
    #: `_ask` is a programming error and must propagate regardless of policy. Subclasses fill
    #: this in with their provider's refusal and unreachability classes -- the status-error and
    #: connection-error family of whichever SDK or HTTP stack the backend wraps.
    #: An empty tuple -- the default -- means the backend classifies nothing as transport failure.
    transport_errors: ClassVar[tuple[type[BaseException], ...]] = ()

    def __init__(self, *, model: str | None = None) -> None:
        self.model = model

    def ask(
        self,
        state: Mapping[str, object],
        questions: Mapping[str, Question],
    ) -> DecisionResponse:
        """Send one decision request and return the typed answers.

        Validates the contract every backend must enforce identically, then delegates to `_ask`.
        The model is whatever this backend was constructed with; there is no per-call override.

        Args:
            state: Context shared by every question in the request (e.g. the query).
            questions: Question id to question; must be non-empty.

        Returns:
            The `DecisionResponse` produced by the transport.

        Raises:
            ValueError: `state` is `None`, `questions` is empty, or a question id is not a
                non-empty string -- a wiring bug, raised before any network call.
            BaseException: Whatever the transport raises; the classes named by `transport_errors`
                are the ones meaning "unreachable or refused".
        """
        if state is None:
            raise ValueError("state is required; a request without state has nothing to score")
        if not questions:
            raise ValueError("questions must be non-empty; there is nothing to ask")
        for question_id in questions:
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"question id {question_id!r} is not a non-empty str")

        response = self._ask(state, questions)
        # Provenance on the seam itself, so it is logged even by callers that do no ranking.
        logger.debug("answered by model=%s usage=%s", response.model, response.usage)
        return response

    @abc.abstractmethod
    def _ask(
        self,
        state: Mapping[str, object],
        questions: Mapping[str, Question],
    ) -> DecisionResponse:
        """Execute one request.

        The subclass's only required method: build the provider call -- passing `self.model`,
        whose `None` means "defer to the transport's own default" -- map the reply to
        `DecisionResponse`, and let provider exceptions propagate unmodified.

        Args:
            state: Context shared by every question in the request.
            questions: Question id to question; validated non-empty by `ask`.

        Returns:
            The provider's reply mapped onto a `DecisionResponse`.
        """

    def close(self) -> None:
        """Release whatever this backend owns.

        Backends holding a pooled client override it; stateless ones inherit the no-op.
        """

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "Answer",
    "BaseDecisionBackend",
    "ChoiceAnswer",
    "DecisionResponse",
    "NoulAnswer",
    "Question",
    "ScoreAnswer",
]
