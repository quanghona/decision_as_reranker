#!/usr/bin/env -S uv run python
"""Serve a decision model as a vLLM-compatible reranking endpoint.

One process, one decision model, the HTTP surface vLLM's reranker exposes: `POST /rerank`,
`POST /v1/rerank`, `POST /v2/rerank` with the Cohere/Jina request and response shapes, plus the
discovery routes a RAG framework probes first (`/health`, `/version`, `/v1/models`). Point any
client that speaks that dialect -- LlamaIndex, LangChain, a hand-rolled `requests.post` -- at this
server instead of `vllm serve`, and the ranking behind the URL is a typed decision model rather
than a cross-encoder.

    {"query": "...", "documents": ["...", "..."], "top_n": 5}
        -> {"id": "rerank-...", "model": "...", "usage": {...},
            "results": [{"index": 1, "relevance_score": 0.87, "document": {"text": "..."}}]}

`results` is sorted by relevance descending and `index` is the document's position in the request,
so a client can always recover the input order -- and, when some indices are missing, tell that
those documents went unscored rather than scoring last.

The two layers this script joins:

* `decision_as_reranker.reranker` owns the questions, the parsing and the ordering. This file
  supplies none of that; it translates HTTP in and HTTP out.
* `decision_as_reranker.backend` owns the transport. The client is built once per process here and
  shared by every request -- a client per request would pay a fresh TLS handshake on each rerank.

Inherited invariants, restated because an HTTP surface is where they get violated:

* A failure is an HTTP error, never a score. An unreachable or refusing backend is `502`; with
  `--on-error drop` the affected documents are *absent* from `results` (their `index` is missing),
  never returned as `relevance_score: 0.0`.
* Scores are model-bound. The `model` field of every response is the id that actually answered,
  read off the backend's reply -- not the string in the request, which is echoed nowhere. A
  threshold or a ranking tuned against `kev-latest` does not transfer when the alias moves.
* Concurrency is bounded by `--max-concurrency`, because one rerank is not one call: `noul` sends
  `K` questions in a single request, `choice` sends two requests whenever the label prior is
  measured (`k` content questions in one, the matching `k` probes in the other).

What a shortlist costs, per rerank:

| primitive | requests | passage copies per request |
| --- | --- | --- |
| `noul` | 1 | `K` |
| `choice`, `alpha=0` | 1 | `K` |
| `choice`, `alpha>0` | 2 | `K` |

Context is the hard limit for both: every request carries the whole shortlist, so `--max-documents`
rejects a shortlist that cannot fit rather than letting the backend refuse it mid-ranking. Splitting
a long shortlist across requests is the documented extension point -- override `ask` on the
reranker, and the parsing and ordering here still apply.

Examples:
    Local Kev, per-candidate `noul` (the TypeSafe cookbook's shape):

        uv run serve_vllm.py --backend typesafe --endpoint http://localhost:8009 --model kev-latest

    The same shortlist ranked by one joint `choice`, averaged over 4 orderings:

        uv run serve_vllm.py --primitive choice --k 4 --alpha 0.75

    OpenAI's Decisions endpoint:

        uv run serve_vllm.py --backend openai --model gpt-6-luna --api-key $OPENAI_API_KEY

    Then, from any vLLM-aware client:

        curl -s localhost:8000/v1/rerank -H 'content-type: application/json' \\
             -d '{"query": "what is the capital of France?", "documents": ["Paris is the capital of France.", "Berlin is German."], "top_n": 2}'
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from decision_as_reranker.backend.base import BaseDecisionBackend, DecisionResponse
from decision_as_reranker.reranker import (
    BaseDecisionReranker,
    DecisionChoiceReranker,
    DecisionNoulReranker,
    OnError,
)

logger = logging.getLogger("serve_vllm")

#: Defaults for the two primitives, kept here so the served wording is one auditable decision.
#: The text names `query` and `candidate` because those are the values `state` and the question carry.
DEFAULT_NOUL_INSTRUCTIONS = "Could the candidate passage be what the query asks for?"
DEFAULT_NOUL_TRUE = "It states the specific fact or rule the query is asking about."
DEFAULT_NOUL_FALSE = "It merely shares the query's topic."
DEFAULT_CHOICE_INSTRUCTIONS = "Which of these candidate passages is the one the query asks for?"

#: `base_url` conventions differ and both are wrong in the obvious way: the TypeSafe SDK appends
#: `/v1/systemone` itself, the OpenAI SDK appends only `/decisions`.
TYPESAFE_DEFAULT_URL = "http://localhost:8009"
OPENAI_DEFAULT_URL = "https://api.openai.com/v1"
TYPESAFE_DEFAULT_MODEL = "kev-latest"
OPENAI_DEFAULT_MODEL = "gpt-6-luna"

Primitive = Literal["noul", "choice"]
BackendKind = Literal["typesafe", "openai"]


# ---------------------------------------------------------------------------------------------- HTTP schema


class RerankDocument(BaseModel):
    """The object form vLLM/Cohere accept for a document: `{"text": "..."}`.

    Multimodal `{"content": [...]}` inputs are out of scope -- a decision model answers questions
    about text, so a non-text document is a client error rather than a degraded score.
    """

    text: str


class RerankRequest(BaseModel):
    """`POST /rerank` body, as vLLM defines it.

    Unknown fields are ignored on purpose: Cohere and Jina clients send `max_chunks_per_doc`,
    `overlap_tokens` and `truncate_prompt_tokens`, none of which mean anything to a decision model,
    and rejecting them would break the drop-in compatibility this server exists for.

    Attributes:
        model: The client's notion of which model is served. Not used for routing -- the answering
            model is fixed by this process's configuration -- and never echoed back.
        query: The information need the documents were retrieved against.
        documents: The passages to rank, in input order.
        top_n: How many results to return, best first. `0` (vLLM's default) means all of them.
        return_documents: Include each passage's text in its result. `False` saves bytes, not work.
    """

    model: str | None = None
    query: str
    documents: list[str | RerankDocument]
    top_n: int = Field(default=0, ge=0)
    return_documents: bool = True


class RerankResultDocument(BaseModel):
    """A ranked passage, wrapped the way Cohere's response wraps it."""

    text: str


class RerankResult(BaseModel):
    """One ranked passage.

    Attributes:
        index: Position in the request's `documents`, so the client can recover input order and
            notice any index that is missing (an unscored document under `--on-error drop`).
        relevance_score: P(the document answers the query), in `[0, 1]`.
        document: The passage text, absent when `return_documents` is false.
    """

    index: int
    relevance_score: float
    document: RerankResultDocument | None = None


class RerankResponse(BaseModel):
    """`POST /rerank` response: `results` sorted by relevance, descending.

    Attributes:
        id: Request id, for correlating with server logs.
        model: The id of the model that answered, read from the backend's reply.
        usage: Token counts summed over every decision request this rerank issued; `{}` when the
            backend reported none.
        results: The ranked documents, truncated to `top_n`.
    """

    id: str
    model: str
    usage: dict[str, int]
    results: list[RerankResult]


# ---------------------------------------------------------------------------------------------- provenance


class _RecordsResponses(BaseDecisionReranker):
    """Mixin keeping every `DecisionResponse` one `rerank` call produced.

    Overriding `ask` is the extension point the base documents for observing a request without
    touching parsing or ordering. The recorded responses are what let the HTTP layer report the
    answering model and the token usage -- neither of which reaches the ranking path by design.

    A reranker instance is therefore good for exactly one `rerank` call: it is constructed per
    request, sharing the process-wide backend, so concurrent requests cannot read each other's
    responses. The object is cheap; the client inside the backend is the one that is reused.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.responses: list[DecisionResponse] = []

    def ask(self, state: Mapping[str, object], questions: Mapping[str, Any]) -> DecisionResponse:
        """Send the request through the base implementation and record the reply.

        Args:
            state: Context shared by every question in the request.
            questions: Question id to question.

        Returns:
            The backend's `DecisionResponse`, also appended to `responses`.
        """
        response = super().ask(state, questions)
        self.responses.append(response)
        return response


class ServingNoulReranker(_RecordsResponses, DecisionNoulReranker):
    """`DecisionNoulReranker` that records its responses, for the HTTP layer's provenance."""


class ServingChoiceReranker(_RecordsResponses, DecisionChoiceReranker):
    """`DecisionChoiceReranker` that records its responses, for the HTTP layer's provenance."""


# ---------------------------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class ServerConfig:
    """Everything the process decides at startup, resolved from arguments and environment.

    Attributes:
        primitive: Which reranking strategy to serve; see `make_reranker`.
        backend: Which adapter speaks to the serving instance.
        endpoint: Base url of the serving instance.
        model: Model id sent with every request; also the fallback for the `model` field of a
            response, before a probe replaces it with what actually answered.
        api_key: Credential for the endpoint; `None` defers to the SDK's own environment lookup.
        timeout: Seconds allowed per backend call, enforced by the client, not by this file.
        instructions: The question text asked of the candidates.
        noul_true: What makes a `noul` answer yes; `None` omits the criterion.
        noul_false: What makes a `noul` answer no; `None` omits the criterion.
        alpha: Strength of the `choice` label-prior correction; `0` skips the probe request.
        k: Number of option orderings the `choice` ensemble averages, asked as `k` questions in one
            request. Larger `k` averages position bias harder without costing extra round trips.
        seed: Seed for those orderings; `None` shuffles differently per request.
        eps: Floor in the label-prior division.
        on_error: `"raise"` answers a transport failure with HTTP 502; `"drop"` omits the affected
            documents from `results` and ranks the rest.
        max_documents: Reject a shortlist longer than this with HTTP 400. `0` means unbounded.
        max_candidate_chars: Truncate every document to this many characters before asking. `0`
            means no truncation; note that truncation changes the score, it does not only shorten it.
        host: Bind address.
        port: Bind port; `8000` is vLLM's default, so the swap is a URL change.
        max_concurrency: Decision requests in flight at once, across all clients.
        probe: Ask one throwaway question at startup to confirm the endpoint and learn the real
            model id. `False` starts without contacting the backend.
        log_level: Root log level.
    """

    primitive: Primitive
    backend: BackendKind
    endpoint: str
    model: str
    api_key: str | None
    timeout: float
    instructions: str
    noul_true: str | None
    noul_false: str | None
    alpha: float
    k: int
    seed: int | None
    eps: float
    on_error: OnError
    max_documents: int
    max_candidate_chars: int
    host: str
    port: int
    max_concurrency: int
    probe: bool
    log_level: str


def make_reranker(config: ServerConfig, backend: BaseDecisionBackend) -> _RecordsResponses:
    """Build a one-shot reranker for a single request against the shared backend.

    Called per request, not once at startup: the recorded responses must belong to one ranking, and
    constructing a reranker is a few attribute assignments -- the expensive object (the client)
    stays in `backend`.

    Args:
        config: Startup configuration; only the ranking fields are read here.
        backend: The process-wide adapter to call.

    Returns:
        A `ServingNoulReranker` or `ServingChoiceReranker`, with no responses recorded yet.
    """
    if config.primitive == "noul":
        return ServingNoulReranker(
            backend,
            instructions=config.instructions,
            noul_true=config.noul_true,
            noul_false=config.noul_false,
            on_error=config.on_error,
        )
    return ServingChoiceReranker(
        backend,
        instructions=config.instructions,
        alpha=config.alpha,
        k=config.k,
        seed=config.seed,
        eps=config.eps,
        on_error=config.on_error,
    )


def build_backend(config: ServerConfig) -> BaseDecisionBackend:
    """Construct the adapter for the configured serving instance.

    The SDK import lives inside its branch so a missing optional extra fails with the name of the
    extra to install, at startup, instead of breaking this module at import time.

    Args:
        config: Startup configuration; `backend`, `endpoint`, `model`, `api_key`, `timeout`.

    Returns:
        A `TypeSafeBackend` or `OpenAIBackend` holding a client configured for `config.endpoint`.

    Raises:
        RuntimeError: The selected backend's package is not installed.
    """
    if config.backend == "typesafe":
        try:
            from typesafe_sdk import TypeSafeClient

            from decision_as_reranker.backend.typesafe import TypeSafeBackend
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("the typesafe backend needs the `typesafe` extra: uv sync --extra typesafe") from exc
        # `base_url` stops before `/v1`; the SDK appends `/v1/systemone`. An empty key would raise
        # at construction, so `None` is the only way to say "fall back to TYPESAFE_API_KEY".
        client = TypeSafeClient(
            api_key=config.api_key,
            base_url=config.endpoint,
            model=config.model,
            timeout=config.timeout,
        )
        return TypeSafeBackend(client, model=config.model)

    try:
        from openai import OpenAI

        from decision_as_reranker.backend.openai import OpenAIBackend
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("the openai backend needs the `openai` extra: uv sync --extra openai") from exc
    # The OpenAI SDK appends only `/decisions`, so `base_url` must already end in `/v1`.
    client = OpenAI(api_key=config.api_key, base_url=config.endpoint, timeout=config.timeout)
    return OpenAIBackend(client, model=config.model)


def probe_backend(backend: BaseDecisionBackend) -> str:
    """Ask one throwaway question to confirm the endpoint answers, and report the model that did.

    Costs a single call, at boot. Worth it for the same reason vLLM loading weights is worth it:
    a server that starts while its backend is unreachable would otherwise first fail on a real
    user's rerank, and the `model` field of every response needs the resolved id rather than the
    alias typed on the command line.

    Args:
        backend: The adapter to call.

    Returns:
        The model id from the reply.

    Raises:
        BaseException: Whatever the backend's `transport_errors` name, or any wiring error, when the
            endpoint does not answer.
    """
    response = backend.ask(
        {"query": "This is a startup connectivity check."},
        {"probe": {"type": "noul", "instructions": {"question": "Is this a valid request?", "candidate": "Yes."}}},
    )
    return response.model


def document_texts(documents: Sequence[str | RerankDocument]) -> list[str]:
    """Flatten the request's documents into plain passages, in input order.

    Args:
        documents: Strings or `{"text": ...}` objects, as sent by Cohere/Jina/vLLM clients.

    Returns:
        The passage texts, positionally aligned with the request.
    """
    return [document if isinstance(document, str) else document.text for document in documents]


def aggregate_usage(responses: Sequence[DecisionResponse]) -> dict[str, int]:
    """Sum the token counts of every decision request one rerank issued.

    `choice` with a label-prior probe issues two requests, so the numbers are per rerank, not per
    HTTP round trip. `{}` when no response carried usage -- an absent count is honest, a zero is not.

    Args:
        responses: The recorded responses, in call order.

    Returns:
        vLLM-shaped `{"prompt_tokens", "completion_tokens", "total_tokens"}`, or `{}`.
    """
    prompt = completion = total = 0
    reported = False
    for response in responses:
        usage = response.usage or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        if input_tokens is None and output_tokens is None and total_tokens is None:
            continue
        reported = True
        prompt += input_tokens or 0
        completion += output_tokens or 0
        total += total_tokens if total_tokens is not None else (input_tokens or 0) + (output_tokens or 0)
    if not reported:
        return {}
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


# ---------------------------------------------------------------------------------------------- application


def create_app(config: ServerConfig) -> FastAPI:
    """Build the ASGI app: routes, lifespan, and the process-wide backend.

    Args:
        config: Everything decided at startup.

    Returns:
        A `FastAPI` ready for `uvicorn.run`, with no backend connected until it is served.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        backend = build_backend(config)
        app.state.config = config
        app.state.backend = backend
        app.state.limiter = anyio.CapacityLimiter(config.max_concurrency)
        app.state.served_model = config.model
        if config.probe:
            try:
                app.state.served_model = await run_in_threadpool(probe_backend, backend)
            except backend.transport_errors as exc:
                backend.close()
                raise RuntimeError(f"backend at {config.endpoint} is not answering: {exc!r}") from exc
            logger.info("connected to %s (model=%s)", config.endpoint, app.state.served_model)
        else:
            logger.warning("skipping the startup probe; reporting model=%s unverified", config.model)
        try:
            yield
        finally:
            backend.close()

    app = FastAPI(
        title="decision-as-reranker",
        summary="A typed decision model behind vLLM's reranking API.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(HTTPException)
    async def openai_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Rewrite FastAPI's `{"detail": ...}` into the `{"error": {...}}` vLLM clients parse."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": str(exc.detail), "type": exc.__class__.__name__, "code": exc.status_code}},
        )

    @app.exception_handler(RequestValidationError)
    async def openai_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Same envelope for a malformed body, so one client-side parser covers both failures."""
        message = "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())
        return JSONResponse(
            status_code=422, content={"error": {"message": message, "type": "BadRequestError", "code": 422}}
        )

    @app.post("/rerank", response_model=RerankResponse)
    @app.post("/v1/rerank", response_model=RerankResponse)
    @app.post("/v2/rerank", response_model=RerankResponse)
    async def rerank(body: RerankRequest, request: Request) -> RerankResponse:
        """Rank `documents` for `query` with the decision model, vLLM's rerank contract.

        Args:
            body: The rerank request.
            request: The ASGI request, used only to reach the app state.

        Returns:
            The ranked documents, best first, truncated to `top_n`.

        Raises:
            HTTPException: 400 when the shortlist exceeds `max_documents`; 502 when the decision
                model was unreachable, refused, answered incompletely, or returned an off-contract
                probability; 500 for anything else.
        """
        state = request.app.state
        config: ServerConfig = state.config
        backend: BaseDecisionBackend = state.backend
        documents = document_texts(body.documents)
        if config.max_candidate_chars:
            documents = [text[: config.max_candidate_chars] for text in documents]
        if config.max_documents and len(documents) > config.max_documents:
            raise HTTPException(
                400,
                f"{len(documents)} documents exceeds this server's --max-documents={config.max_documents}; "
                "retrieve a shorter shortlist or raise the bound",
            )

        request_id = f"rerank-{uuid.uuid4().hex}"
        if not documents:
            return RerankResponse(id=request_id, model=state.served_model, usage={}, results=[])

        reranker = make_reranker(config, backend)
        started = time.perf_counter()
        async with state.limiter:
            try:
                ranked = await run_in_threadpool(reranker.rerank, body.query, documents)
            except backend.transport_errors as exc:
                raise HTTPException(502, f"decision model unavailable: {exc}") from exc
            except (KeyError, ValueError) as exc:
                # Both are the backend's fault on this path: `KeyError` means a question came back
                # unanswered, `ValueError` means an answer landed outside [0, 1] or named an option
                # that was never offered. The wiring-side ValueErrors cannot fire here -- every
                # document index is targeted, and an empty shortlist returns above without a call.
                raise HTTPException(502, f"decision model answered off-contract: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000

        # Measured before the `top_n` slice: documents a client did not ask to see back are not
        # documents the model failed to score, and conflating them makes every truncated response
        # look partially failed.
        missing = sorted({index for index in range(len(documents))} - {scored.index for scored in ranked})
        if missing:
            logger.warning(
                "%s: %d document(s) unscored and omitted from results: %s", request_id, len(missing), missing
            )
        if not ranked:
            # A non-empty shortlist cannot legitimately rank to nothing: `rank` drops only unscored
            # documents, and a score of exactly 0.0 stays. Under `--on-error drop` one explanation
            # remains -- the backend answered none of them -- which is a 502, not an empty 200.
            raise HTTPException(502, f"no document could be scored by {config.endpoint}")
        if body.top_n:
            ranked = ranked[: body.top_n]

        model = reranker.responses[-1].model if reranker.responses else state.served_model
        logger.info(
            "%s primitive=%s K=%d model=%s %.0fms top=%.4f%s",
            request_id,
            config.primitive,
            len(documents),
            model,
            elapsed_ms,
            ranked[0].score,
            "" if not missing else f" unscored={len(missing)}",
        )
        return RerankResponse(
            id=request_id,
            model=model,
            usage=aggregate_usage(reranker.responses),
            results=[
                RerankResult(
                    index=scored.index,
                    relevance_score=scored.score,
                    document=RerankResultDocument(text=scored.candidate) if body.return_documents else None,
                )
                for scored in ranked
            ],
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness: the process is up. Deliberately does not touch the backend -- a readiness
        probe that costs a decision call per poll is a self-inflicted outage."""
        return {"status": "ok"}

    @app.get("/version")
    async def version(request: Request) -> dict[str, object]:
        """Server identity: this is not vLLM, and pretending otherwise hides the difference."""
        config: ServerConfig = request.app.state.config
        return {
            "server": "decision-as-reranker",
            "version": "0.1.0",
            "primitive": config.primitive,
            "backend": config.backend,
            "model": request.app.state.served_model,
        }

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, object]:
        """The OpenAI model-listing shape, with the one model this process serves."""
        served: str = request.app.state.served_model
        return {
            "object": "list",
            "data": [
                {
                    "id": served,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "decision-as-reranker",
                    "root": served,
                    "parent": None,
                }
            ],
        }

    return app


# ---------------------------------------------------------------------------------------------- entry point


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Argument list; `None` reads `sys.argv`.

    Returns:
        The parsed namespace, before environment defaults are resolved.
    """
    parser = argparse.ArgumentParser(
        description="Serve a typed decision model behind vLLM's reranking API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    serving = parser.add_argument_group("serving")
    serving.add_argument("--host", default="127.0.0.1", help="bind address")
    serving.add_argument("--port", type=int, default=8000, help="bind port; 8000 matches `vllm serve`")
    serving.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    serving.add_argument(
        "--max-concurrency", type=int, default=8, help="decision requests in flight at once, across all clients"
    )
    serving.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="reject a longer shortlist with HTTP 400; 0 is unbounded. Context, not call count, is the limit",
    )
    serving.add_argument(
        "--max-candidate-chars",
        type=int,
        default=0,
        help="truncate every document to this many characters before asking; 0 disables truncation",
    )
    serving.add_argument(
        "--on-error",
        default="raise",
        choices=["raise", "drop"],
        help="'raise' answers a failure with 502; 'drop' omits the unscored documents",
    )
    serving.add_argument(
        "--skip-probe",
        action="store_true",
        help="start without contacting the backend; report the configured model id unverified",
    )

    wiring = parser.add_argument_group("backend")
    wiring.add_argument(
        "--backend",
        default="typesafe",
        choices=["typesafe", "openai"],
        help="which protocol the serving instance speaks",
    )
    wiring.add_argument(
        "--endpoint",
        default=None,
        help="base url. Typesafe stops before /v1 (the SDK appends /v1/systemone); OpenAI must include /v1. "
        f"Defaults: {TYPESAFE_DEFAULT_URL} / {OPENAI_DEFAULT_URL}, or $DECISION_BASE_URL",
    )
    wiring.add_argument(
        "--model", default=None, help=f"$DECISION_MODEL, else {TYPESAFE_DEFAULT_MODEL} / {OPENAI_DEFAULT_MODEL}"
    )
    wiring.add_argument(
        "--api-key",
        default=None,
        help="$DECISION_API_KEY, else the SDK's own lookup (TYPESAFE_API_KEY / OPENAI_API_KEY)",
    )
    wiring.add_argument("--timeout", type=float, default=60.0, help="seconds per backend call")

    ranking = parser.add_argument_group("ranking")
    ranking.add_argument(
        "--primitive",
        default="noul",
        choices=["noul", "choice"],
        help="independent per-candidate, or one joint comparison",
    )
    ranking.add_argument(
        "--instructions", default=None, help="the question asked of the candidates; defaults per primitive"
    )
    ranking.add_argument("--noul-true", default=None, help="what makes a noul answer yes (noul only)")
    ranking.add_argument("--noul-false", default=None, help="what makes a noul answer no (noul only)")
    ranking.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="label-prior correction strength (choice only); 0 skips the probe request",
    )
    ranking.add_argument(
        "--k",
        type=int,
        default=1,
        help="option orderings to average (choice only), asked as k questions in one request; "
        "1 asks the shortlist as given",
    )
    ranking.add_argument(
        "--seed", type=int, default=None, help="seed for those orderings (choice only); set it to reproduce a ranking"
    )
    ranking.add_argument("--eps", type=float, default=1e-7, help="floor in the label-prior division (choice only)")

    args = parser.parse_args(argv)
    noul_only = {"--noul-true": args.noul_true, "--noul-false": args.noul_false}
    choice_only = {
        "--alpha": args.alpha,
        "--k": args.k,
        "--seed": args.seed,
        "--eps": args.eps,
    }
    defaults = {name: parser.get_default(name.lstrip("-").replace("-", "_")) for name in {**noul_only, **choice_only}}
    set_elsewhere = [name for name, value in (({**noul_only, **choice_only}).items()) if value != defaults[name]]
    applies_to = {name: ("noul" if name in noul_only else "choice") for name in set_elsewhere}
    wrong = sorted({name for name, kind in applies_to.items() if kind != args.primitive})
    if wrong:
        parser.error(
            f"{', '.join(wrong)} apply to --primitive {' or '.join(sorted(applies_to[name] for name in wrong))}, not {args.primitive}"
        )
    return args


def resolve_config(args: argparse.Namespace) -> ServerConfig:
    """Merge arguments, environment defaults, and per-backend constants into one config.

    Explicit arguments beat environment variables beat backend defaults, so pointing the same
    command at hosted Jev or a local Kev changes only the environment.

    Args:
        args: The parsed namespace from `parse_args`.

    Returns:
        The frozen configuration the app is built from.
    """
    typesafe = args.backend == "typesafe"
    endpoint = (
        args.endpoint
        or os.environ.get("DECISION_BASE_URL")
        or (TYPESAFE_DEFAULT_URL if typesafe else OPENAI_DEFAULT_URL)
    )
    model = (
        args.model or os.environ.get("DECISION_MODEL") or (TYPESAFE_DEFAULT_MODEL if typesafe else OPENAI_DEFAULT_MODEL)
    )
    api_key = args.api_key or os.environ.get("DECISION_API_KEY")
    if api_key is None and typesafe and ("localhost" in endpoint or "127.0.0.1" in endpoint):
        api_key = "local"  # a self-hosted Kev checks no key, but the SDK requires a non-empty one

    default_instructions = DEFAULT_NOUL_INSTRUCTIONS if args.primitive == "noul" else DEFAULT_CHOICE_INSTRUCTIONS
    return ServerConfig(
        primitive=args.primitive,
        backend=args.backend,
        endpoint=endpoint.rstrip("/"),
        model=model,
        api_key=api_key,
        timeout=args.timeout,
        instructions=args.instructions or default_instructions,
        noul_true=args.noul_true if args.noul_true is not None else DEFAULT_NOUL_TRUE,
        noul_false=args.noul_false if args.noul_false is not None else DEFAULT_NOUL_FALSE,
        alpha=args.alpha,
        k=args.k,
        seed=args.seed,
        eps=args.eps,
        on_error=args.on_error,
        max_documents=args.max_documents,
        max_candidate_chars=args.max_candidate_chars,
        host=args.host,
        port=args.port,
        max_concurrency=args.max_concurrency,
        probe=not args.skip_probe,
        log_level=args.log_level,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments, build the app, and serve it until interrupted.

    Args:
        argv: Argument list; `None` reads `sys.argv`.
    """
    import uvicorn

    config = resolve_config(parse_args(argv))
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    logger.info(
        "serving %s reranker on %s:%d -> %s (model=%s, on_error=%s)",
        config.primitive,
        config.host,
        config.port,
        config.endpoint,
        config.model,
        config.on_error,
    )
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level=config.log_level)


if __name__ == "__main__":
    main()
