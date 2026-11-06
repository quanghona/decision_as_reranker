"""Tests for `BaseDecisionReranker` (the backend call, answer parsing, and ordering).

Offline by design: the base is exercised against a fake backend adapter that returns a real
`DecisionResponse`, so the parse path is tested without a server. No test asserts an exact
probability from a live model -- only ordering, bounds, and error propagation.
"""

from __future__ import annotations

import pytest
from typesafe_sdk import Choice, Noul, Score, SystemOneResponse
from typesafe_sdk import ChoiceAnswer as SDKChoiceAnswer
from typesafe_sdk import NoulAnswer as SDKNoulAnswer
from typesafe_sdk import ScoreAnswer as SDKScoreAnswer

from decision_as_reranker.backend.base import (
    BaseDecisionBackend,
    ChoiceAnswer,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
)
from decision_as_reranker.reranker import (
    BaseDecisionReranker,
    DecisionChoiceReranker,
    DecisionNoulReranker,
    DecisionQuery,
    ScoredCandidate,
)

CANDIDATES = ["first passage", "second passage", "third passage"]

# The question text is constructor-required now -- there are no package-level defaults to drift --
# so every `DecisionNoulReranker` here states its own boundary.
INSTRUCTIONS = "Could this passage be what the query asks for?"
NOUL_TRUE = "It states the fact the query asks about."
NOUL_FALSE = "It only shares the query's topic."


def to_backend_answer(sdk_answer):
    """SDK answer -> neutral answer: the conversion `TypeSafeBackend` will perform, done here so
    the reranker is tested against its actual contract instead of the SDK's objects."""
    if isinstance(sdk_answer, SDKNoulAnswer):
        return NoulAnswer(noul=sdk_answer.noul)
    if isinstance(sdk_answer, SDKChoiceAnswer):
        return ChoiceAnswer(choice=sdk_answer.choice, probabilities=sdk_answer.probabilities)
    if isinstance(sdk_answer, SDKScoreAnswer):
        return ScoreAnswer(score=sdk_answer.score, probabilities=sdk_answer.probabilities, legend=sdk_answer.legend)
    raise AssertionError(f"unexpected SDK answer {sdk_answer!r}")


class FakeBackend(BaseDecisionBackend):
    """Stands in for a concrete adapter: records the request, returns whatever the test needs.

    Dict-form answers go through the SDK's own validation (shape-checked), then the same
    SDK -> neutral conversion a TypeSafe-backed backend performs, so `_answer_value` sees only
    `backend`'s typed objects. `ConnectionError` plays the transport-failure role -- proof that
    the reranker classifies failures by the backend's `transport_errors`, not by hardcoded SDK
    exception classes.
    """

    transport_errors = (ConnectionError,)

    def __init__(self, answers=None, error=None, model=None):
        super().__init__(model=model)
        self.answers = answers or {}
        self.error = error
        self.calls: list[dict] = []

    def _ask(self, state, questions):
        self.calls.append({"state": state, "questions": questions, "model": self.model})
        if self.error is not None:
            raise self.error
        parsed = SystemOneResponse.model_validate(
            {"model": "kev-latest", "usage": {"input_tokens": 10, "output_tokens": 3}, "answers": self.answers}
        )
        answers = {qid: to_backend_answer(a) for qid, a in parsed.answers.items()}
        return DecisionResponse(answers=answers, model=parsed.model, usage=parsed.usage.model_dump())


class StubReranker(BaseDecisionReranker):
    """Minimal concrete subclass: the base does all the work, the subclass only picks questions."""

    def __init__(self, backend, decisions, **kwargs):
        super().__init__(backend, **kwargs)
        self.decisions = decisions

    def rerank(self, query, candidates):
        return self.rank(candidates, self.score(query, candidates, self.decisions))


def noul_decisions(count):
    return [DecisionQuery(Noul(instructions=f"q{i}"), {i: None}) for i in range(count)]


def noul_answers(values):
    return {f"q{i}": {"type": "noul", "noul": value} for i, value in enumerate(values)}


# --------------------------------------------------------------------------- construction


def test_base_class_cannot_be_instantiated():
    # Regression: without abc.ABC in the bases, @abstractmethod was inert and this constructed fine.
    # A backend is supplied so the only thing left to fail on is the abstractness itself.
    with pytest.raises(TypeError):
        BaseDecisionReranker(FakeBackend())  # type: ignore[abstract]


def test_on_error_must_be_a_known_policy():
    with pytest.raises(ValueError, match="on_error"):
        StubReranker(FakeBackend(), [], on_error="ignore")


# --------------------------------------------------------------------------- noul parsing


def test_score_reads_one_noul_per_candidate_in_input_order():
    backend = FakeBackend(answers=noul_answers([0.2, 0.9, 0.5]))
    reranker = StubReranker(backend, noul_decisions(3))

    assert reranker.score("query", CANDIDATES, noul_decisions(3)) == [0.2, 0.9, 0.5]


def test_all_candidates_are_answered_in_one_request():
    backend = FakeBackend(answers=noul_answers([0.2, 0.9, 0.5]))
    reranker = StubReranker(backend, noul_decisions(3))
    reranker.score("query", CANDIDATES, noul_decisions(3))

    assert len(backend.calls) == 1, "K questions belong in one call, not K calls"
    assert sorted(backend.calls[0]["questions"]) == ["q0", "q1", "q2"]
    assert backend.calls[0]["state"] == {"query": "query"}


def test_empty_shortlist_never_reaches_the_backend():
    backend = FakeBackend()
    assert StubReranker(backend, []).score("query", [], []) == []
    assert backend.calls == []


# --------------------------------------------------------------------------- choice / score


def test_choice_answer_is_read_through_its_option_probability():
    answers = {
        "q0": {"type": "choice", "choice": "d1", "confidence": 0.21, "probabilities": {"d0": 0.2, "d1": 0.7, "d2": 0.1}}
    }
    decisions = [DecisionQuery(Choice(criteria={"d0": "a", "d1": "b", "d2": "c"}), {0: "d0", 1: "d1", 2: "d2"})]
    scores = StubReranker(FakeBackend(answers=answers), decisions).score("query", CANDIDATES, decisions)

    assert scores == [0.2, 0.7, 0.1], "ranking reads probabilities, never confidence"


def test_unknown_choice_option_names_the_options_the_backend_returned():
    answers = {"q0": {"type": "choice", "choice": "d0", "confidence": 1.0, "probabilities": {"d0": 1.0}}}
    decisions = [DecisionQuery(Choice(criteria={"d0": "a"}), {0: "d9"})]

    with pytest.raises(ValueError, match="asked for 'd9'"):
        StubReranker(FakeBackend(answers=answers), decisions).score("query", ["a"], decisions)


def test_score_answer_is_normalised_onto_the_unit_interval():
    # 3 levels -> mean index 1.44 of a maximum 2 -> 0.72.
    answers = {
        "q0": {
            "type": "score",
            "score": 1.44,
            "confidence": 0.34,
            "legend": {0: "low", 1: "mid", 2: "high"},
            "probabilities": {0: 0.0, 1: 0.56, 2: 0.44},
        }
    }
    decisions = [DecisionQuery(Score(criteria=["low", "mid", "high"]), {0: None})]

    assert StubReranker(FakeBackend(answers=answers), decisions).score("q", ["x"], decisions) == [pytest.approx(0.72)]


def test_several_signals_on_one_candidate_are_averaged():
    answers = {**noul_answers([0.8, 0.4]), "q2": {"type": "noul", "noul": 0.2}}
    decisions = noul_decisions(2) + [DecisionQuery(Noul(instructions="extra"), {0: None})]

    scores = StubReranker(FakeBackend(answers=answers), decisions).score("query", CANDIDATES[:2], decisions)
    assert scores == [pytest.approx(0.5), 0.4]


# --------------------------------------------------------------------------- ordering


def test_rerank_orders_by_score_then_input_position():
    backend = FakeBackend(answers=noul_answers([0.5, 0.5, 0.9]))
    ranked = StubReranker(backend, noul_decisions(3)).rerank("query", CANDIDATES)

    assert [entry.candidate for entry in ranked] == [CANDIDATES[2], CANDIDATES[0], CANDIDATES[1]]
    assert [entry.index for entry in ranked] == [2, 0, 1], "ties must break on position, not iteration order"
    assert all(isinstance(entry, ScoredCandidate) for entry in ranked)


def test_rerank_keeps_duplicate_texts_as_separate_entries():
    duplicated = ["same", "same"]
    backend = FakeBackend(answers=noul_answers([0.4, 0.4]))

    ranked = StubReranker(backend, noul_decisions(2)).rerank("query", duplicated)
    assert [entry.index for entry in ranked] == [0, 1]


def test_rank_returns_the_whole_shortlist_and_top_k_is_a_slice():
    # `rank` orders; it does not truncate. How many to show is caller policy, not an ordering
    # parameter -- the `k` argument that used to sit here could only ever mirror `[:k]`.
    backend = FakeBackend(answers=noul_answers([0.1, 0.9, 0.5]))
    ranked = StubReranker(backend, noul_decisions(3)).rerank("query", CANDIDATES)

    assert [entry.score for entry in ranked] == [0.9, 0.5, 0.1]
    assert [entry.score for entry in ranked[:2]] == [0.9, 0.5]


def test_rank_requires_one_score_per_candidate():
    with pytest.raises(ValueError, match="2 scores for 3 candidates"):
        StubReranker(FakeBackend(), []).rank(CANDIDATES, [0.1, 0.2])


# --------------------------------------------------------------------------- failure handling


def test_transport_failure_propagates_and_is_never_a_zero_score():
    reranker = StubReranker(FakeBackend(error=ConnectionError("down")), noul_decisions(3))

    with pytest.raises(ConnectionError):
        reranker.score("query", CANDIDATES, noul_decisions(3))


def test_non_transport_failure_is_never_dropped_even_under_the_drop_policy():
    # `transport_errors` is the only classification `on_error` may act on; a bug inside the
    # backend must not degrade into unscored candidates.
    reranker = StubReranker(FakeBackend(error=RuntimeError("bug")), noul_decisions(3), on_error="drop")

    with pytest.raises(RuntimeError):
        reranker.score("query", CANDIDATES, noul_decisions(3))


def test_drop_policy_leaves_unscored_candidates_out_of_the_ordering():
    reranker = StubReranker(FakeBackend(error=ConnectionError("down")), noul_decisions(3), on_error="drop")

    scores = reranker.score("query", CANDIDATES, noul_decisions(3))
    assert scores == [None, None, None]
    assert reranker.rank(CANDIDATES, scores) == [], "an unreachable model is not evidence of irrelevance"


def test_answer_missing_from_the_response_is_not_silently_ignored():
    decisions = noul_decisions(3)
    reranker = StubReranker(FakeBackend(answers={"q0": {"type": "noul", "noul": 0.5}}), decisions)

    with pytest.raises(KeyError, match="q1"):
        reranker.score("query", CANDIDATES, decisions)


def test_drop_policy_scores_what_the_backend_answered_and_nothing_else():
    decisions = noul_decisions(3)
    answers = {"q0": {"type": "noul", "noul": 0.5}, "q2": {"type": "noul", "noul": 0.1}}
    reranker = StubReranker(FakeBackend(answers=answers), decisions, on_error="drop")

    scores = reranker.score("query", CANDIDATES, decisions)
    assert scores == [0.5, None, 0.1]
    assert [entry.index for entry in reranker.rank(CANDIDATES, scores)] == [0, 2]


def test_candidate_targeted_by_no_question_is_a_configuration_error():
    decisions = [DecisionQuery(Noul(instructions="only one"), {1: None})]

    with pytest.raises(ValueError, match="not targeted"):
        StubReranker(FakeBackend(answers={"q0": {"type": "noul", "noul": 0.5}}), decisions).score(
            "query", CANDIDATES, decisions
        )


@pytest.mark.parametrize("on_error", ["raise", "drop"])
def test_untargeted_candidate_raises_whatever_the_error_policy_is(on_error):
    # `drop` covers unreachable backends; it must not hide a mis-wired targets map.
    decisions = [DecisionQuery(Noul(instructions="only one"), {1: None})]
    reranker = StubReranker(FakeBackend(answers={"q0": {"type": "noul", "noul": 0.5}}), decisions, on_error=on_error)

    with pytest.raises(ValueError, match="not targeted"):
        reranker.score("query", CANDIDATES, decisions)


def test_off_contract_probability_is_reported_as_such():
    decisions = noul_decisions(1)

    with pytest.raises(ValueError, match="off-contract"):
        StubReranker(FakeBackend(answers={"q0": {"type": "noul", "noul": 1.7}}), decisions).score("q", ["x"], decisions)


# --------------------------------------------------------------------------- backend seam


def test_the_answering_model_is_the_backend_configuration():
    # The reranker forwards no model: whatever the backend was constructed with is what answers.
    backend = FakeBackend(answers=noul_answers([0.5]), model="kev-latest")
    StubReranker(backend, []).score("q", ["x"], noul_decisions(1))

    assert backend.calls[0]["model"] == "kev-latest"


def test_the_injected_backend_is_exposed_and_the_reranker_owns_no_lifecycle():
    backend = FakeBackend()
    reranker = StubReranker(backend, [])

    assert reranker.backend is backend
    assert not hasattr(reranker, "close"), "closing belongs to whoever configured the endpoint"


# --------------------------------------------------------------------------- noul reranker


def noul_backend(values):
    """A backend answering `q0..qN` with the given nouls -- the ids `score` generates, in candidate order."""
    return FakeBackend(answers=noul_answers(values))


def noul_reranker(backend, **kwargs):
    """A `DecisionNoulReranker` with the required question text filled in; kwargs override."""
    settings = {"instructions": INSTRUCTIONS, "noul_true": NOUL_TRUE, "noul_false": NOUL_FALSE}
    settings.update(kwargs)
    return DecisionNoulReranker(backend, **settings)


def test_noul_reranker_orders_by_noul_and_the_caller_slices_the_top_k():
    reranker = noul_reranker(noul_backend([0.2, 0.9, 0.5]))

    ranked = reranker.rerank("what is the rule?", CANDIDATES)[:2]

    assert [(entry.candidate, entry.score) for entry in ranked] == [(CANDIDATES[1], 0.9), (CANDIDATES[2], 0.5)]


def test_noul_reranker_asks_one_question_per_candidate_in_a_single_request():
    backend = noul_backend([0.1, 0.2, 0.3])

    noul_reranker(backend).rerank("q", CANDIDATES)

    assert len(backend.calls) == 1, "K candidates are K questions in one call, not K calls"
    asked = backend.calls[0]["questions"]
    assert sorted(asked) == ["q0", "q1", "q2"]
    assert [asked[f"q{i}"]["instructions"]["candidate"] for i in range(3)] == CANDIDATES


def test_noul_reranker_shares_one_question_text_and_varies_only_the_candidate():
    backend = noul_backend([0.1, 0.2, 0.3])
    reranker = DecisionNoulReranker(
        backend, instructions="Is this the passage?", noul_true="It states the rule.", noul_false="It shares the topic."
    )

    reranker.rerank("q", CANDIDATES)

    asked = backend.calls[0]["questions"]
    assert {asked[f"q{i}"]["instructions"]["question"] for i in range(3)} == {"Is this the passage?"}
    assert (
        asked["q0"]["criteria"]
        == asked["q2"]["criteria"]
        == {"true": "It states the rule.", "false": "It shares the topic."}
    )


def test_noul_reranker_puts_the_query_in_state_and_never_in_a_question():
    # `state` is read once for every question; a query repeated per question would rescale the request.
    backend = noul_backend([0.1, 0.2, 0.3])

    noul_reranker(backend).rerank("the information need", CANDIDATES)

    assert backend.calls[0]["state"] == {"query": "the information need"}
    assert all("the information need" not in str(question) for question in backend.calls[0]["questions"].values())


def test_noul_reranker_emits_a_question_the_sdk_accepts():
    # The dict form is this module's only way to build a question; it must stay on the SDK's contract.
    question = noul_reranker(FakeBackend()).build_question("a passage")

    parsed = Noul.model_validate(question)
    assert parsed.instructions == {"question": INSTRUCTIONS, "candidate": "a passage"}
    assert parsed.criteria == {"true": NOUL_TRUE, "false": NOUL_FALSE}


def test_noul_reranker_sends_no_criteria_when_both_sides_are_omitted():
    backend = noul_backend([0.4])

    noul_reranker(backend, noul_true=None, noul_false=None).rerank("q", ["only"])

    assert "criteria" not in backend.calls[0]["questions"]["q0"]


def test_noul_reranker_never_reaches_the_backend_for_an_empty_shortlist():
    backend = FakeBackend()

    assert noul_reranker(backend).rerank("q", []) == []
    assert backend.calls == []


def test_noul_reranker_scores_duplicate_passages_separately():
    backend = noul_backend([0.4, 0.9])

    ranked = noul_reranker(backend).rerank("q", ["same", "same"])

    assert [entry.index for entry in ranked] == [1, 0]
    asked = backend.calls[0]["questions"]
    assert [question["instructions"]["candidate"] for question in asked.values()] == ["same", "same"]


def test_noul_reranker_propagates_a_transport_failure_instead_of_ranking_zeroes():
    reranker = noul_reranker(FakeBackend(error=ConnectionError("down")))

    with pytest.raises(ConnectionError):
        reranker.rerank("q", CANDIDATES)


def test_noul_reranker_drops_candidates_the_backend_left_unanswered_under_the_drop_policy():
    reranker = noul_reranker(FakeBackend(answers={"q1": {"type": "noul", "noul": 0.8}}), on_error="drop")

    ranked = reranker.rerank("q", CANDIDATES)

    assert [entry.index for entry in ranked] == [1], "a missing answer is not evidence of irrelevance"


def test_noul_reranker_reports_an_off_contract_probability_as_such():
    reranker = noul_reranker(noul_backend([0.5, 1.4, 0.1]))

    with pytest.raises(ValueError, match="off-contract"):
        reranker.rerank("q", CANDIDATES)


def test_noul_reranker_requires_a_backend_and_its_question_text():
    # Regression: the stub took no arguments at all, so a reranker with no transport constructed
    # fine. The question text is required too: with no package-level defaults, a reranker that
    # never states its boundary cannot exist.
    with pytest.raises(TypeError):
        DecisionNoulReranker()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DecisionNoulReranker(FakeBackend())  # type: ignore[call-arg]

    backend = FakeBackend()
    reranker = noul_reranker(backend)
    assert isinstance(reranker, BaseDecisionReranker)
    assert reranker.backend is backend


# --------------------------------------------------------------------------- untouched subclasses


def test_choice_stub_still_satisfies_the_abstract_interface():
    # Left alone this session: it constructs, and its `rerank` is unimplemented.
    assert DecisionChoiceReranker().rerank(CANDIDATES, [0.1, 0.2, 0.3]) is None
