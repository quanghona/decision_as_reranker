"""Tests for `BaseDecisionReranker` (the backend call, answer parsing, and ordering).

Offline by design: the base is exercised against a fake backend adapter that returns a real
`DecisionResponse`, so the parse path is tested without a server. No test asserts an exact
probability from a live model -- only ordering, bounds, and error propagation.
"""

from __future__ import annotations

import math

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


# --------------------------------------------------------------------------- choice reranker


class ChoiceBackend(FakeBackend):
    """A `choice` model with separable, switchable biases: P(option) ∝ name x position.

    `name_weights` is the label prior and `position_weights` the position bias; leaving either out
    switches that defect off. A probe question (descriptions all `None`) is answered by the same
    formula, which is the point of a probe: with the content removed, only the biases are left.
    `fail_on_probe` raises the transport error on probe requests only -- the prior is optional
    evidence, the candidate answers are not.
    """

    def __init__(self, name_weights=None, position_weights=None, fail_on_probe=False, **kwargs):
        super().__init__(**kwargs)
        self.name_weights = name_weights or {}
        self.position_weights = position_weights
        self.fail_on_probe = fail_on_probe

    @staticmethod
    def _is_probe(question):
        return all(description is None for description in question["criteria"].values())

    def _ask(self, state, questions):
        self.calls.append({"state": state, "questions": questions, "model": self.model})
        if self.error is not None:
            raise self.error
        if self.fail_on_probe and any(self._is_probe(q) for q in questions.values()):
            raise ConnectionError("prior probe unavailable")
        answers = {}
        for question_id, question in questions.items():
            names = list(question["criteria"])
            slots = self.position_weights or [1.0] * len(names)
            raw = {name: self.name_weights.get(name, 1.0) * slots[slot] for slot, name in enumerate(names)}
            total = sum(raw.values())
            probabilities = {name: value / total for name, value in raw.items()}
            best = max(probabilities, key=lambda name: probabilities[name])
            answers[question_id] = ChoiceAnswer(choice=best, probabilities=probabilities)
        return DecisionResponse(answers=answers, model="kev-latest")


def choice_reranker(backend, **kwargs):
    """A `DecisionChoiceReranker` with the required question text filled in; kwargs override."""
    settings = {"instructions": "Which of these passages answers the query?"}
    settings.update(kwargs)
    return DecisionChoiceReranker(backend, **settings)


LABEL_PRIOR = {"d0": 0.5, "d1": 0.3, "d2": 0.2}


def test_choice_reranker_requires_a_backend_and_its_question_text():
    # Regression: the stub took only `alpha`, so a reranker with no transport and no question
    # constructed fine -- and its `rerank` returned None silently instead of ranking.
    with pytest.raises(TypeError):
        DecisionChoiceReranker()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DecisionChoiceReranker(FakeBackend())  # type: ignore[call-arg]

    backend = FakeBackend()
    reranker = choice_reranker(backend)
    assert isinstance(reranker, BaseDecisionReranker)
    assert reranker.backend is backend


@pytest.mark.parametrize(
    "kwargs",
    [{"alpha": -0.5}, {"alpha": math.inf}, {"eps": 0.0}, {"eps": math.nan}, {"k": 0}, {"k": 2.5}],
)
def test_choice_reranker_rejects_off_contract_knobs(kwargs):
    with pytest.raises(ValueError):
        choice_reranker(FakeBackend(), **kwargs)


def test_choice_reranker_reads_one_joint_distribution_per_ordering():
    backend = ChoiceBackend(position_weights=[0.6, 0.3, 0.1])

    ranked = choice_reranker(backend, alpha=0).rerank("q", CANDIDATES)

    assert len(backend.calls) == 1, "one choice over all candidates, not one question per candidate"
    assert sorted(backend.calls[0]["questions"]) == ["q0"]
    assert [entry.index for entry in ranked] == [0, 1, 2]
    assert [entry.score for entry in ranked] == pytest.approx([0.6, 0.3, 0.1])


def test_choice_reranker_asks_every_ordering_in_one_request_with_names_pinned_to_candidates():
    backend = ChoiceBackend()

    choice_reranker(backend, k=3, alpha=0).rerank("q", CANDIDATES)

    assert len(backend.calls) == 1, "k orderings are k questions in one call, not k calls"
    asked = backend.calls[0]["questions"]
    assert sorted(asked) == ["q0", "q1", "q2"]
    for question in asked.values():
        assert sorted(question["criteria"]) == ["d0", "d1", "d2"]
        assert question["criteria"]["d0"] == CANDIDATES[0], "a shuffle moves positions, never labels"
    assert list(asked["q0"]["criteria"]) == ["d0", "d1", "d2"], "the first ordering is the input order"


def test_choice_reranker_averages_position_bias_over_the_k_permutations():
    positions = [0.7, 0.2, 0.1]
    backend = ChoiceBackend(position_weights=positions)
    reranker = choice_reranker(backend, k=6, alpha=0, seed=3)

    orders = reranker.build_permutations(len(CANDIDATES))
    expected = [sum(positions[order.index(i)] for order in orders) / len(orders) for i in range(len(CANDIDATES))]
    ranked = reranker.rerank("q", CANDIDATES)

    assert [entry.score for entry in ranked] == pytest.approx(sorted(expected, reverse=True))
    assert max(expected) < positions[0], "no candidate may keep the first slot's whole advantage"


def test_alpha_one_divides_the_label_prior_out_exactly():
    backend = ChoiceBackend(name_weights=LABEL_PRIOR)

    ranked = choice_reranker(backend, alpha=1.0).rerank("q", CANDIDATES)

    assert [entry.score for entry in ranked] == [pytest.approx(1 / 3)] * 3
    assert [entry.index for entry in ranked] == [0, 1, 2], "an even field ranks by input position"


def test_the_prior_is_divided_out_to_the_alpha_power():
    backend = ChoiceBackend(name_weights=LABEL_PRIOR)

    ranked = choice_reranker(backend, alpha=0.75).rerank("q", CANDIDATES)

    weights = [value**0.25 for value in LABEL_PRIOR.values()]
    assert [entry.score for entry in ranked] == pytest.approx([w / sum(weights) for w in weights])


def test_alpha_zero_skips_the_prior_probe():
    backend = ChoiceBackend(name_weights=LABEL_PRIOR)

    choice_reranker(backend, alpha=0).rerank("q", CANDIDATES)

    assert len(backend.calls) == 1, "no correction means no calibration request"


def test_the_prior_probe_reuses_the_orderings_without_the_candidate_text():
    backend = ChoiceBackend(name_weights=LABEL_PRIOR)

    choice_reranker(backend, k=3).rerank("q", CANDIDATES)

    content, probed = (call["questions"] for call in backend.calls)
    assert [list(q["criteria"]) for q in probed.values()] == [list(q["criteria"]) for q in content.values()]
    assert all(value is None for question in probed.values() for value in question["criteria"].values())
    assert backend.calls[1]["state"] == {"query": "q"}


def test_correct_label_prior_renormalises_the_divided_vector():
    reranker = choice_reranker(FakeBackend(), alpha=1.0)

    assert reranker.correct_label_prior([0.5, 0.3, 0.2], [0.5, 0.3, 0.2]) == [pytest.approx(1 / 3)] * 3
    assert reranker.correct_label_prior([0.2, 0.8], [0.5, 0.5]) == pytest.approx([0.2, 0.8])


def test_a_zero_prior_is_floored_not_divided_by():
    reranker = choice_reranker(FakeBackend(), alpha=1.0)

    assert reranker.correct_label_prior([0.0, 1.0], [0.0, 1.0]) == [0.0, pytest.approx(1.0)]


def test_eps_damps_the_boost_a_zero_prior_receives():
    # The whole-vector division is (q + eps) ** -alpha, so eps is the only lever on an option the
    # probe scored at exactly 0: the smaller the floor, the more the division inflates its share.
    tight = choice_reranker(FakeBackend(), alpha=1.0, eps=1e-7)
    loose = choice_reranker(FakeBackend(), alpha=1.0, eps=1e-2)

    assert tight.correct_label_prior([0.5, 0.5], [0.0, 1.0])[0] > loose.correct_label_prior([0.5, 0.5], [0.0, 1.0])[0]


def test_unscored_candidates_stay_out_of_the_renormalisation():
    reranker = choice_reranker(FakeBackend(), alpha=1.0)

    assert reranker.correct_label_prior([0.5, None, 0.5], [0.5, 0.5, 0.5]) == [0.5, None, 0.5]


def test_correct_label_prior_needs_one_prior_per_candidate():
    with pytest.raises(ValueError, match="1 priors for 2"):
        choice_reranker(FakeBackend()).correct_label_prior([0.5, 0.5], [1.0])


def test_build_permutations_returns_k_orderings_of_the_K_options():
    # K (the shortlist length) fixes how long each ordering is; k fixes how many there are. The
    # draw budget is k, not a retry count, so an ensemble as large as the option space is filled
    # exactly: 4! = 24 orderings of 4 options, all of them present once.
    orders = choice_reranker(FakeBackend(), k=24, seed=7).build_permutations(4)

    assert len(orders) == 24 == len(set(orders))
    assert orders[0] == (0, 1, 2, 3)
    assert all(sorted(order) == [0, 1, 2, 3] for order in orders)


def test_build_permutations_truncates_at_K_factorial_instead_of_repeating():
    # 2 options have only 2! = 2 distinct orderings; asking for k=5 draws 2, not 5 padded with
    # repeats -- a repeated ordering is the same question with the same answer, so the ensemble
    # shrinks rather than spending a slot for no information.
    orders = choice_reranker(FakeBackend(), k=5, seed=2).build_permutations(2)

    assert len(orders) == 2
    assert set(orders) == {(0, 1), (1, 0)}


def test_build_permutations_scales_with_k_not_with_K():
    # A 3-option shortlist and a 30-option one both get exactly k orderings: the ensemble size is
    # the tuned budget, decoupled from K the way AnyJev's K-rotation ensemble is not.
    drawn = [choice_reranker(FakeBackend(), k=4, seed=11).build_permutations(count) for count in (3, 30)]

    assert [len(orders) for orders in drawn] == [4, 4]
    assert all(len(set(orders)) == 4 for orders in drawn)


def test_choice_reranker_asks_k_questions_per_request():
    backend = ChoiceBackend()

    choice_reranker(backend, k=5, alpha=0).rerank("q", CANDIDATES)

    assert sorted(backend.calls[0]["questions"]) == [f"q{i}" for i in range(5)]


def test_build_permutations_starts_sorted_and_honours_the_budget():
    orders = choice_reranker(FakeBackend(), k=4, seed=7).build_permutations(3)

    assert len(orders) == 4 == len(set(orders))
    assert orders[0] == (0, 1, 2)
    assert all(sorted(order) == [0, 1, 2] for order in orders)


def test_build_permutations_is_reproducible_under_a_seed():
    drawn = [choice_reranker(FakeBackend(), k=5, seed=1).build_permutations(4) for _ in range(2)]

    assert drawn[0] == drawn[1], "a cached eval must redraw the same ensembles it scored"


def test_choice_reranker_never_reaches_the_backend_for_an_empty_shortlist():
    backend = ChoiceBackend()

    assert choice_reranker(backend).rerank("q", []) == []
    assert backend.calls == []


def test_choice_reranker_propagates_a_transport_failure_instead_of_ranking_zeroes():
    reranker = choice_reranker(ChoiceBackend(error=ConnectionError("down")))

    with pytest.raises(ConnectionError):
        reranker.rerank("q", CANDIDATES)


def test_choice_reranker_drops_everything_when_the_backend_is_unreachable():
    reranker = choice_reranker(ChoiceBackend(error=ConnectionError("down")), on_error="drop")

    assert reranker.rerank("q", CANDIDATES) == []


def test_a_failed_prior_probe_leaves_the_ordering_uncorrected_not_unscored():
    # The probe is auxiliary: losing it costs the debiasing, not the ranking. Under `raise` the
    # failure still aborts -- this is the `drop` policy's judgement, not a silent fallback.
    backend = ChoiceBackend(name_weights=LABEL_PRIOR, fail_on_probe=True)

    ranked = choice_reranker(backend, on_error="drop").rerank("q", CANDIDATES)

    assert [entry.score for entry in ranked] == [0.5, 0.3, 0.2]


def test_choice_reranker_emits_a_question_the_sdk_accepts():
    question = choice_reranker(FakeBackend()).build_question(CANDIDATES, (2, 0, 1))

    parsed = Choice.model_validate(question)
    assert parsed.instructions == "Which of these passages answers the query?"
    assert parsed.criteria == {f"d{i}": CANDIDATES[i] for i in range(3)}
    assert list(parsed.criteria) == ["d2", "d0", "d1"], "the ordering is the position signal"


def test_choice_reranker_scores_duplicate_passages_as_separate_options():
    backend = ChoiceBackend(position_weights=[0.5, 0.5])

    ranked = choice_reranker(backend, alpha=0).rerank("q", ["same", "same"])

    assert [entry.index for entry in ranked] == [0, 1]
    assert list(backend.calls[0]["questions"]["q0"]["criteria"].values()) == ["same", "same"]
