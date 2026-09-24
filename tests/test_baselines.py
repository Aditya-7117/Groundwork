"""The cheaper measures, on cases worked out by hand."""

import numpy as np
import pytest

from groundwork.baselines import (
    contains_reference,
    lexical_support,
    normalise_answer,
    strongest_entailment,
)

QUESTION = "longest river in ireland"
PASSAGES = ["River Shannon\nThe Shannon is the longest river in Ireland. It reaches Limerick."]


class TestContainment:
    def test_normalisation_follows_the_squad_rule(self) -> None:
        assert normalise_answer("  The River   Shannon! ") == "river shannon"

    def test_an_answer_containing_a_reference_is_correct(self) -> None:
        assert contains_reference("It is the River Shannon.", ["River Shannon", "Shannon"])

    def test_references_match_whole_words_only(self) -> None:
        assert not contains_reference("It passes Shannonbridge.", ["Shannon"])

    def test_a_reference_that_normalises_to_nothing_matches_nothing(self) -> None:
        assert not contains_reference("The end.", ["The"])


class TestLexicalSupport:
    def test_every_new_word_in_the_passages_scores_one(self) -> None:
        answer = "The Shannon is the longest river in Ireland."
        assert lexical_support(QUESTION, PASSAGES, answer) == 1.0

    def test_an_invented_name_scores_zero(self) -> None:
        # "Liffey" is the only word the answer adds beyond the question, and no passage has it.
        answer = "The Liffey is the longest river in Ireland."
        assert lexical_support(QUESTION, PASSAGES, answer) == 0.0

    def test_partial_support_is_the_share_of_new_words_found(self) -> None:
        # New words: shannon, flow, past, limerick. The passage has shannon and limerick.
        answer = "The Shannon, which flows past Limerick."
        assert lexical_support(QUESTION, PASSAGES, answer) == 0.5

    def test_restating_the_question_claims_nothing(self) -> None:
        assert lexical_support(QUESTION, PASSAGES, "The longest river in Ireland.") == 1.0


class TestStrongestEntailment:
    def test_the_best_passage_counts(self) -> None:
        probabilities = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.05, 0.05, 0.9]])
        assert strongest_entailment(probabilities) == pytest.approx(0.7)

    def test_no_passages_means_no_entailment(self) -> None:
        assert strongest_entailment(np.zeros((0, 3))) == 0.0
