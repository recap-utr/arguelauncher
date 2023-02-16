from __future__ import absolute_import, annotations

import logging
import typing as t
from abc import ABC
from collections import defaultdict

from arg_services.cbr.v1beta import adaptation_pb2, retrieval_pb2

from arguelauncher.config import RetrievalEvaluationConfig
from arguelauncher.libs.ndcg import ndcg

logger = logging.getLogger(__name__)


class BaseEvaluation(ABC):
    query: str
    cases: set[str]
    config: RetrievalEvaluationConfig
    duration: float
    tp: set[str]
    fp: set[str]
    fn: set[str]
    tn: set[str]

    def __init__(
        self,
        cases: t.Iterable[str],
        query: str,
        duration: float,
        config: RetrievalEvaluationConfig,
    ):
        self.cases = set(cases)
        self.query = query
        self.config = config
        self.duration = duration

    def __dict__(self):
        out = {
            "precision": self.precision(),
            "recall": self.recall(),
            "accuracy": self.accuracy(),
            "balanced_accuracy": self.balanced_accuracy(),
            "error_rate": self.error_rate(),
            "sensitivity": self.sensitivity(),
            "specificity": self.specificity(),
        }

        for beta in self.config.f_scores:
            out[f"f{beta}"] = self.f_score(beta)

        return out

    def precision(self) -> t.Optional[float]:
        den = len(self.tp) + len(self.fp)

        return len(self.tp) / den if den > 0 else None

    def recall(self) -> t.Optional[float]:
        den = len(self.tp) + len(self.fn)

        return len(self.tp) / den if den > 0 else None

    def f_score(self, beta: float) -> t.Optional[float]:
        prec = self.precision()
        rec = self.recall()

        if prec is not None and rec is not None:
            num = (1 + pow(beta, 2)) * prec * rec
            den = pow(beta, 2) * prec + rec

            return num / den if den > 0 else None

        return None

    def accuracy(self) -> t.Optional[float]:
        den = len(self.tp) + len(self.tn) + len(self.fp) + len(self.fn)

        return (len(self.tp) + len(self.tn)) / den if den > 0 else None

    def balanced_accuracy(self) -> t.Optional[float]:
        tpr = self.sensitivity()
        tnr = self.specificity()

        return (tpr + tnr) / 2 if tnr is not None and tpr is not None else None

    def error_rate(self) -> t.Optional[float]:
        den = len(self.tp) + len(self.tn) + len(self.fp) + len(self.fn)

        return (len(self.fp) + len(self.fn)) / den if den > 0 else None

    def sensitivity(self) -> t.Optional[float]:
        return self.recall()

    def specificity(self) -> t.Optional[float]:
        den = len(self.tn) + len(self.fp)

        return len(self.tn) / den if den > 0 else None


pos2proto = {
    "noun": adaptation_pb2.Pos.POS_NOUN,
    "verb": adaptation_pb2.Pos.POS_VERB,
    "adjective": adaptation_pb2.Pos.POS_ADJECTIVE,
    "adverb": adaptation_pb2.Pos.POS_ADVERB,
}


def str2concept(text: str) -> adaptation_pb2.Concept:
    concept = text.strip().lower()
    lemma, pos = concept.split("/")

    return adaptation_pb2.Concept(lemma=lemma, pos=pos2proto[pos])


AdaptationRule = dict[str, str]


# TODO: Currently, we only evaluate the FIRST entry of the evaluations
class UserEvaluation(t.TypedDict):
    name: str
    ranking: dict[str, int]
    specializations: dict[str, AdaptationRule]
    generalizations: dict[str, AdaptationRule]


class AdaptationEvaluation(BaseEvaluation):
    user_adaptations: dict[str, list[adaptation_pb2.Rule]]
    system_adaptations: dict[str, list[adaptation_pb2.Rule]]

    def __init__(
        self,
        cases: t.Iterable[str],
        query: str,
        duration: float,
        config: RetrievalEvaluationConfig,
        system_adaptations: dict[str, list[adaptation_pb2.Rule]],
        user_evals: t.Sequence[UserEvaluation],
    ) -> None:
        super().__init__(cases, query, duration, config)
        self.system_adaptations = system_adaptations
        self.user_adaptations = defaultdict(list)

        for casename, adaptations in user_evals[0]["generalizations"].items():
            for source, target in adaptations.items():
                self.user_adaptations[casename].append(
                    adaptation_pb2.Rule(
                        source=str2concept(source), target=str2concept(target)
                    )
                )


class RetrievalEvaluation(BaseEvaluation):
    """Class for calculating and storing evaluation measures

    Candiates are fetched automatically from a file.
    The order of the candiates is not relevant for the calculations.
    """

    user_ranking: dict[str, int]
    system_ranking: list[str]

    def __init__(
        self,
        cases: t.Iterable[str],
        query: str,
        duration: float,
        config: RetrievalEvaluationConfig,
        retrieved_cases: t.Sequence[retrieval_pb2.RetrievedCase],
        user_evals: t.Sequence[UserEvaluation],
    ) -> None:
        super().__init__(cases, query, duration, config)
        self.user_ranking = user_evals[0]["ranking"]
        self.system_ranking = [x.id for x in retrieved_cases]

        relevant_keys = set(self.user_ranking)
        not_relevant_keys = {key for key in cases if key not in relevant_keys}

        self.tp = relevant_keys.intersection(set(self.system_ranking))
        self.fp = not_relevant_keys.intersection(set(self.system_ranking))
        self.fn = relevant_keys.difference(set(self.system_ranking))
        self.tn = not_relevant_keys.difference(set(self.system_ranking))

    def __dict__(self):
        out = super().__dict__()
        completeness, correctness = self.correctness_completeness()

        out |= {
            "ndcg": self.ndcg(),
            "average_precision": self.average_precision(),
            "correctness": correctness,
            "completeness": completeness,
        }

        return out

    def average_precision(self) -> t.Optional[float]:
        """Compute the average prescision between two lists of items.

        https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
        """

        score = 0.0
        num_hits = 0.0

        for i, result in enumerate(self.system_ranking):
            if result in self.user_ranking and result not in self.system_ranking[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / len(self.user_ranking)

    def correctness_completeness(self) -> t.Tuple[float, float]:
        orders = 0
        concordances = 0
        disconcordances = 0

        correctness = 1
        completeness = 1

        for user_key_1, user_rank_1 in self.user_ranking.items():
            for user_key_2, user_rank_2 in self.user_ranking.items():
                if user_key_1 != user_key_2 and user_rank_1 > user_rank_2:
                    orders += 1

                    system_rank_1 = self.system_ranking.index(user_key_1)
                    system_rank_2 = self.system_ranking.index(user_key_2)

                    if system_rank_1 is not None and system_rank_2 is not None:
                        if system_rank_1 > system_rank_2:
                            concordances += 1
                        elif system_rank_1 < system_rank_2:
                            disconcordances += 1

        if concordances + disconcordances > 0:
            correctness = (concordances - disconcordances) / (
                concordances + disconcordances
            )
        if orders > 0:
            completeness = (concordances + disconcordances) / orders

        return correctness, completeness

    def ndcg(self) -> float:
        ranking_inv = {
            name: self.config.max_user_rank + 1 - rank
            for name, rank in self.user_ranking.items()
        }
        results_ratings = [ranking_inv.get(result, 0) for result in self.system_ranking]

        return ndcg(results_ratings, len(results_ratings))
