from __future__ import absolute_import, annotations

import logging
import typing as t
from abc import ABC, abstractmethod

from arg_services.cbr.v1beta import adaptation_pb2, retrieval_pb2
from google.protobuf.json_format import MessageToDict
from typing_extensions import override

from arguelauncher import model
from arguelauncher.config.cbr import EvaluationConfig
from arguelauncher.libs.ndcg import ndcg
from arguelauncher.model import parse_rules

logger = logging.getLogger(__name__)


class BaseEvaluation(ABC):
    cases: t.Mapping[str, model.Graph]
    query: model.Graph
    config: EvaluationConfig
    tp: set[str]
    fp: set[str]
    fn: set[str]
    tn: set[str]

    def __init__(
        self,
        cases: t.Mapping[str, model.Graph],
        query: model.Graph,
        config: EvaluationConfig,
    ):
        self.cases = cases
        self.query = query
        self.config = config

    def compute_metrics(self) -> dict[str, t.Optional[float]]:
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

    @abstractmethod
    def get_results(self) -> t.Any:
        ...

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


class RetrievalEvaluation(BaseEvaluation):
    """Class for calculating and storing evaluation measures

    Candiates are fetched automatically from a file.
    The order of the candiates is not relevant for the calculations.
    """

    user_ranking: dict[str, int]
    system_ranking: t.Sequence[retrieval_pb2.RetrievedCase]

    @override
    def __init__(
        self,
        cases: t.Mapping[str, model.Graph],
        query: model.Graph,
        config: EvaluationConfig,
        system_ranking: t.Sequence[retrieval_pb2.RetrievedCase],
    ) -> None:
        super().__init__(cases, query, config)
        self.user_ranking = query.userdata["cbrEvaluations"][0]["ranking"]
        self.system_ranking = system_ranking

        relevant_keys = set(self.user_ranking)
        not_relevant_keys = {key for key in cases if key not in relevant_keys}

        ranked_ids = {case.id for case in self.system_ranking}
        self.tp = relevant_keys.intersection(ranked_ids)
        self.fp = not_relevant_keys.intersection(ranked_ids)
        self.fn = relevant_keys.difference(ranked_ids)
        self.tn = not_relevant_keys.difference(ranked_ids)

    @property
    def system_ranking_ids(self) -> t.Sequence[str]:
        return [case.id for case in self.system_ranking]

    @override
    def compute_metrics(self):
        out = super().compute_metrics()
        completeness, correctness = self.correctness_completeness()

        out |= {
            "ndcg": self.ndcg(),
            "average_precision": self.average_precision(),
            "correctness": correctness,
            "completeness": completeness,
        }

        return out

    @override
    def get_results(self) -> t.Any:
        return [MessageToDict(case) for case in self.system_ranking]

    def average_precision(self) -> t.Optional[float]:
        """Compute the average prescision between two lists of items.

        https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
        """

        score = 0.0
        num_hits = 0.0

        for i, result in enumerate(self.system_ranking_ids):
            if (
                result in self.user_ranking
                and result not in self.system_ranking_ids[:i]
            ):
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

                    system_rank_1 = self.system_ranking_ids.index(user_key_1)
                    system_rank_2 = self.system_ranking_ids.index(user_key_2)

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
        results_ratings = [
            ranking_inv.get(result, 0) for result in self.system_ranking_ids
        ]

        return ndcg(results_ratings, len(results_ratings))


class AdaptationEvaluation(BaseEvaluation):
    user_adaptations: dict[str, list[adaptation_pb2.Rule]]
    system_response: t.Mapping[str, adaptation_pb2.AdaptedCaseResponse]

    @override
    def __init__(
        self,
        cases: t.Mapping[str, model.Graph],
        query: model.Graph,
        config: EvaluationConfig,
        system_response: t.Mapping[str, adaptation_pb2.AdaptedCaseResponse],
    ) -> None:
        super().__init__(cases, query, config)

        # TODO: We only evaluate the first cbrEvaluation
        generalizations = query.userdata["cbrEvaluations"][0].get("generalizations")
        assert generalizations is not None

        self.system_response = system_response
        self.user_adaptations = {
            casename: list(parse_rules(adaptations))
            for casename, adaptations in generalizations.items()
        }

        for casename, user_rules in self.user_adaptations.items():
            case_response = system_response[casename]
            all_concepts = {
                *case_response.extracted_concepts,
                *case_response.discarded_concepts,
            }

            self.compute_confusion_matrix(
                case_response.applied_rules, user_rules, all_concepts
            )

            # TODO: Compare similarity of retrieved and adapted graph

            # TODO: Support ranked measures for adaptation

            # TODO: Investigate deliberation-based adaptation for comparison
            # i.e., adapting all concepts to themselves

    @property
    def system_adaptations(self) -> dict[str, list[adaptation_pb2.Rule]]:
        return {
            casename: list(result.applied_rules)
            for casename, result in self.system_response.items()
        }

    @override
    def compute_metrics(self):
        return super().compute_metrics()

    @override
    def get_results(self) -> t.Any:
        return {
            key: MessageToDict(value) for key, value in self.system_response.items()
        }

    def compute_confusion_matrix(
        self,
        system_rules: t.Collection[adaptation_pb2.Rule],
        user_rules: t.Collection[adaptation_pb2.Rule],
        all_concepts: t.AbstractSet[adaptation_pb2.Concept],
    ) -> None:
        tp: set[str] = set()
        fn: set[str] = set()
        fp: set[str] = set()

        computed_adaptations = {rule.source: rule.target for rule in system_rules}
        benchmark_adaptations = {rule.source: rule.target for rule in user_rules}

        for concept in benchmark_adaptations.keys():
            if concept in computed_adaptations:
                tp.add(str(concept))
            else:
                fn.add(str(concept))

        for concept in computed_adaptations.keys():
            if concept not in benchmark_adaptations:
                fp.add(str(concept))

        self.tp = tp
        self.fn = fn
        self.fp = fp
        self.tn = {
            str(x)
            for x in all_concepts
            if x not in computed_adaptations and x not in benchmark_adaptations
        }
