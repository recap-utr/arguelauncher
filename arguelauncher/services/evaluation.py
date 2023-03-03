from __future__ import absolute_import, annotations

import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from arg_services.cbr.v1beta import adaptation_pb2, retrieval_pb2
from google.protobuf.json_format import MessageToDict
from sklearn import metrics
from typing_extensions import override

from arguelauncher import model
from arguelauncher.config.cbr import EvaluationConfig
from arguelauncher.model import parse_rules

logger = logging.getLogger(__name__)


@dataclass
class Labeling:
    true: npt.NDArray[t.Any]
    predicted: npt.NDArray[t.Any]


@dataclass
class Ranking(Labeling):
    true_ranks: dict[str, int]
    predicted_scores: dict[str, float]
    predicted_ranks: dict[str, int]

    def __init__(self, true_ranks: dict[str, int], predicted_scores: dict[str, float]):
        super().__init__(np.array(true_ranks.keys()), np.array(predicted_scores.keys()))
        self.true_ranks = true_ranks
        self.predicted_scores = predicted_scores
        self.predicted_ranks = {
            key: i + 1
            for i, key in enumerate(
                sorted(
                    self.predicted_scores.keys(),
                    key=lambda x: self.predicted_scores[x],
                    reverse=True,
                )
            )
        }


class AbstractEvaluation(ABC):
    cases: t.Mapping[str, model.Graph]
    query: model.Graph
    config: EvaluationConfig

    def __init__(
        self,
        cases: t.Mapping[str, model.Graph],
        query: model.Graph,
        config: EvaluationConfig,
    ):
        self.cases = cases
        self.query = query
        self.config = config

    @abstractmethod
    def compute_metrics(self) -> dict[str, np.float_]:
        ...

    @abstractmethod
    def get_results(self) -> t.Any:
        ...

    def _aggregate(self, values: t.Iterable[np.float_]) -> np.float_:
        return np.mean(np.array(values))


class UnrankedEvaluation(AbstractEvaluation):
    labels: t.Collection[Labeling]

    def __init__(
        self,
        cases: t.Mapping[str, model.Graph],
        query: model.Graph,
        config: EvaluationConfig,
        labels: t.Collection[Labeling],
    ):
        super().__init__(cases, query, config)
        self.labels = labels

    @override
    def compute_metrics(self) -> dict[str, np.float_]:
        out = {
            "precision": self.precision(),
            "recall": self.recall(),
            "accuracy": self.accuracy(),
            "balanced_accuracy": self.balanced_accuracy(),
        }

        for beta in self.config.f_scores:
            out[f"f{beta}"] = self.f_score(beta)

        return out

    def precision(self) -> np.float_:
        return self._aggregate(
            metrics.precision_score(label.true, label.predicted)
            for label in self.labels
        )

    def recall(self) -> np.float_:
        return self._aggregate(
            metrics.recall_score(label.true, label.predicted) for label in self.labels
        )

    def f_score(self, beta: float) -> np.float_:
        return self._aggregate(
            metrics.fbeta_score(label.true, label.predicted, beta=beta)
            for label in self.labels
        )

    def accuracy(self) -> np.float_:
        return self._aggregate(
            t.cast(np.float_, metrics.accuracy_score(label.true, label.predicted))
            for label in self.labels
        )

    def balanced_accuracy(self) -> np.float_:
        return self._aggregate(
            metrics.balanced_accuracy_score(label.true, label.predicted)
            for label in self.labels
        )


class RankedEvaluation(UnrankedEvaluation):
    rankings: t.Collection[Ranking]

    def __init__(
        self,
        cases: t.Mapping[str, model.Graph],
        query: model.Graph,
        config: EvaluationConfig,
        rankings: t.Collection[Ranking],
    ) -> None:
        super().__init__(cases, query, config, rankings)
        self.rankings = rankings

    def average_precision(self) -> np.float_:
        return self._aggregate(
            t.cast(
                np.float_,
                metrics.average_precision_score(
                    [
                        1 if key in ranking.true_ranks else 0
                        for key in ranking.predicted_scores.keys()
                    ],
                    list(ranking.predicted_scores.values()),
                ),
            )
            for ranking in self.rankings
        )

    def ndcg(self) -> np.float_:
        return self._aggregate(
            metrics.ndcg_score(
                [
                    ranking.true_ranks.get(key, 0)
                    for key in ranking.predicted_scores.keys()
                ],
                list(ranking.predicted_scores.values()),
            )
            for ranking in self.rankings
        )

    def correctness_completeness(self) -> t.Tuple[np.float_, np.float_]:
        scores = [self._correctness_completeness(ranking) for ranking in self.rankings]
        correctness_scores = [score[0] for score in scores]
        completeness_scores = [score[1] for score in scores]

        return self._aggregate(correctness_scores), self._aggregate(completeness_scores)

    def _correctness_completeness(
        self, ranking: Ranking
    ) -> t.Tuple[np.float_, np.float_]:
        orders = 0
        concordances = 0
        disconcordances = 0

        correctness = 1
        completeness = 1

        for user_key_1, user_rank_1 in ranking.true_ranks.items():
            for user_key_2, user_rank_2 in ranking.true_ranks.items():
                if user_key_1 != user_key_2 and user_rank_1 > user_rank_2:
                    orders += 1

                    system_rank_1 = ranking.predicted_ranks.get(user_key_1)
                    system_rank_2 = ranking.predicted_ranks.get(user_key_2)

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

        return np.floating(correctness), np.floating(completeness)

    @override
    def compute_metrics(self) -> dict[str, np.float_]:
        out = super().compute_metrics()
        completeness, correctness = self.correctness_completeness()

        out |= {
            "ndcg": self.ndcg(),
            "average_precision": self.average_precision(),
            "correctness": correctness,
            "completeness": completeness,
        }

        return out


class RetrievalEvaluation(RankedEvaluation):
    """Class for calculating and storing evaluation measures

    Candiates are fetched automatically from a file.
    The order of the candiates is not relevant for the calculations.
    """

    @override
    def __init__(
        self,
        cases: t.Mapping[str, model.Graph],
        query: model.Graph,
        config: EvaluationConfig,
        retrieved_cases: t.Sequence[retrieval_pb2.RetrievedCase],
    ) -> None:
        true_ranks = query.userdata["cbrEvaluations"][0]["ranking"]
        predicted_scores = {case.id: case.similarity for case in retrieved_cases}
        super().__init__(cases, query, config, [Ranking(true_ranks, predicted_scores)])

        self.retrieved_cases = retrieved_cases

    @override
    def get_results(self) -> t.Any:
        return [MessageToDict(case) for case in self.retrieved_cases]


class AdaptationEvaluation(RankedEvaluation):
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
        # TODO: We only evaluate the first cbrEvaluation
        generalizations = query.userdata["cbrEvaluations"][0].get("generalizations")
        assert generalizations is not None

        self.system_response = system_response
        self.user_adaptations = {
            casename: list(parse_rules(adaptations))
            for casename, adaptations in generalizations.items()
        }
        rankings: list[Ranking] = []

        for casename, user_rules in self.user_adaptations.items():
            res = system_response[casename]

            rankings.append(
                Ranking(
                    {
                        f"{rule.source.lemma}/{rule.source.pos}": i + 1
                        for i, rule in enumerate(user_rules)
                    },
                    {
                        f"{rule.source.lemma}/{rule.source.pos}": rule.source.score
                        for rule in res.applied_rules
                    },
                )
            )

            # TODO: Compare similarity of retrieved and adapted graph

            # TODO: Investigate deliberation-based adaptation for comparison
            # i.e., adapting all concepts to themselves

        super().__init__(cases, query, config, rankings)

    @property
    def system_adaptations(self) -> dict[str, list[adaptation_pb2.Rule]]:
        return {
            casename: list(result.applied_rules)
            for casename, result in self.system_response.items()
        }

    @override
    def get_results(self) -> t.Any:
        return {
            key: MessageToDict(value) for key, value in self.system_response.items()
        }
