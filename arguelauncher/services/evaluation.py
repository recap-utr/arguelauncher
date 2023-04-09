from __future__ import absolute_import, annotations

import itertools
import logging
import statistics
import typing as t
from abc import ABC, abstractmethod

import ranx
from arg_services.cbr.v1beta import adaptation_pb2, retrieval_pb2
from google.protobuf.json_format import MessageToDict
from typing_extensions import override

from arguelauncher import model
from arguelauncher.config.cbr import EvaluationConfig
from arguelauncher.model import parse_rules

logger = logging.getLogger(__name__)


# https://amenra.github.io/ranx/metrics/
RANX_METRICS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "hits",
    "hit_rate",
    "mrr",
    "map",
    "ndcg",
)


def cutoffs(limit: t.Optional[int]) -> list[int]:
    if limit is None:
        return [1000]

    return [limit // 4, limit // 2, 1000]


def cutoff_metrics(metrics: t.Iterable[str], limit: t.Optional[int]) -> list[str]:
    return [
        metric if k is None else f"{metric}@{k}"
        for metric, k in itertools.product(metrics, cutoffs(limit))
    ]


class AbstractEvaluation(ABC):
    cases: t.Mapping[str, model.Graph]
    query: model.Graph
    config: EvaluationConfig
    qrels: ranx.Qrels
    run: ranx.Run
    limit: t.Optional[int]

    def __init__(
        self,
        cases: t.Mapping[str, model.Graph],
        query: model.Graph,
        config: EvaluationConfig,
        qrels: dict[str, dict[str, int]],
        run: dict[str, dict[str, float]],
        limit: t.Optional[int],
    ):
        self.cases = cases
        self.query = query
        self.config = config
        self.qrels = ranx.Qrels(qrels)
        # self.qrels.set_relevance_level(self.config.relevance_levels)
        self.run = ranx.Run(run)
        self.limit = limit

    def compute_metrics(self) -> dict[str, float]:
        eval_results = ranx.evaluate(
            self.qrels,
            self.run,
            metrics=cutoff_metrics(RANX_METRICS, self.limit),
        )

        assert isinstance(eval_results, dict)

        for k in cutoffs(self.limit):
            correctness, completeness = self.correctness_completeness(k)

            eval_results[f"correctness@{k}"] = correctness
            eval_results[f"completeness@{k}"] = completeness

        return eval_results

    @abstractmethod
    def get_results(self) -> t.Any:
        ...

    def correctness_completeness(self, k: int) -> t.Tuple[float, float]:
        keys = set(self.qrels.keys()).intersection(set(self.run.keys()))

        scores = [self._correctness_completeness(key, k) for key in keys]
        correctness_scores = [score[0] for score in scores]
        completeness_scores = [score[1] for score in scores]

        return statistics.mean(correctness_scores), statistics.mean(completeness_scores)

    def _correctness_completeness(self, key: str, k: int) -> t.Tuple[float, float]:
        qrel = self.qrels[key]
        sorted_run = sorted(self.run[key].items(), key=lambda x: x[1], reverse=True)
        run_ranking = {x[0]: i + 1 for i, x in enumerate(sorted_run[:k])}

        orders = 0
        concordances = 0
        disconcordances = 0

        correctness = 1
        completeness = 1

        for user_key_1, user_rank_1 in qrel.items():
            for user_key_2, user_rank_2 in qrel.items():
                if user_key_1 != user_key_2 and user_rank_1 > user_rank_2:
                    orders += 1

                    system_rank_1 = run_ranking.get(user_key_1)
                    system_rank_2 = run_ranking.get(user_key_2)

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


class RetrievalEvaluation(AbstractEvaluation):
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
        limit: t.Optional[int],
    ) -> None:
        true_ranks = query.userdata["cbrEvaluations"][0]["ranking"]
        predicted_scores = {case.id: case.similarity for case in retrieved_cases}
        super().__init__(
            cases,
            query,
            config,
            {"query": true_ranks},
            {"query": predicted_scores},
            limit,
        )

        self.retrieved_cases = retrieved_cases

    @override
    def get_results(self) -> t.Any:
        out = []

        for value in self.retrieved_cases:
            if not self.config.export_graph:
                value.graph.graph.Clear()

            out.append(MessageToDict(value))

        return out

    @override
    def compute_metrics(self) -> dict[str, float]:
        metrics = super().compute_metrics()

        for k in cutoffs(self.limit):
            metrics[f"similarity@{k}"] = statistics.mean(
                case.similarity for case in self.retrieved_cases[:k]
            )

        return metrics


class AdaptationEvaluation(AbstractEvaluation):
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
        qrels = {
            casename: {
                f"{rule.source.lemma}/{rule.source.pos}": i + 1
                for i, rule in enumerate(user_rules)
            }
            for casename, user_rules in self.user_adaptations.items()
        }
        run = {
            casename: {
                f"{rule.source.lemma}/{rule.source.pos}": rule.source.score
                for rule in system_response[casename].applied_rules
            }
            for casename in qrels.keys()
            if len(system_response[casename].applied_rules) > 0
        }

        if not run:
            raise ValueError("Computed adaptations are not present in user benchmark.")

        super().__init__(cases, query, config, qrels, run, limit=None)

    @property
    def system_adaptations(self) -> dict[str, list[adaptation_pb2.Rule]]:
        return {
            casename: list(result.applied_rules)
            for casename, result in self.system_response.items()
        }

    @override
    def get_results(self) -> t.Any:
        out = {}

        for key, value in self.system_response.items():
            if not self.config.export_graph:
                value.case.graph.Clear()

            out[key] = MessageToDict(value)

        return out
