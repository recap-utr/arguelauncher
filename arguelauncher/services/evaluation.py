from __future__ import absolute_import, annotations

import itertools
import logging
import statistics
import typing as t
import warnings
from abc import ABC, abstractmethod

import ranx
from arg_services.cbr.v1beta import adaptation_pb2, retrieval_pb2, retrieval_pb2_grpc
from google.protobuf.json_format import MessageToDict
from typing_extensions import override

from arguelauncher import model
from arguelauncher.config.cbr import CbrConfig, EvaluationConfig
from arguelauncher.config.nlp import NLP_CONFIG
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


def cutoffs(limit: t.Optional[int]) -> list[t.Optional[int]]:
    return [
        cutoff for cutoff in [1, 3, 5, 10, 25, 50] if limit is None or cutoff <= limit
    ] + [None]


def cutoff_metrics(metrics: t.Iterable[str], limit: t.Optional[int]) -> list[str]:
    return [
        metric_name(metric, k)
        for metric, k in itertools.product(metrics, cutoffs(limit))
    ]


def metric_name(name: str, k: t.Optional[int]) -> str:
    return name if k is None else f"{name}@{k}"


class AbstractEvaluation(ABC):
    cases: t.Mapping[str, model.Graph]
    query: model.Graph
    config: EvaluationConfig
    qrels: ranx.Qrels
    run: ranx.Run
    limit: int | None

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
        self.limit = limit

        try:
            self.qrels = ranx.Qrels(qrels)
            self.run = ranx.Run(run)
        except ValueError:
            self.qrels = ranx.Qrels()
            self.run = ranx.Run()

    @property
    def benchmark(self) -> model.CbrEvaluation:
        return self.query.userdata["cbrEvaluations"][0]

    def compute_metrics(self) -> dict[str, float]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            eval_results = ranx.evaluate(
                self.qrels,
                self.run,
                metrics=cutoff_metrics(RANX_METRICS, self.limit),
            )

        assert isinstance(eval_results, dict)

        for k in cutoffs(self.limit):
            correctness, completeness = self.correctness_completeness(k)

            eval_results[metric_name("correctness", k)] = correctness
            eval_results[metric_name("completeness", k)] = completeness

        return eval_results

    @abstractmethod
    def get_results(self) -> t.Any:
        ...

    def correctness_completeness(self, k: t.Optional[int]) -> tuple[float, float]:
        keys = set(self.qrels.keys()).intersection(set(self.run.keys()))

        scores = [self._correctness_completeness(key, k) for key in keys]
        correctness_scores = [score[0] for score in scores]
        completeness_scores = [score[1] for score in scores]

        try:
            return statistics.mean(correctness_scores), statistics.mean(
                completeness_scores
            )
        except statistics.StatisticsError:
            return float("nan"), float("nan")

    def _correctness_completeness(
        self, key: str, k: t.Optional[int]
    ) -> t.Tuple[float, float]:
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
            metrics[metric_name("similarity", k)] = statistics.mean(
                case.similarity for case in self.retrieved_cases[:k]
            )

        return metrics


class AdaptationEvaluation(AbstractEvaluation):
    user_adaptations: dict[str, list[adaptation_pb2.Rule]]
    system_response: t.Mapping[str, adaptation_pb2.AdaptedCaseResponse]
    retrieval_client: t.Optional[retrieval_pb2_grpc.RetrievalServiceStub]
    cbr_config: CbrConfig

    @override
    def __init__(
        self,
        cases: t.Mapping[str, model.Graph],
        query: model.Graph,
        config: EvaluationConfig,
        system_response: t.Mapping[str, adaptation_pb2.AdaptedCaseResponse],
        retrieval_client: t.Optional[retrieval_pb2_grpc.RetrievalServiceStub],
        cbr_config: CbrConfig,
    ) -> None:
        # TODO: We only evaluate the first cbrEvaluation
        generalizations = query.userdata["cbrEvaluations"][0].get("generalizations")

        if generalizations is None:
            raise ValueError("No adaptations are present in user benchmark.")

        self.system_response = system_response
        self.retrieval_client = retrieval_client
        self.cbr_config = cbr_config

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

        super().__init__(cases, query, config, qrels, run, limit=None)

    @property
    def adapted_cases(self) -> dict[str, model.Graph]:
        return {
            name: model.Graph.from_protobuf(
                res.case, self.original_cases[name].userdata
            )
            for name, res in self.system_response.items()
        }

    @property
    def original_cases(self) -> dict[str, model.Graph]:
        return {name: self.cases[name] for name in self.system_response.keys()}

    @property
    def system_adaptations(self) -> dict[str, list[adaptation_pb2.Rule]]:
        return {
            casename: list(result.applied_rules)
            for casename, result in self.system_response.items()
        }

    @override
    def compute_metrics(self) -> dict[str, float]:
        metrics = super().compute_metrics()

        if self.retrieval_client:
            original_similarities = self.retrieval_client.Similarities(
                retrieval_pb2.SimilaritiesRequest(
                    cases=[
                        case.to_protobuf(self.cbr_config.graph2text)
                        for case in self.original_cases.values()
                    ],
                    query=self.query.to_protobuf(self.cbr_config.graph2text),
                    nlp_config=NLP_CONFIG[self.cbr_config.nlp_config],
                )
            ).similarities

            adapted_similarities = self.retrieval_client.Similarities(
                retrieval_pb2.SimilaritiesRequest(
                    cases=[
                        case.to_protobuf(self.cbr_config.graph2text)
                        for case in self.adapted_cases.values()
                    ],
                    query=self.query.to_protobuf(self.cbr_config.graph2text),
                    nlp_config=NLP_CONFIG[self.cbr_config.nlp_config],
                )
            ).similarities

            metrics["sim_original"] = statistics.mean(
                x.semantic_similarity for x in original_similarities
            )
            metrics["sim_adapted"] = statistics.mean(
                x.semantic_similarity for x in adapted_similarities
            )
            metrics["sim_improvement"] = (
                metrics["sim_adapted"] / metrics["sim_original"]
            ) - 1

        metrics["adapted_ratio"] = 1 if self.run.size > 0 else 0
        metrics["rule_ratio"] = statistics.mean(
            len(self.system_adaptations[casename]) / len(user_rules)
            for casename, user_rules in self.user_adaptations.items()
        )

        return metrics

    @override
    def get_results(self) -> t.Any:
        out = {}

        for case_name, res in self.system_response.items():
            if not self.config.export_graph:
                res.case.graph.Clear()

            out[case_name] = MessageToDict(res)

            if expert_rules := self.user_adaptations.get(case_name):
                out[case_name]["expertRules"] = [
                    MessageToDict(rule) for rule in expert_rules
                ]

        return out
