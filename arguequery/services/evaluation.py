from __future__ import absolute_import, annotations

import json
import logging
import math
import os
from collections import OrderedDict
from typing import Dict, List, Set, Tuple

import arguebuf as ag
from arguequery.libs.ndcg import ndcg
from arguequery.models.result import Result

logger = logging.getLogger("recap")
from arguequery.config import config


class Evaluation(object):
    """Class for calculating and storing evaluation measures

    Candiates are fetched automatically from a file.
    The order of the candiates is not relevant for the calculations.
    """

    user_candidates: List[str]
    user_rankings: Dict[str, int]
    system_candidates: List[str]
    system_rankings: Dict[str, int]

    precision: float
    recall: float
    average_precision: float
    correctness: float
    completeness: float
    ndcg: float

    def __init__(
        self, case_base: Dict[str, ag.Graph], results: List[Result], query: ag.Graph
    ) -> None:
        self._get_candidates(case_base, query.name)

        self.system_rankings = {res.graph.name: i + 1 for i, res in enumerate(results)}
        self.system_candidates = list(self.system_rankings.keys())

        self._calculate_metrics(case_base, results)

    def as_dict(self):
        return {
            "unranked": {
                "Precision": self.precision,
                "Recall": self.recall,
                "F1 Score": self.f_score(1),
                "F2 Score": self.f_score(2),
            },
            "ranked": {
                "Average Precision": self.average_precision,
                "NDCG": self.ndcg,
                "Correctness": self.correctness,
                "Completeness": self.completeness,
            },
        }

    def _get_candidates(self, case_base: Dict[str, ag.Graph], filename: str) -> None:
        filepath = os.path.join(config["candidates_folder"], filename)

        try:
            with open(filepath) as file:
                data = json.load(file)
                self.user_candidates = data["candidates"]
                self.user_rankings = data["rankings"]
        except Exception:
            self.user_candidates = []
            self.user_rankings = {}

    def _calculate_metrics(
        self, case_base: Dict[str, ag.Graph], results: List[Result]
    ) -> None:
        relevant_keys = set(self.user_candidates)
        not_relevant_keys = {key for key in case_base if key not in relevant_keys}

        tp = relevant_keys.intersection(set(self.system_candidates))
        fp = not_relevant_keys.intersection(set(self.system_candidates))
        fn = relevant_keys.difference(set(self.system_candidates))

        self.precision = len(tp) / (len(tp) + len(fp))
        self.recall = len(tp) / (len(tp) + len(fn))

        self._average_precision()
        self._correctness_completeness(case_base, results)
        self._ndcg()

    def f_score(self, beta: int):
        if self.precision + self.recall == 0:
            return 0

        return ((1 + math.pow(beta, 2)) * self.precision * self.recall) / (
            math.pow(beta, 2) * self.precision + self.recall
        )

    def _average_precision(self) -> None:
        """Compute the average prescision between two lists of items.

        https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
        """

        score = 0.0
        num_hits = 0.0

        for i, result in enumerate(self.system_candidates):
            if (
                result in self.user_candidates
                and result not in self.system_candidates[:i]
            ):
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        self.average_precision = score / len(self.user_candidates)

    def _correctness_completeness(
        self, case_base: Dict[str, ag.Graph], results: List[Result]
    ) -> None:
        orders = 0
        concordances = 0
        disconcordances = 0

        self.correctness = 1
        self.completeness = 1

        for user_key_1, user_rank_1 in self.user_rankings.items():
            for user_key_2, user_rank_2 in self.user_rankings.items():
                if user_key_1 != user_key_2:
                    system_rank_1 = self.system_rankings.get(user_key_1)
                    system_rank_2 = self.system_rankings.get(user_key_2)

                    if user_rank_1 > user_rank_2:
                        orders += 1

                        if system_rank_1 is not None and system_rank_2 is not None:
                            if system_rank_1 > system_rank_2:
                                concordances += 1
                            elif system_rank_1 < system_rank_2:
                                disconcordances += 1

        if concordances + disconcordances > 0:
            self.correctness = (concordances - disconcordances) / (
                concordances + disconcordances
            )
        if orders > 0:
            self.completeness = (concordances + disconcordances) / orders

    def _ndcg(self) -> None:
        ranking_inv = {
            name: config["candidates_max_rank"] + 1 - rank
            for name, rank in self.user_rankings.items()
        }
        results_ratings = [
            ranking_inv.get(result, 0) for result in self.system_rankings.keys()
        ]

        self.ndcg = ndcg(results_ratings, len(results_ratings))
