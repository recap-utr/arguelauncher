from __future__ import absolute_import, annotations

import json
import logging
import statistics
import typing as t
from collections import defaultdict
from pathlib import Path

from arguelauncher.services.evaluation import BaseEvaluation

logger = logging.getLogger(__name__)


def get_individual(
    eval: BaseEvaluation,
):
    return {
        "metrics": eval.compute_metrics(),
        "results": eval.get_results(),
    }


def get_named_individual(eval_map: t.Mapping[str, BaseEvaluation]):
    return {name: get_individual(eval) for name, eval in eval_map.items()}


AGGREGATORS: dict[str, t.Callable[[t.Iterable[float]], float]] = {
    "mean": statistics.mean,
    "min": min,
    "max": max,
}


def _aggregate_eval_values(
    values: t.Iterable[t.Optional[float]],
) -> dict[str, t.Optional[float]]:
    values = [v for v in values if v is not None]

    if not values:
        return {key: None for key in AGGREGATORS.keys()}

    return {key: round(func(values), 3) for key, func in AGGREGATORS.items()}


# One eval_map for each query
def get_aggregated(eval_maps: t.Iterable[t.Mapping[str, BaseEvaluation]]):
    metrics: defaultdict[tuple[str, str], list[t.Optional[float]]] = defaultdict(list)

    for eval_map in eval_maps:
        for eval_name, eval_value in eval_map.items():
            for metric_name, metric_value in eval_value.compute_metrics().items():
                metrics[(eval_name, metric_name)].append(metric_value)

    return {
        eval_name: {metric_name: _aggregate_eval_values(metric_values)}
        for (eval_name, metric_name), metric_values in metrics.items()
    }


def get_json(value: dict[str, t.Any]) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def get_file(
    value: dict[str, t.Any],
    path: Path,
) -> None:
    with path.open("w") as f:
        f.write(get_json(value))
