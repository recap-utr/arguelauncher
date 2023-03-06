from __future__ import absolute_import, annotations

import json
import logging
import statistics
import typing as t
from collections import defaultdict
from pathlib import Path

from arguelauncher.config.cbr import EvaluationConfig
from arguelauncher.services.evaluation import AbstractEvaluation

logger = logging.getLogger(__name__)


def get_individual(
    eval: AbstractEvaluation,
):
    return {
        "metrics": eval.compute_metrics(),
        "results": eval.get_results(),
    }


def get_named_individual(eval_map: t.Mapping[str, AbstractEvaluation]):
    return {name: get_individual(eval) for name, eval in eval_map.items()}


AGGREGATORS: dict[str, t.Callable[[t.Sequence[float]], float]] = {
    "mean": statistics.mean,
    "min": min,
    "max": max,
}


def _aggregate_eval_values(
    values: t.Sequence[float], aggregators: t.Collection[str]
) -> t.Union[None, float, dict[str, float]]:
    if not values:
        return None

    if len(aggregators) == 1:
        key = next(iter(aggregators))
        return round(AGGREGATORS[key](values), 3)

    return {key: round(func(values), 3) for key, func in AGGREGATORS.items()}


# One eval_map for each query
def get_aggregated(
    eval_maps: t.Iterable[t.Mapping[str, AbstractEvaluation]], config: EvaluationConfig
):
    metrics: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for eval_map in eval_maps:
        for eval_name, eval_value in eval_map.items():
            for metric_name, metric_value in eval_value.compute_metrics().items():
                metrics[eval_name][metric_name].append(metric_value)

    return {
        eval_name: {
            metric_name: _aggregate_eval_values(metric_values, config.aggregators)
            for metric_name, metric_values in eval_metrics.items()
        }
        for eval_name, eval_metrics in metrics.items()
    }


def get_json(value: dict[str, t.Any]) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def get_file(
    value: dict[str, t.Any],
    path: Path,
) -> None:
    with path.open("w") as f:
        f.write(get_json(value))
