from __future__ import absolute_import, annotations

import json
import logging
import typing as t
from collections import defaultdict
from pathlib import Path

import nptyping as npt
import numpy as np

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


AGGREGATORS: dict[
    str, t.Callable[[npt.NDArray[npt.Shape["*"], npt.Floating]], npt.Floating]
] = {
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
}


def _aggregate_eval_values(
    values: npt.NDArray[npt.Shape["*"], npt.Floating],
) -> dict[str, t.Optional[npt.Floating]]:
    values = np.array(v for v in values if v is not None)

    if not values:
        return {key: None for key in AGGREGATORS.keys()}

    return {key: round(func(values), 3) for key, func in AGGREGATORS.items()}


# One eval_map for each query
def get_aggregated(eval_maps: t.Iterable[t.Mapping[str, AbstractEvaluation]]):
    metrics: defaultdict[tuple[str, str], list[npt.Floating]] = defaultdict(list)

    for eval_map in eval_maps:
        for eval_name, eval_value in eval_map.items():
            for metric_name, metric_value in eval_value.compute_metrics().items():
                metrics[(eval_name, metric_name)].append(metric_value)

    return {
        eval_name: {metric_name: _aggregate_eval_values(np.array(metric_values))}
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
