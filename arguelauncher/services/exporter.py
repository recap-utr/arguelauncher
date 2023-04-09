from __future__ import absolute_import, annotations

import json
import logging
import statistics
import typing as t
from collections import defaultdict
from pathlib import Path

import pandas as pd

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


# One eval_map for each query
def get_aggregated(
    eval_maps: t.Iterable[t.Mapping[str, AbstractEvaluation]],
    config: t.Optional[EvaluationConfig] = None,
) -> dict[str, dict[str, float]]:
    if config is None:
        config = EvaluationConfig()

    metrics: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for eval_map in eval_maps:
        for eval_name, eval_value in eval_map.items():
            for metric_name, metric_value in eval_value.compute_metrics().items():
                metrics[eval_name][metric_name].append(metric_value)

    return {
        eval_name: {
            metric_name: statistics.mean(metric_values)
            for metric_name, metric_values in eval_metrics.items()
            if len(metric_values) > 0
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


def get_dataframe(eval: dict[str, dict[str, float]], path: Path) -> None:
    data: defaultdict[str, list[t.Any]] = defaultdict(list)

    for eval_stage, stage_metrics in eval.items():
        for metric_k, metric_value in stage_metrics.items():
            metric_name, k = metric_k.split("@")

            data["stage"].append(eval_stage)
            data["metric"].append(metric_name)
            data["k"].append(int(k))
            data["value"].append(metric_value)

    df = pd.DataFrame(data)

    with path.open("w") as f:
        df.to_csv(f, index=False, encoding="utf-8")
