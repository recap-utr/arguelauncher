from __future__ import absolute_import, annotations

import itertools
import json
import logging
import statistics
import typing as t
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rich.table import Table

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


def get_dataframe(eval: dict[str, dict[str, float]], path: Path) -> pd.DataFrame:
    metrics_at_k: set[str] = set().union(*[stage.keys() for stage in eval.values()])
    k_values: set[int] = set()
    metric_names: set[str] = set()

    for metric_at_k in metrics_at_k:
        metric_name, k = metric_at_k.split("@")

        k_values.add(int(k))
        metric_names.add(metric_name)

    data: defaultdict[str, list[t.Any]] = defaultdict(list)

    for (eval_stage, stage_metrics), k in itertools.product(eval.items(), k_values):
        data["stage"].append(eval_stage)
        data["k"].append(k)

        for metric_name in metric_names:
            metric_value = stage_metrics.get(f"{metric_name}@{k}", float("nan"))
            data[metric_name].append(metric_value)

    df = pd.DataFrame(data)
    df.sort_values(["stage", "k"], ascending=True, inplace=True)

    with path.open("w") as f:
        df.to_csv(f, index=False, encoding="utf-8")

    return df


def df_to_table(
    df: pd.DataFrame,
    show_index: bool = False,
    index_name: t.Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    https://gist.github.com/neelabalan/33ab34cf65b43e305c3f12ec6db05938

    Args:
        df: A Pandas DataFrame to be converted to a rich Table.
        show_index: Add a column with a row count to the table. Defaults to True.
        index_name: The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        The rich Table instance, populated with the DataFrame values."""

    rich_table = Table()

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in df.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(df.values.tolist()):
        row = [str(index)] if show_index else []
        row += [
            "{:.3f}".format(x) if isinstance(x, float) else str(x) for x in value_list
        ]
        rich_table.add_row(*row)

    return rich_table
