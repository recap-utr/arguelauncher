from __future__ import annotations

import logging
import typing as t
from pathlib import Path
from timeit import default_timer as timer

import arguebuf as ag
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from rich import print_json
from rich.progress import Progress

from arguelauncher import model
from arguelauncher.config.cbr import CbrConfig
from arguelauncher.services import exporter
from arguelauncher.services.adaptation import adapt
from arguelauncher.services.evaluation import (
    AbstractEvaluation,
    AdaptationEvaluation,
    RetrievalEvaluation,
)
from arguelauncher.services.retrieval import retrieve

log = logging.getLogger(__name__)


cs = ConfigStore.instance()
cs.store(name="cbr", node=CbrConfig)


@hydra.main(version_base=None, config_path="config", config_name="app")
def main(config: CbrConfig) -> None:
    """Calculate similarity of queries and case base"""
    with Progress() as progress:
        task_initializing = progress.add_task("Initializing...", total=1)
        output_folder = Path(HydraConfig.get().runtime.output_dir)

        cases = {
            str(file.relative_to(config.path.cases.parent).with_suffix("")): t.cast(
                model.Graph,
                ag.load.file(file, config=ag.load.Config(GraphClass=model.Graph)),
            )
            for file in Path(config.path.cases).glob(config.path.cases_pattern)
        }
        requests = {
            str(file.relative_to(config.path.requests).with_suffix("")): t.cast(
                model.Graph,
                ag.load.file(file, config=ag.load.Config(GraphClass=model.Graph)),
            )
            for file in Path(config.path.requests).glob(config.path.requests_pattern)
        }
        ordered_requests = list(requests.values())
        progress.update(task_initializing, completed=1)

        # RETRIEVAL
        retrieval_task = progress.add_task("Retrieving...", total=len(ordered_requests))
        retrieval_start = timer()
        _, retrieve_response = retrieve(cases, ordered_requests, config)
        retrieval_responses = list(retrieve_response.query_responses)
        progress.update(retrieval_task, completed=len(ordered_requests))

        retrieval_duration = timer() - retrieval_start

        # ADAPTATION
        adaptation_start = timer()
        adaptation_responses = []
        adaptation_task = progress.add_task(
            "Adapting...", total=len(retrieval_responses)
        )

        for i, res in enumerate(retrieval_responses):
            _, adapt_response = adapt(
                cases,
                ordered_requests[i],
                res,
                config,
            )
            adaptation_responses.append(adapt_response)
            progress.update(adaptation_task, advance=1)

        adaptation_duration = timer() - adaptation_start

        # EVALUATION
        evaluation_start = timer()
        evaluation_responses: list[dict[str, AbstractEvaluation]] = []
        evaluation_task = progress.add_task(
            "Evaluating...", total=len(retrieval_responses)
        )

        if config.evaluation:
            for i, (retrieval_response, adaptation_response) in enumerate(
                zip(retrieval_responses, adaptation_responses)
            ):
                eval_map: dict[str, AbstractEvaluation] = {}

                if ranking := retrieval_response.semantic_ranking:
                    eval_map["mac"] = RetrievalEvaluation(
                        cases,
                        ordered_requests[i],
                        config.evaluation,
                        ranking,
                    )

                if ranking := retrieval_response.structural_ranking:
                    eval_map["fac"] = RetrievalEvaluation(
                        cases,
                        ordered_requests[i],
                        config.evaluation,
                        ranking,
                    )

                if adaptation_response.cases:
                    eval_map["adapt"] = AdaptationEvaluation(
                        cases,
                        ordered_requests[i],
                        config.evaluation,
                        adaptation_response.cases,
                    )

                evaluation_responses.append(eval_map)
                progress.update(evaluation_task, advance=1)

        evaluation_duration = timer() - evaluation_start

        # EXPORT
        export_task = progress.add_task("Exporting...", total=1)
        evals_aggregated = exporter.get_aggregated(
            evaluation_responses, config.evaluation
        )
        durations = {
            "retrieval": retrieval_duration,
            "adaptation": adaptation_duration,
            "evaluation": evaluation_duration,
        }
        progress.update(export_task, completed=1)

    print_json(
        exporter.get_json({"evaluations": evals_aggregated, "durations": durations})
    )

    eval_dump = {
        "durations": durations,
        "aggregated": evals_aggregated,
        "individual": [
            exporter.get_named_individual(eval) for eval in evaluation_responses
        ],
    }

    exporter.get_file(eval_dump, output_folder / "eval.json")
