from __future__ import annotations

import logging
import typing as t
from pathlib import Path
from timeit import default_timer as timer

import arguebuf as ag
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from rich import print_json
from rich.progress import track

from arguelauncher import model
from arguelauncher.config.cbr import CbrConfig
from arguelauncher.services import exporter
from arguelauncher.services.adaptation import adapt
from arguelauncher.services.evaluation import (
    AdaptationEvaluation,
    BaseEvaluation,
    RetrievalEvaluation,
)
from arguelauncher.services.retrieval import retrieve

log = logging.getLogger(__name__)


cs = ConfigStore.instance()
cs.store(name="cbr", node=CbrConfig)


@hydra.main(version_base=None, config_path="config", config_name="cbr_run")
def main(config: CbrConfig) -> None:
    """Calculate similarity of queries and case base"""
    output_folder = Path(HydraConfig.get().runtime.output_dir)

    start_time = 0
    duration = 0

    cases = {
        str(file.relative_to(config.path.cases)): t.cast(
            model.Graph,
            ag.load.file(file, config=ag.load.Config(GraphClass=model.Graph)),
        )
        for file in Path(config.path.cases).glob(config.path.cases_pattern)
    }
    requests = {
        str(file.relative_to(config.path.requests)): t.cast(
            model.Graph,
            ag.load.file(file, config=ag.load.Config(GraphClass=model.Graph)),
        )
        for file in Path(config.path.requests).glob(config.path.requests_pattern)
    }
    ordered_requests = list(requests.values())

    start_time = timer()

    log.info("Retrieving...")
    _, retrieve_response = retrieve(cases, ordered_requests, config)

    log.info("Processing results...")
    evals: list[dict[str, BaseEvaluation]] = []

    for i, res in track(list(enumerate(retrieve_response.query_responses))):
        eval_map: dict[str, BaseEvaluation] = {}

        if ranking := res.semantic_ranking:
            eval_map["mac"] = RetrievalEvaluation(
                cases,
                ordered_requests[i],
                config.evaluation,
                ranking,
            )

        if ranking := res.structural_ranking:
            eval_map["fac"] = RetrievalEvaluation(
                cases,
                ordered_requests[i],
                config.evaluation,
                ranking,
            )

        _, adapt_response = adapt(cases, ordered_requests[i], config)
        eval_map["adapt"] = AdaptationEvaluation(
            cases,
            ordered_requests[i],
            config.evaluation,
            {key: case for key, case in zip(cases.keys(), adapt_response.cases)},
        )
        # TODO: Save adapted graphs

        evals.append(eval_map)

    duration = timer() - start_time
    config_dump = (t.cast(CbrConfig, OmegaConf.to_object(config)).to_dict(),)
    aggregated_eval = exporter.get_aggregated(evals)

    print_json(exporter.get_json(aggregated_eval))

    eval_dump = {
        "config": config_dump,
        "duration": duration,
        "aggregated": aggregated_eval,
        "individual": [exporter.get_named_individual(eval) for eval in evals],
    }

    exporter.get_file(eval_dump, output_folder / "eval.json")
