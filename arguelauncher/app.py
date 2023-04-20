from __future__ import annotations

import logging
import random
import typing as t
from pathlib import Path
from timeit import default_timer as timer

import arguebuf as ag
import grpc
import hydra
from arg_services.cbr.v1beta import adaptation_pb2_grpc, retrieval_pb2_grpc
from arg_services.cbr.v1beta.adaptation_pb2 import AdaptResponse
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from rich import print, print_json

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


def randomize_grpc_address(address: str) -> str:
    """Randomize the order of the grpc addresses to speed up initial load balancing"""

    if address.startswith("ipv4:"):
        addr = address.removeprefix("ipv4:")
        hosts = addr.split(",")
        random.shuffle(hosts)
        random_addr = ",".join(hosts)

        return f"ipv4:{random_addr}"

    return address


def grpc_channel(address: str) -> grpc.Channel:
    return grpc.insecure_channel(
        randomize_grpc_address(address),
        [
            ("grpc.lb_policy_name", "round_robin"),
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )


log = logging.getLogger(__name__)


cs = ConfigStore.instance()
cs.store(name="cbr", node=CbrConfig)


@hydra.main(version_base=None, config_path="config", config_name="app")
def main(config: CbrConfig) -> None:
    """Calculate similarity of queries and case base"""

    adaptation_client = (
        adaptation_pb2_grpc.AdaptationServiceStub(
            grpc_channel(config.adaptation.address)
        )
        if config.adaptation
        else None
    )

    retrieval_client = (
        retrieval_pb2_grpc.RetrievalServiceStub(grpc_channel(config.retrieval.address))
        if config.retrieval
        else None
    )

    assert retrieval_client

    log.info("Initializing...")
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
    request_keys = list(requests.keys())

    # RETRIEVAL
    log.info("Retrieving...")
    retrieval_start = timer()
    _, retrieve_response = retrieve(retrieval_client, cases, ordered_requests, config)
    retrieval_responses = list(retrieve_response.query_responses)

    retrieval_duration = timer() - retrieval_start

    # ADAPTATION
    adaptation_start = timer()
    adaptation_responses: list[AdaptResponse] = []
    log.info("Adapting...")

    for i, res in enumerate(retrieval_responses):
        log.debug(
            f"Adapting query {request_keys[i]} ({i+1}/{len(retrieval_responses)})..."
        )
        _, adapt_response = adapt(
            adaptation_client,
            cases,
            ordered_requests[i],
            res,
            config,
        )
        adaptation_responses.append(adapt_response)

    adaptation_duration = timer() - adaptation_start

    # EVALUATION
    evaluation_start = timer()
    evaluation_responses: dict[str, dict[str, AbstractEvaluation]] = {}
    retrieval_limit = config.retrieval.limit if config.retrieval else None
    log.info("Evaluating...")

    if config.evaluation:
        for i, (retrieval_response, adaptation_response) in enumerate(
            zip(retrieval_responses, adaptation_responses)
        ):
            log.debug(
                "Evaluating query"
                f" {request_keys[i]} ({i+1}/{len(retrieval_responses)})..."
            )
            current_request = ordered_requests[i]

            eval_map: dict[str, AbstractEvaluation] = {}

            if ranking := retrieval_response.semantic_ranking:
                eval_map["mac"] = RetrievalEvaluation(
                    cases, current_request, config.evaluation, ranking, retrieval_limit
                )

            if ranking := retrieval_response.structural_ranking:
                eval_map["fac"] = RetrievalEvaluation(
                    cases, current_request, config.evaluation, ranking, retrieval_limit
                )

            if adaptation_response.cases:
                eval_map["adaptation"] = AdaptationEvaluation(
                    cases,
                    current_request,
                    config.evaluation,
                    adaptation_response.cases,
                    retrieval_client,
                    config,
                )

                cases_adapted = {
                    name: model.Graph.from_protobuf(res.case, cases[name].userdata)
                    for name, res in adaptation_response.cases.items()
                }

                _, retrieve_responses_adapted = retrieve(
                    retrieval_client,
                    cases_adapted,
                    [current_request],
                    config,
                )

                retrieve_response_adapted = retrieve_responses_adapted.query_responses[
                    0
                ]

                if ranking := retrieve_response_adapted.semantic_ranking:
                    eval_map["mac_adapted"] = RetrievalEvaluation(
                        cases,
                        current_request,
                        config.evaluation,
                        ranking,
                        retrieval_limit,
                    )

                if ranking := retrieve_response_adapted.structural_ranking:
                    eval_map["fac_adapted"] = RetrievalEvaluation(
                        cases,
                        current_request,
                        config.evaluation,
                        ranking,
                        retrieval_limit,
                    )

            evaluation_responses[request_keys[i]] = eval_map

    evaluation_duration = timer() - evaluation_start

    # EXPORT
    log.info("Exporting...")
    eval_export = exporter.get_aggregated(evaluation_responses, config.evaluation)

    durations = {
        "retrieval": retrieval_duration,
        "adaptation": adaptation_duration,
        "evaluation": evaluation_duration,
    }

    print("Durations:")
    print_json(exporter.get_json(durations))

    eval_df = exporter.get_dataframe(eval_export, output_folder / "eval.csv")
    print(exporter.df_to_table(eval_df))

    eval_dump = {
        "durations": durations,
        "aggregated": eval_export,
        "individual": {
            key: exporter.get_named_individual(value)
            for key, value in evaluation_responses.items()
        },
    }

    exporter.get_file(eval_dump, output_folder / "eval.json")
