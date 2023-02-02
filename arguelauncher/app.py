from __future__ import annotations

import json
import logging
import typing as t
from pathlib import Path
from timeit import default_timer as timer
from typing import List

import arguebuf as ag
import grpc
import hydra
from arg_services.cbr.v1beta import retrieval_pb2, retrieval_pb2_grpc
from arg_services.cbr.v1beta.model_pb2 import AnnotatedGraph
from arg_services.nlp.v1 import nlp_pb2
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from rich import print_json

from arguelauncher.algorithms.graph2text import graph2text
from arguelauncher.config import Config
from arguelauncher.services import exporter
from arguelauncher.services.evaluation import Evaluation

log = logging.getLogger(__name__)

_nlp_configs = {
    "default": nlp_pb2.NlpConfig(
        spacy_model="en_core_web_lg",
        similarity_method=nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE,
    ),
    "trf": nlp_pb2.NlpConfig(
        language="en",
        spacy_model="en_core_web_trf",
    ),
    "sbert": nlp_pb2.NlpConfig(
        language="en",
        embedding_models=[
            nlp_pb2.EmbeddingModel(
                model_type=nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SENTENCE_TRANSFORMERS,
                model_name="stsb-mpnet-base-v2",
                pooling_type=nlp_pb2.Pooling.POOLING_MEAN,
            )
        ],
    ),
}


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: Config) -> None:
    """Calculate similarity of queries and case base"""
    output_folder = Path(HydraConfig.get().runtime.output_dir)

    start_time = 0
    duration = 0
    eval_dict = {}
    evaluations: List[t.Optional[Evaluation]] = []

    client = retrieval_pb2_grpc.RetrievalServiceStub(
        grpc.insecure_channel(config.retrieval_address)
    )

    cases: t.Dict[Path, ag.Graph] = {
        file: ag.from_file(file)
        for file in Path(config.path.cases).glob(config.path.case_graphs_pattern)
    }
    arguebuf_cases = {
        str(key.relative_to(config.path.cases)): graph for key, graph in cases.items()
    }
    protobuf_cases = {
        key: AnnotatedGraph(
            graph=ag.to_protobuf(graph),
            text=graph2text(graph, config.request.graph2text_algorithm),
        )
        for key, graph in arguebuf_cases.items()
    }

    queries: t.Dict[Path, ag.Graph] = {
        file: ag.from_file(file)
        for file in Path(config.path.queries).glob(config.path.query_graphs_pattern)
    }
    for file in Path(config.path.queries).glob(config.path.query_texts_pattern):
        with file.open("r", encoding="utf-8") as f:
            text = f.read()
            g = ag.Graph()
            g.add_node(ag.AtomNode(text))
            g.add_resource(ag.Resource(text))
            queries[file] = g
    query_files = [file for file in queries]

    protobuf_queries = [
        AnnotatedGraph(
            graph=ag.to_protobuf(query),
            text=graph2text(query, config.request.graph2text_algorithm),
        )
        for query in queries.values()
    ]

    nlp_config = _nlp_configs[config.request.nlp_config]
    nlp_config.language = config.request.language

    start_time = timer()

    req = retrieval_pb2.RetrieveRequest(
        cases=protobuf_cases,
        queries=protobuf_queries,
        limit=config.request.limit,
        semantic_retrieval=config.request.mac,
        structural_retrieval=config.request.fac,
        nlp_config=nlp_config,
        scheme_handling=config.request.scheme_handling.value,
        mapping_algorithm=config.request.mapping_algorithm.value[0],
        mapping_algorithm_variant=config.request.mapping_algorithm.value[1],
    )

    res_wrapper: retrieval_pb2.RetrieveResponse = client.Retrieve(req)

    for i, res in enumerate(res_wrapper.query_responses):
        evaluation = None
        mac_export = None
        fac_export = None

        if mac_results := res.semantic_ranking:
            mac_export = exporter.get_results(mac_results)
            evaluation = Evaluation(
                cases,
                mac_results,
                query_files[i],
                config.path,
                config.evaluation,
            )

        if fac_results := res.structural_ranking:
            fac_export = exporter.get_results(fac_results)
            evaluation = Evaluation(
                cases,
                fac_results,
                query_files[i],
                config.path,
                config.evaluation,
            )

        evaluations.append(evaluation)

        if config.evaluation.individual_results:
            exporter.export_results(
                query_files[i],
                mac_export,
                fac_export,
                evaluation,
                config.path,
                output_folder,
            )
            log.info("Individual Results were exported.")

    duration = timer() - start_time
    eval_dict = exporter.get_results_aggregated(evaluations)

    print_json(json.dumps(eval_dict))

    if config.evaluation.aggregated_results:
        exporter.export_results_aggregated(
            eval_dict,
            duration,
            t.cast(Config, OmegaConf.to_object(config)).to_dict(),
            config.path,
            output_folder,
        )
        log.info("Aggregated Results were exported.")
