from __future__ import annotations

import logging
import typing as t

import grpc
from arg_services.cbr.v1beta import retrieval_pb2, retrieval_pb2_grpc

from arguelauncher import model
from arguelauncher.config import CbrConfig
from arguelauncher.config.nlp import NLP_CONFIG

log = logging.getLogger(__name__)


def retrieve(
    cases: t.Mapping[str, model.Graph],
    queries: t.Sequence[model.Graph],
    config: CbrConfig,
) -> tuple[retrieval_pb2.RetrieveRequest, retrieval_pb2.RetrieveResponse]:
    """Calculate similarity of queries and case base"""

    client = retrieval_pb2_grpc.RetrievalServiceStub(
        grpc.insecure_channel(config.retrieval.address)
    )

    req = retrieval_pb2.RetrieveRequest(
        cases={
            key: value.to_annotated_graph(config.graph2text)
            for key, value in cases.items()
        },
        queries=[value.to_annotated_graph(config.graph2text) for value in queries],
        limit=config.retrieval.limit,
        semantic_retrieval=config.retrieval.mac,
        structural_retrieval=config.retrieval.fac,
        nlp_config=NLP_CONFIG[config.nlp_config],
        scheme_handling=config.retrieval.scheme_handling.value,
        mapping_algorithm=config.retrieval.mapping_algorithm.value[0],
        mapping_algorithm_variant=config.retrieval.mapping_algorithm.value[1],
    )

    return req, client.Retrieve(req)
