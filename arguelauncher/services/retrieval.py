from __future__ import annotations

import logging
import typing as t

import grpc
from arg_services.cbr.v1beta import retrieval_pb2, retrieval_pb2_grpc

from arguelauncher import model
from arguelauncher.config.cbr import CbrConfig
from arguelauncher.config.nlp import NLP_CONFIG

log = logging.getLogger(__name__)


def retrieve(
    cases: t.Mapping[str, model.Graph],
    queries: t.Sequence[model.Graph],
    config: CbrConfig,
) -> tuple[retrieval_pb2.RetrieveRequest, retrieval_pb2.RetrieveResponse]:
    """Calculate similarity of queries and case base"""

    req = retrieval_pb2.RetrieveRequest(
        cases={
            key: value.to_annotated_graph(config.graph2text)
            for key, value in cases.items()
        },
        queries=[value.to_annotated_graph(config.graph2text) for value in queries],
        nlp_config=NLP_CONFIG[config.nlp_config],
    )

    if config.retrieval is None:
        return req, retrieval_pb2.RetrieveResponse(
            query_responses=[retrieval_pb2.QueryResponse() for _ in queries]
        )

    client = retrieval_pb2_grpc.RetrievalServiceStub(
        grpc.insecure_channel(config.retrieval.address)
    )

    req.limit = config.retrieval.limit
    req.semantic_retrieval = config.retrieval.mac
    req.structural_retrieval = config.retrieval.fac
    req.scheme_handling = config.retrieval.scheme_handling.value
    req.mapping_algorithm = config.retrieval.mapping_algorithm.value[0]
    req.mapping_algorithm_variant = config.retrieval.mapping_algorithm.value[1]

    return req, client.Retrieve(req)
