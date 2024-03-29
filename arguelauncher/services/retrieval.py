import logging
import typing as t

from arg_services.cbr.v1beta import retrieval_pb2, retrieval_pb2_grpc

from arguelauncher import model
from arguelauncher.config.cbr import CbrConfig
from arguelauncher.config.nlp import NLP_CONFIG

log = logging.getLogger(__name__)


def retrieve(
    client: t.Optional[retrieval_pb2_grpc.RetrievalServiceStub],
    cases: t.Mapping[str, model.Graph],
    queries: t.Sequence[model.Graph],
    config: CbrConfig,
) -> tuple[retrieval_pb2.RetrieveRequest, retrieval_pb2.RetrieveResponse]:
    """Calculate similarity of queries and case base"""

    req = retrieval_pb2.RetrieveRequest(
        cases={
            key: value.to_protobuf(config.graph2text) for key, value in cases.items()
        },
        queries=[value.to_protobuf(config.graph2text) for value in queries],
        nlp_config=NLP_CONFIG[config.nlp_config],
    )

    if (
        config.retrieval is None
        or client is None
        or (not config.retrieval.mac and not config.retrieval.fac)
    ):
        return req, retrieval_pb2.RetrieveResponse(
            query_responses=[retrieval_pb2.QueryResponse() for _ in queries]
        )

    req.limit = config.retrieval.limit
    req.semantic_retrieval = config.retrieval.mac
    req.structural_retrieval = config.retrieval.fac
    req.scheme_handling = config.retrieval.scheme_handling.value
    req.mapping_algorithm = config.retrieval.mapping_algorithm.value[0]
    req.mapping_algorithm_variant = config.retrieval.mapping_algorithm.value[1]
    req.extras.update({"astar_queue_limit": config.retrieval.astar_queue_limit})

    return req, client.Retrieve(req)
