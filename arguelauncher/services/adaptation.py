from __future__ import annotations

import logging
import typing as t

from arg_services.cbr.v1beta import adaptation_pb2, adaptation_pb2_grpc, retrieval_pb2
from mashumaro import DataClassDictMixin
from omegaconf import OmegaConf

from arguelauncher import model
from arguelauncher.config.adaptation import AdaptationConfig
from arguelauncher.config.cbr import CbrConfig
from arguelauncher.config.nlp import NLP_CONFIG

log = logging.getLogger(__name__)


def build_case_request(
    case_key: str,
    cases: t.Mapping[str, model.Graph],
    rules_per_case: t.Optional[dict[str, dict[str, str]]],
    config: CbrConfig,
) -> adaptation_pb2.AdaptedCaseRequest:
    assert config.adaptation is not None
    case = cases[case_key]

    rules_limit = config.adaptation.predefined_rules_limit
    proto_case = adaptation_pb2.AdaptedCaseRequest(
        case=case.to_protobuf(config.graph2text)
    )

    if rules_per_case and (case_rules := rules_per_case.get(case_key)):
        parsed_rules = list(model.parse_rules(case_rules))

        if rules_limit is not None:
            parsed_rules = parsed_rules[:rules_limit]

        proto_case.rules.extend(parsed_rules)

    return proto_case


def adapt(
    client: t.Optional[adaptation_pb2_grpc.AdaptationServiceStub],
    cases: t.Mapping[str, model.Graph],
    query: model.Graph,
    retrieval: t.Optional[retrieval_pb2.QueryResponse],
    config: CbrConfig,
) -> tuple[adaptation_pb2.AdaptRequest, adaptation_pb2.AdaptResponse]:
    """Calculate similarity of queries and case base"""

    if config.adaptation is None or client is None:
        return adaptation_pb2.AdaptRequest(), adaptation_pb2.AdaptResponse()

    ranking = (
        (retrieval.structural_ranking or retrieval.semantic_ranking)
        if retrieval
        else None
    )

    # TODO: We only evaluate the first cbrEvaluation
    rules_per_case = query.userdata["cbrEvaluations"][0].get("generalizations")

    # If there is a retrieval ranking, we only adapt the cases in the ranking
    # Otherwise, we adapt all cases that have benchmark rules
    proto_cases: dict[str, adaptation_pb2.AdaptedCaseRequest] = {}

    if ranking:
        proto_cases = {
            ranked_case.id: build_case_request(
                ranked_case.id, cases, rules_per_case, config
            )
            for ranked_case in ranking
        }
    elif rules_per_case:
        proto_cases = {
            case_key: build_case_request(case_key, cases, rules_per_case, config)
            for case_key in rules_per_case.keys()
        }

    req = adaptation_pb2.AdaptRequest(
        cases=proto_cases,
        query=query.to_protobuf(config.graph2text),
        nlp_config=NLP_CONFIG[config.nlp_config],
    )
    req.extras.update(_get_extras(config.adaptation).to_dict())

    return req, client.Adapt(req)


def _get_extras(config: AdaptationConfig) -> DataClassDictMixin:
    return t.cast(DataClassDictMixin, OmegaConf.to_object(config.extras))
