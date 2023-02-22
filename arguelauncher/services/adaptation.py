from __future__ import annotations

import logging
import typing as t

import grpc
from arg_services.cbr.v1beta import adaptation_pb2, adaptation_pb2_grpc
from omegaconf import OmegaConf

from arguelauncher import model
from arguelauncher.config.arguegen import ExtrasConfig
from arguelauncher.config.cbr import CbrConfig
from arguelauncher.config.nlp import NLP_CONFIG

log = logging.getLogger(__name__)


def adapt(
    cases: t.Mapping[str, model.Graph],
    query: model.Graph,
    config: CbrConfig,
) -> tuple[adaptation_pb2.AdaptRequest, adaptation_pb2.AdaptResponse]:
    """Calculate similarity of queries and case base"""

    client = adaptation_pb2_grpc.AdaptationServiceStub(
        grpc.insecure_channel(config.adaptation.address)
    )
    rules_limit = config.adaptation.predefined_rules_limit

    # TODO: We only evaluate the first cbrEvaluation
    rules_per_case = query.userdata["cbrEvaluations"][0].get("generalizations")

    proto_cases: dict[str, adaptation_pb2.AdaptedCaseRequest] = {}

    for case_key, case in cases.items():
        proto_case = adaptation_pb2.AdaptedCaseRequest(
            case=case.to_annotated_graph(config.graph2text)
        )

        if rules_per_case and (case_rules := rules_per_case.get(case_key)):
            parsed_rules = list(model.parse_rules(case_rules))

            if rules_limit is not None:
                parsed_rules = parsed_rules[:rules_limit]

            proto_case.rules.extend(parsed_rules)

        proto_cases[case_key] = proto_case

    req = adaptation_pb2.AdaptRequest(
        cases=proto_cases,
        query=query.to_annotated_graph(config.graph2text),
        nlp_config=NLP_CONFIG[config.nlp_config],
    )
    req.extras.update(
        t.cast(ExtrasConfig, OmegaConf.to_object(config.adaptation.extras)).to_dict()
    )

    return req, client.Adapt(req)
