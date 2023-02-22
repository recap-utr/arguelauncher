from __future__ import annotations

import typing as t

import arguebuf
from arg_services.cbr.v1beta import adaptation_pb2
from arg_services.cbr.v1beta.model_pb2 import AnnotatedGraph
from typing_extensions import Required, TypedDict

from arguelauncher.algorithms.graph2text import Graph2TextAlgorithm, graph2text

pos2proto = {
    "noun": adaptation_pb2.Pos.POS_NOUN,
    "verb": adaptation_pb2.Pos.POS_VERB,
    "adjective": adaptation_pb2.Pos.POS_ADJECTIVE,
    "adverb": adaptation_pb2.Pos.POS_ADVERB,
}

AdaptationRules = dict[str, str]


def str2concept(text: str) -> adaptation_pb2.Concept:
    concept = text.strip().lower()
    lemma, pos = concept.split("/")

    return adaptation_pb2.Concept(lemma=lemma, pos=pos2proto[pos])


def parse_rules(rules: AdaptationRules) -> t.Iterator[adaptation_pb2.Rule]:
    for source, target in rules.items():
        yield adaptation_pb2.Rule(
            source=str2concept(source), target=str2concept(target)
        )


class CbrEvaluation(TypedDict, total=False):
    ranking: Required[dict[str, int]]
    generalizations: dict[str, AdaptationRules]
    specializations: dict[str, AdaptationRules]
    name: str


class Userdata(TypedDict):
    cbrEvaluations: list[CbrEvaluation]


class Graph(arguebuf.Graph):
    userdata: Userdata

    def to_annotated_graph(self, text_algorithm: Graph2TextAlgorithm) -> AnnotatedGraph:
        g = AnnotatedGraph(
            graph=arguebuf.dump.protobuf(self), text=graph2text(self, text_algorithm)
        )
        g.graph.userdata.Clear()

        return g


# @dataclass
# class CbrRun:
#     query: AnnotatedGraph
#     cases: dict[str, AnnotatedGraph]
#     user_ranking: dict[str, int] = field(default_factory=dict)
#     user_adaptations: dict[str, dict[str, str]] = field(default_factory=dict)
#     retrieved_ranking: list[str] = field(default_factory=list)
#     adapted_ranking: list[str] = field(default_factory=list)
#     system_adaptations: dict[str, dict[str, str]] = field(default_factory=dict)
