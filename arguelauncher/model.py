from __future__ import annotations

import typing as t

import arguebuf
import lemminflect
from arg_services.cbr.v1beta import adaptation_pb2
from arg_services.cbr.v1beta.adaptation_pb2 import Pos
from arg_services.cbr.v1beta.model_pb2 import AnnotatedGraph
from typing_extensions import Required, TypedDict

from arguelauncher.algorithms.graph2text import Graph2TextAlgorithm, graph2text

ADDITIONAL_INFLECTIONS: dict[str, dict[str, list[str]]] = {
    "prove": {"VPN": ["proven"]},
    "journey": {"NN": ["journeying"]},
    "relinquish": {"NN": ["relinquishing"]},
    "impedimentum": {"NNS": ["impedimenta"]},
    "be": {"VBZ": ["'s"]},
}


def _lemma_parts(text: str, pos: str) -> t.List[str]:
    tokens: t.List[str] = text.split()

    *parts, tail = tokens
    tail_lemmas = t.cast(tuple[str, ...], lemminflect.getLemma(tail, pos))

    if len(tail_lemmas) > 0:
        parts.append(tail_lemmas[0])

    return parts


def inflect(text: str, pos: str, lemmatize: bool) -> str:
    """Return the lemma of `text` and all inflected forms of `text`."""

    lemma_parts = _lemma_parts(text, pos) if lemmatize else text.split()

    return " ".join(lemma_parts)


pos2proto: dict[str, Pos.ValueType] = {
    "noun": Pos.POS_NOUN,
    "verb": Pos.POS_VERB,
    "adjective": Pos.POS_ADJECTIVE,
    "adverb": Pos.POS_ADVERB,
}

pos2spacy: dict[Pos.ValueType, str] = {
    Pos.POS_NOUN: "NOUN",
    Pos.POS_VERB: "VERB",
    Pos.POS_ADJECTIVE: "ADJ",
    Pos.POS_ADVERB: "ADV",
}


def str2concept(text: str) -> adaptation_pb2.Concept:
    text = text.strip().lower()
    raw_lemma, raw_pos = text.split("/")

    pos = pos2proto[raw_pos]
    lemma = inflect(raw_lemma, pos2spacy[pos], lemmatize=True)

    return adaptation_pb2.Concept(lemma=lemma, pos=pos)


class AdaptationRule(TypedDict):
    source: str
    target: str


AdaptationRules = list[AdaptationRule]


class CbrEvaluation(TypedDict, total=False):
    ranking: Required[dict[str, int]]
    generalizations: dict[str, AdaptationRules]
    specializations: dict[str, AdaptationRules]
    name: str


class Userdata(TypedDict):
    cbrEvaluations: list[CbrEvaluation]


def parse_rules(rules: AdaptationRules) -> list[adaptation_pb2.Rule]:
    return [
        adaptation_pb2.Rule(
            source=str2concept(rule["source"]), target=str2concept(rule["target"])
        )
        for rule in rules
    ]


class Graph(arguebuf.Graph):
    userdata: Userdata

    def to_protobuf(self, text_algorithm: Graph2TextAlgorithm) -> AnnotatedGraph:
        g = AnnotatedGraph(
            graph=arguebuf.dump.protobuf(self), text=graph2text(self, text_algorithm)
        )
        g.graph.userdata.Clear()

        return g

    @classmethod
    def from_protobuf(
        cls, obj: AnnotatedGraph, userdata: t.Optional[Userdata]
    ) -> Graph:
        g = t.cast(
            Graph,
            arguebuf.load.protobuf(
                obj.graph, config=arguebuf.load.Config(GraphClass=cls)
            ),
        )

        if userdata is not None:
            g.userdata = userdata

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
