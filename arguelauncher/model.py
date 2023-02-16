from dataclasses import dataclass, field

from arg_services.cbr.v1beta.model_pb2 import AnnotatedGraph


@dataclass
class CbrRun:
    query: AnnotatedGraph
    cases: dict[str, AnnotatedGraph]
    user_ranking: dict[str, int] = field(default_factory=dict)
    user_adaptations: dict[str, dict[str, str]] = field(default_factory=dict)
    retrieved_ranking: list[str] = field(default_factory=list)
    adapted_ranking: list[str] = field(default_factory=list)
    system_adaptations: dict[str, dict[str, str]] = field(default_factory=dict)
    system_adaptations: dict[str, dict[str, str]] = field(default_factory=dict)
