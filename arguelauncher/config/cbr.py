import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from mashumaro import DataClassDictMixin

from arguelauncher.config.adaptation import AdaptationConfig
from arguelauncher.config.nlp import NlpConfig
from arguelauncher.config.retrieval import RetrievalConfig
from arguelauncher.model import Graph2TextAlgorithm


@dataclass
class PathConfig(DataClassDictMixin):
    cases: Path = Path("data/cases/microtexts")
    cases_pattern: str = "*.json"
    requests: Path = Path("data/requests/microtexts-retrieval-complex")
    requests_pattern: str = "**/*.json"


@dataclass
class EvaluationConfig(DataClassDictMixin):
    relevance_levels: int = 3
    export_graph: bool = True


@dataclass
class CbrConfig(DataClassDictMixin):
    nlp_config: NlpConfig = NlpConfig.DEFAULT
    graph2text: Graph2TextAlgorithm = Graph2TextAlgorithm.NODE_ID
    path: PathConfig = field(default_factory=PathConfig)
    retrieval: t.Optional[RetrievalConfig] = field(default_factory=RetrievalConfig)
    adaptation: t.Optional[AdaptationConfig] = field(default_factory=AdaptationConfig)
    evaluation: t.Optional[EvaluationConfig] = field(default_factory=EvaluationConfig)
