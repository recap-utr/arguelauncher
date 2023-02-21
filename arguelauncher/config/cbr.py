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
    cases_pattern: str = "*.xml"
    requests: Path = Path("data/requests/microtexts-retrieval-complex")
    requests_pattern: str = "*.json"


@dataclass
class EvaluationConfig(DataClassDictMixin):
    max_user_rank: int = 3
    f_scores: list[float] = field(default_factory=lambda: [1, 2])


@dataclass
class CbrConfig(DataClassDictMixin):
    nlp_config: NlpConfig = NlpConfig.DEFAULT
    graph2text: Graph2TextAlgorithm = Graph2TextAlgorithm.NODE_ID
    path: PathConfig = field(default_factory=PathConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
