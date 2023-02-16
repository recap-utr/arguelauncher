from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from arg_services.cbr.v1beta.retrieval_pb2 import MappingAlgorithm as PbMappingAlgorithm
from arg_services.cbr.v1beta.retrieval_pb2 import SchemeHandling as PbSchemeHandling
from hydra.core.config_store import ConfigStore
from mashumaro import DataClassDictMixin


class RetrievalMappingAlgorithm(Enum):
    ASTAR_1 = (PbMappingAlgorithm.MAPPING_ALGORITHM_ASTAR, 1)
    ASTAR_2 = (PbMappingAlgorithm.MAPPING_ALGORITHM_ASTAR, 2)
    ASTAR_3 = (PbMappingAlgorithm.MAPPING_ALGORITHM_ASTAR, 3)
    GREEDY_1 = (PbMappingAlgorithm.MAPPING_ALGORITHM_GREEDY, 1)
    GREEDY_2 = (PbMappingAlgorithm.MAPPING_ALGORITHM_GREEDY, 2)
    ISOMORPHISM_1 = (PbMappingAlgorithm.MAPPING_ALGORITHM_ISOMORPHISM, 1)


class RetrievalSchemeHandling(Enum):
    UNSPECIFIED = PbSchemeHandling.SCHEME_HANDLING_UNSPECIFIED
    BINARY = PbSchemeHandling.SCHEME_HANDLING_BINARY
    TAXONOMY = PbSchemeHandling.SCHEME_HANDLING_TAXONOMY


class RetrievalGraph2TextAlgorithm(Enum):
    DFS = auto()
    DFS_RECONSTRUCTION = auto()
    BFS = auto()
    RANDOM = auto()
    ORIGINAL_RESOURCE = auto()
    NODE_ID = auto()


@dataclass
class PathConfig(DataClassDictMixin):
    cases: Path = Path("data/cases/english/microtexts-plain")
    case_graphs_pattern: str = "*.json"
    queries: Path = Path("data/queries/english/microtexts-plain/all")
    query_graphs_pattern: str = "*.json"
    query_texts_pattern: str = "*.json"
    user_ranking: Path = Path("data/benchmark/microtexts")


@dataclass
class RetrievalRequestConfig(DataClassDictMixin):
    nlp_config: str = "default"
    scheme_handling: RetrievalSchemeHandling = RetrievalSchemeHandling.BINARY
    mapping_algorithm: RetrievalMappingAlgorithm = RetrievalMappingAlgorithm.ASTAR_1
    graph2text_algorithm: RetrievalGraph2TextAlgorithm = (
        RetrievalGraph2TextAlgorithm.DFS
    )
    mac: bool = True
    fac: bool = False
    limit: int = 10
    language: str = "en"


@dataclass
class RetrievalEvaluationConfig(DataClassDictMixin):
    individual_results: bool = False
    aggregated_results: bool = True
    max_user_rank: int = 3
    f_scores: list[float] = field(default_factory=lambda: [1, 2])


@dataclass
class RetrievalConfig(DataClassDictMixin):
    retrieval_address: str = "127.0.0.1:6789"
    path: PathConfig = field(default_factory=PathConfig)
    request: RetrievalRequestConfig = field(default_factory=RetrievalRequestConfig)
    evaluation: RetrievalEvaluationConfig = field(
        default_factory=RetrievalEvaluationConfig
    )


@dataclass
class AdaptationScoreConfig(DataClassDictMixin):
    adus_semantic_similarity: float = 0.0


@dataclass
class AdaptationRequestConfig(DataClassDictMixin):
    score: AdaptationScoreConfig = field(default_factory=AdaptationScoreConfig)


@dataclass
class AdaptationConfig(DataClassDictMixin):
    adaptation_address: str = ""
    predefined_rules_limit: int = 0  # if 0, no rule will be sent at all
    request: AdaptationRequestConfig = field(default_factory=AdaptationRequestConfig)


cs = ConfigStore.instance()
cs.store(name="retreival_config", node=RetrievalConfig)
cs.store(name="adaptation_config", node=AdaptationConfig)
