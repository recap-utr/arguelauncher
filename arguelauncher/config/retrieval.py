from dataclasses import dataclass
from enum import Enum

from arg_services.cbr.v1beta.retrieval_pb2 import MappingAlgorithm as PbMappingAlgorithm
from arg_services.cbr.v1beta.retrieval_pb2 import SchemeHandling as PbSchemeHandling
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


@dataclass
class RetrievalConfig(DataClassDictMixin):
    address: str = "127.0.0.1:50200"
    scheme_handling: RetrievalSchemeHandling = RetrievalSchemeHandling.BINARY
    mapping_algorithm: RetrievalMappingAlgorithm = RetrievalMappingAlgorithm.ASTAR_1
    mac: bool = True
    fac: bool = True
    limit: int = 10
    astar_queue_limit: int = 1000
