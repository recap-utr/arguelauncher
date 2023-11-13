from __future__ import annotations

import logging
from enum import Enum, auto

from arg_services.nlp.v1 import nlp_pb2

log = logging.getLogger(__name__)


class NlpConfig(Enum):
    DEFAULT = auto()
    STRF = auto()
    OPENAI = auto()


NLP_CONFIG: dict[NlpConfig, nlp_pb2.NlpConfig] = {
    NlpConfig.DEFAULT: nlp_pb2.NlpConfig(
        language="en",
        spacy_model="en_core_web_lg",
        similarity_method=nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE,
    ),
    NlpConfig.STRF: nlp_pb2.NlpConfig(
        language="en",
        similarity_method=nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE,
        embedding_models=[
            nlp_pb2.EmbeddingModel(
                model_type=nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SENTENCE_TRANSFORMERS,
                model_name="multi-qa-MiniLM-L6-cos-v1",
                pooling_type=nlp_pb2.Pooling.POOLING_MEAN,
            )
        ],
    ),
    NlpConfig.OPENAI: nlp_pb2.NlpConfig(
        language="en",
        similarity_method=nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE,
        embedding_models=[
            nlp_pb2.EmbeddingModel(
                model_type=nlp_pb2.EmbeddingType.EMBEDDING_TYPE_OPENAI,
                model_name="text-embedding-ada-002",
                pooling_type=nlp_pb2.Pooling.POOLING_MEAN,
            )
        ],
    ),
}
