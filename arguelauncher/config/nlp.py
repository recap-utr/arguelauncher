from __future__ import annotations

import logging
from enum import Enum, auto

from arg_services.nlp.v1 import nlp_pb2

log = logging.getLogger(__name__)


class NlpConfig(Enum):
    DEFAULT = auto()
    TRF = auto()
    SBERT = auto()


NLP_CONFIG: dict[NlpConfig, nlp_pb2.NlpConfig] = {
    NlpConfig.DEFAULT: nlp_pb2.NlpConfig(
        language="en",
        spacy_model="en_core_web_lg",
    ),
    NlpConfig.TRF: nlp_pb2.NlpConfig(
        language="en",
        spacy_model="en_core_web_trf",
    ),
    NlpConfig.SBERT: nlp_pb2.NlpConfig(
        language="en",
        embedding_models=[
            nlp_pb2.EmbeddingModel(
                model_type=nlp_pb2.EmbeddingType.EMBEDDING_TYPE_SENTENCE_TRANSFORMERS,
                model_name="stsb-mpnet-base-v2",
                pooling_type=nlp_pb2.Pooling.POOLING_MEAN,
            )
        ],
    ),
}
