from hydra.core.config_store import ConfigStore

from arguelauncher.config.adaptation import AdaptationConfig
from arguelauncher.config.cbr import CbrConfig
from arguelauncher.config.retrieval import RetrievalConfig

cs = ConfigStore.instance()
cs.store(name="retrieval", node=RetrievalConfig)
cs.store(name="adaptation", node=AdaptationConfig)
cs.store(name="cbr", node=CbrConfig)
