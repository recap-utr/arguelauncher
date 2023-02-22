import typing as t
from dataclasses import dataclass, field

from mashumaro import DataClassDictMixin

from arguelauncher.config.arguegen import ExtrasConfig


@dataclass
class AdaptationConfig(DataClassDictMixin):
    address: str = "127.0.0.1:50300"
    predefined_rules_limit: t.Optional[int] = 0  # if 0, no rule will be sent at all
    extras: ExtrasConfig = field(default_factory=ExtrasConfig)