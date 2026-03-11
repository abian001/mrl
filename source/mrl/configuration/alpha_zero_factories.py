from typing import Protocol
from abc import abstractmethod
from mrl.alpha_zero.alpha_zero import AlphaZero
from mrl.alpha_zero.distributed_alpha_zero import DistributedAlphaZero
from mrl.configuration.alpha_zero_configuration import UnionAlphaZeroConfiguration


class HasAlphaZero(Protocol):

    @property
    @abstractmethod
    def alpha_zero(self) -> UnionAlphaZeroConfiguration:
        """Returns alpha zero configuration object."""


def make_alpha_zero(config: HasAlphaZero) -> AlphaZero | DistributedAlphaZero:
    if config.alpha_zero.type == "InMemory":
        return AlphaZero(config.alpha_zero)
    assert config.alpha_zero.type == "HDF5"
    return DistributedAlphaZero(config.alpha_zero)
