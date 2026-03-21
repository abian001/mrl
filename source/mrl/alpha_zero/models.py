from dataclasses import dataclass
from typing import Protocol
from abc import ABC, abstractmethod
from pathlib import Path
import math
import torch
import numpy as np
from mrl.alpha_zero.oracle import Oracle, LegalMask, Probabilities


class HasCore(Protocol):
    """Protocol for an observation"""

    @property
    @abstractmethod
    def core(self) -> np.ndarray:
        """Returns the core data of an observation"""


class SaveLoadModule(ABC, torch.nn.Module):

    def save(self, save_path: str | Path):
        path = save_path if isinstance(save_path, Path) else Path(save_path)

        if not path.is_absolute():
            path.parent.mkdir(parents = True, exist_ok = True)
        torch.save(self.state_dict(), path)

    def load(self, save_path: str | Path):
        self.load_state_dict(torch.load(save_path, weights_only = True))
        self.eval()


class OracleMixin(Oracle):

    torso: torch.nn.Module
    policy_head: torch.nn.Module
    value_head: torch.nn.Module

    def __init__(self, input_shape: tuple[int, ...], output_size: int):
        super().__init__()
        self.input_shape = input_shape
        self.input_size = math.prod(input_shape)
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (x.numel() // self.input_size,) + self.input_shape
        torso_out = self.torso(x.reshape(shape))
        policy_logits = self.policy_head(torso_out)
        value_out = self.value_head(torso_out)
        return value_out.squeeze(-1), policy_logits

    def get_value(self, observation: HasCore) -> float:
        with torch.inference_mode():
            x = self._make_input_tensor(observation)
            torso_out = self.torso(x)
            return self.value_head(torso_out).item()

    def get_probabilities(
        self,
        observation: HasCore,
        legal_mask: LegalMask | None = None
    ) -> Probabilities:
        with torch.inference_mode():
            x = self._make_input_tensor(observation)
            torso_out = self.torso(x)
            policy_logits = self.policy_head(torso_out)
            torch_legal_mask = self._get_legal_mask_tensor(legal_mask)
            masked_logits = torch.where(
                torch_legal_mask,
                policy_logits,
                torch.full_like(policy_logits, -1e32)
            )
            probabilities = torch.nn.functional.softmax(masked_logits, dim = -1)
            return probabilities.cpu().numpy()[0]

    def _make_input_tensor(self, observation: HasCore) -> torch.Tensor:
        shape = (1,) + self.input_shape
        return torch.as_tensor(
            observation.core,
            dtype = torch.float32,
            device = self._get_device()
        ).reshape(shape)

    def _get_legal_mask_tensor(self, legal_mask: LegalMask | None) -> torch.Tensor:
        if legal_mask is None:
            return torch.ones((self.output_size,), dtype = torch.bool, device = self._get_device())
        return torch.as_tensor(
            legal_mask,
            dtype = torch.bool,
            device = self._get_device()
        )

    def _get_device(self) -> torch.device:
        return next(self.torso.parameters()).device


@dataclass
class MLPCapacity:
    input_size: int
    output_size: int
    nn_width: int
    nn_depth: int


class OpenSpielMLP(OracleMixin, SaveLoadModule):

    def __init__(self, capacity: MLPCapacity):
        super().__init__((capacity.input_size,), capacity.output_size)
        # Torso layers
        layers: list[torch.nn.Module] = []
        for i in range(capacity.nn_depth):
            layers.append(torch.nn.Linear(
                capacity.input_size if i == 0 else capacity.nn_width, capacity.nn_width
            ))
            layers.append(torch.nn.ReLU())
        self.torso = torch.nn.Sequential(*layers)

        # Policy head
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(capacity.nn_width, capacity.nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(capacity.nn_width, capacity.output_size)
        )

        # Value head
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(capacity.nn_width, capacity.nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(capacity.nn_width, 1),
            torch.nn.Tanh()
        )


@dataclass
class CNNCapacity:
    input_shape: tuple[int, ...]
    output_size: int
    nn_width: int
    nn_depth: int


class OpenSpielConv(OracleMixin, SaveLoadModule):

    def __init__(self, capacity: CNNCapacity):
        super().__init__(capacity.input_shape, capacity.output_size)
        # Convert input shape to channels, height, width
        channels, height, width = capacity.input_shape

        # Torso: stacked Conv2D + BN + ReLU layers
        torso_layers = []
        for i in range(capacity.nn_depth):
            in_channels = channels if i == 0 else capacity.nn_width
            torso_layers += [
                torch.nn.Conv2d(in_channels, capacity.nn_width, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(capacity.nn_width),
                torch.nn.ReLU(inplace=True)
            ]
        self.torso = torch.nn.Sequential(*torso_layers)

        # Policy head: Conv2D -> BN -> ReLU -> Flatten -> Dense
        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(capacity.nn_width, 2, kernel_size=1),
            torch.nn.BatchNorm2d(2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * height * width, capacity.output_size)
        )

        # Value head: Conv2D -> BN -> ReLU -> Flatten -> Dense -> Tanh
        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(capacity.nn_width, 1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(1 * height * width, capacity.nn_width),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(capacity.nn_width, 1),
            torch.nn.Tanh()
        )


class ResidualBlock(torch.nn.Module):

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(channels),
        )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> np.ndarray:
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)


class OpenSpielResnet(OracleMixin, SaveLoadModule):

    def __init__(self, capacity: CNNCapacity):
        super().__init__(capacity.input_shape, capacity.output_size)
        channels, height, width = capacity.input_shape

        # Torso: Initial (Conv2d + BN + ReLU) + stacked residual blocks
        self.torso = torch.nn.Sequential(
            torch.nn.Conv2d(channels, capacity.nn_width, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(capacity.nn_width),
            torch.nn.ReLU(inplace=True),
            *[
                ResidualBlock(capacity.nn_width, kernel_size=3)
                for _ in range(capacity.nn_depth)
            ]
        )

        # Policy head
        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(capacity.nn_width, 2, kernel_size=1),
            torch.nn.BatchNorm2d(2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * height * width, capacity.output_size)
        )

        # Value head
        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(capacity.nn_width, 1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(1 * height * width, capacity.nn_width),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(capacity.nn_width, 1),
            torch.nn.Tanh()
        )
