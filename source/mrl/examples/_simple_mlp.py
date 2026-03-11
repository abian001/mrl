from dataclasses import dataclass
import numpy as np
import torch
from mrl.alpha_zero.models import SaveLoadModule, OracleMixin


class SimpleMLP(OracleMixin, SaveLoadModule):

    def __init__(self, input_size: int, output_size: int):
        super().__init__((input_size,), output_size)
        self.torso = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size),
            torch.nn.ReLU()
        )
        self.policy_head = torch.nn.Linear(output_size, output_size)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(output_size, 1),
            torch.nn.Tanh()
        )
