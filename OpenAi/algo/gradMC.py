import torch
import torch.nn as nn
import torch.nn.functional as F


from torch import Tensor
import typing
# create the network 


class QPolicy(nn.Module):

    def __init__(self, n_states: int, n_actions: int) -> None:
        super().__init__()
        self.e = nn.Sequential(
            nn.Linear(n_states, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64,32),
            nn.Tanh(),
            nn.Linear(32, 4)
        )

    def forward(self, x: Tensor) -> Tensor:

        output = self.e(x)
        return output



