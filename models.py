from typing import Tuple

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Conv DQN for Atari input: (C, H, W).
    Mirrors the architecture you had in the notebook.
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int) -> None:
        super().__init__()
        c, _, _ = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self._output_size = self._get_conv_output_size(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(self._output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def _get_conv_output_size(self, input_shape: Tuple[int, int, int]) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.conv(dummy)
            return out.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # match notebook behaviour: normalize by 255 in the conv branch
        x = self.conv(x / 255.0)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def build_dqn_pair(
    input_shape: Tuple[int, int, int],
    num_actions: int,
    device: torch.device,
):
    """
    Build (single_DQN, target_single, double_DQN, target_double)
    just like the notebook, but packaged.
    """
    single = DQN(input_shape, num_actions).to(device)
    double = DQN(input_shape, num_actions).to(device)

    target_single = DQN(input_shape, num_actions).to(device)
    target_double = DQN(input_shape, num_actions).to(device)

    target_single.load_state_dict(single.state_dict())
    target_double.load_state_dict(double.state_dict())

    target_single.eval()
    target_double.eval()

    return single, target_single, double, target_double
