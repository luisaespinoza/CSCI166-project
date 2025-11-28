from __future__ import annotations

import collections
import random
from typing import Deque, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """
    Same semantics as in the notebook, but no hidden globals:
    device is passed in via the constructor.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.buffer: Deque = collections.deque(maxlen=capacity)
        self.device = device

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        # Ensure batch dimension when storing
        if state.ndim == 3:
            state = np.expand_dims(state, 0)
        if next_state.ndim == 3:
            next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = min(batch_size, len(self.buffer))
        experiences = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences)

        states_arr = np.concatenate(states, axis=0)
        next_states_arr = np.concatenate(next_states, axis=0)

        states_t = torch.from_numpy(states_arr).float().to(self.device)
        next_states_t = torch.from_numpy(next_states_arr).float().to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t
