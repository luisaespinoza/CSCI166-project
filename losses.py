from typing import Tuple

import torch
import torch.nn as nn


def compute_loss(
    batch: Tuple[torch.Tensor, ...],
    model: nn.Module,
    target_model: nn.Module,
    gamma: float,
) -> torch.Tensor:
    """
    Standard DQN loss, using the model and target_model arguments
    (mirrors your notebook).
    """
    states, actions, rewards, next_states, dones = batch

    current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_model(next_states).max(1)[0].detach()

    target_q_values = rewards + gamma * next_q_values * (1.0 - dones)
    loss = nn.MSELoss()(current_q_values, target_q_values)
    return loss


def compute_loss_double_dqn(
    batch: Tuple[torch.Tensor, ...],
    model: nn.Module,
    target_model: nn.Module,
    gamma: float,
) -> torch.Tensor:
    """
    Double DQN loss
    """
    states, actions, rewards, next_states, dones = batch

    current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_actions = model(next_states).argmax(dim=1)
        next_q_values = (
            target_model(next_states)
            .gather(1, next_actions.unsqueeze(1))
            .squeeze(1)
        )

    target_q_values = rewards + gamma * next_q_values * (1.0 - dones)
    loss = nn.MSELoss()(current_q_values, target_q_values)
    return loss
