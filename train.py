import os
import time
import uuid
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
import yaml

from envs import (
    EnvConfig,
    RewardConfig,
    env_config_from_dict,
    reward_config_from_dict,
    make_training_env,
    make_vector_envs,
    evaluate_and_record,
)
from losses import compute_loss, compute_loss_double_dqn
from models import build_dqn_pair
from replay_buffer import ReplayBuffer
import random

def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(cfg: Dict) -> torch.device:
    which = cfg.get("device", {}).get("use", "auto")
    if which == "cpu":
        return torch.device("cpu")
    if which == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def populate_replay_buffers(
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
    training_cfg: Dict,
    device: torch.device,
) -> Tuple[ReplayBuffer, ReplayBuffer]:
    """
    Vectorized population phase using AtariVectorEnv, as in the notebook.
    """
    capacity = training_cfg["replay_buffer_capacity"]
    num_initial_steps = training_cfg["num_initial_steps"]

    buf_single = ReplayBuffer(capacity, device=device)
    buf_double = ReplayBuffer(capacity, device=device)

    vec_single, vec_double = make_vector_envs(env_cfg,reward_cfg)

    obs_single, info_single = vec_single.reset()
    obs_double, info_double = vec_double.reset()

    collected_single = 0
    collected_double = 0

    print(
        f"[POPULATE] num_initial_steps={num_initial_steps}, "
        f"num_envs={env_cfg.vec_num_envs}"
    )

    while collected_single < num_initial_steps or collected_double < num_initial_steps:
        actions_single = vec_single.action_space.sample()
        actions_double = vec_double.action_space.sample()

        next_obs_s, rewards_s, term_s, trunc_s, infos_s = vec_single.step(actions_single)
        next_obs_d, rewards_d, term_d, trunc_d, infos_d = vec_double.step(actions_double)

        dones_s = np.logical_or(term_s, trunc_s)
        dones_d = np.logical_or(term_d, trunc_d)

        for i in range(env_cfg.vec_num_envs):
            if collected_single < num_initial_steps:
                buf_single.push(
                    obs_single[i],
                    int(actions_single[i]),
                    float(rewards_s[i]),
                    next_obs_s[i],
                    bool(dones_s[i]),
                )
                collected_single += 1

            if collected_double < num_initial_steps:
                buf_double.push(
                    obs_double[i],
                    int(actions_double[i]),
                    float(rewards_d[i]),
                    next_obs_d[i],
                    bool(dones_d[i]),
                )
                collected_double += 1

        obs_single = next_obs_s
        obs_double = next_obs_d

        if (collected_single % 500 == 0 and collected_single > 0) or (
            collected_double % 500 == 0 and collected_double > 0
        ):
            print(
                f"[POPULATE] single={collected_single}/{num_initial_steps}, "
                f"double={collected_double}/{num_initial_steps}, "
                f"buf_sizes=({len(buf_single)}, {len(buf_double)})"
            )

    vec_single.close()
    vec_double.close()

    print(
        f"[POPULATE] done. final buffer sizes: "
        f"single={len(buf_single)}, double={len(buf_double)}"
    )
    return buf_single, buf_double
def train_interleaved(
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
    training_cfg: Dict,
    device: torch.device,
    single_DQN: torch.nn.Module,
    target_single: torch.nn.Module,
    double_DQN: torch.nn.Module,
    target_double: torch.nn.Module,
    replay_buffer_single: ReplayBuffer,
    replay_buffer_double: ReplayBuffer,
) -> Tuple[List[float], List[float]]:
    """
    Main interleaved training loop with epsilon-greedy action selection
    using single_DQN and double_DQN respectively.
    """
    batch_size = training_cfg["batch_size"]
    training_iters = training_cfg["training_iters"]
    gamma = training_cfg["gamma"]
    target_update_freq = training_cfg["target_update_frequency"]

    lr = training_cfg["learning_rate"]
    optimizer_single = optim.Adam(single_DQN.parameters(), lr=lr)
    optimizer_double = optim.Adam(double_DQN.parameters(), lr=lr)

    # epsilon schedule
    eps_start = training_cfg.get("epsilon_start", 1.0)
    eps_end = training_cfg.get("epsilon_end", 0.01)
    eps_decay = training_cfg.get("epsilon_decay", 50000)  # in env steps

    def current_epsilon(step: int) -> float:
        # Linear decay from eps_start to eps_end over eps_decay steps
        fraction = min(1.0, step / float(eps_decay)) if eps_decay > 0 else 1.0
        return eps_start + (eps_end - eps_start) * fraction

    def select_action_epsilon_greedy(
        model: torch.nn.Module,
        obs: np.ndarray,
        env,
        epsilon: float,
    ) -> int:
        # obs: (C, H, W) or similar numpy array
        if random.random() < epsilon:
            return env.action_space.sample()
        with torch.no_grad():
            state_t = torch.tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            q_values = model(state_t)
            return int(q_values.argmax(dim=1).item())

    env_single = make_training_env(env_cfg, reward_cfg)
    env_double = make_training_env(env_cfg, reward_cfg)

    obs_single, _ = env_single.reset()
    obs_double, _ = env_double.reset()

    ep_reward_s = 0.0
    ep_reward_d = 0.0
    ep_len_s = 0
    ep_len_d = 0

    episode_rewards_single: List[float] = []
    episode_rewards_double: List[float] = []

    total_steps = 0
    last_loss_single = None
    last_loss_double = None

    run_id = uuid.uuid4().hex
    run_ts = time.time()
    print(f"[TRAIN] run_id={run_id}, ts={run_ts}, iters={training_iters}")

    for t in range(training_iters):
        # compute epsilon from total_steps (shared schedule)
        epsilon = current_epsilon(total_steps)

        # --- collect from single learner env (epsilon-greedy using single_DQN) ---
        a_s = select_action_epsilon_greedy(single_DQN, obs_single, env_single, epsilon)
        n_s, r_s, term_s, trunc_s, info_s = env_single.step(a_s)
        done_s = term_s or trunc_s
        replay_buffer_single.push(obs_single, a_s, r_s, n_s, done_s)
        ep_reward_s += float(r_s)
        ep_len_s += 1
        obs_single = n_s
        if done_s:
            episode_rewards_single.append(ep_reward_s)
            ep_reward_s = 0.0
            ep_len_s = 0
            obs_single, _ = env_single.reset()

        # --- collect from double learner env (epsilon-greedy using double_DQN) ---
        a_d = select_action_epsilon_greedy(double_DQN, obs_double, env_double, epsilon)
        n_d, r_d, term_d, trunc_d, info_d = env_double.step(a_d)
        done_d = term_d or trunc_d
        replay_buffer_double.push(obs_double, a_d, r_d, n_d, done_d)
        ep_reward_d += float(r_d)
        ep_len_d += 1
        obs_double = n_d
        if done_d:
            episode_rewards_double.append(ep_reward_d)
            ep_reward_d = 0.0
            ep_len_d = 0
            obs_double, _ = env_double.reset()

        # --- learner updates ---
        if len(replay_buffer_single) >= batch_size:
            batch_s = replay_buffer_single.sample(batch_size)
            optimizer_single.zero_grad()
            loss_s = compute_loss(batch_s, single_DQN, target_single, gamma)
            loss_s.backward()
            optimizer_single.step()
            last_loss_single = float(loss_s.detach().cpu().item())

        if len(replay_buffer_double) >= batch_size:
            batch_d = replay_buffer_double.sample(batch_size)
            optimizer_double.zero_grad()
            loss_d = compute_loss_double_dqn(batch_d, double_DQN, target_double, gamma)
            loss_d.backward()
            optimizer_double.step()
            last_loss_double = float(loss_d.detach().cpu().item())

        total_steps += 1

        if total_steps % target_update_freq == 0:
            target_single.load_state_dict(single_DQN.state_dict())
            target_double.load_state_dict(double_DQN.state_dict())

        if (t + 1) % 500 == 0 or t == 0:
            print(
                f"[TRAIN] iter={t+1}/{training_iters}, "
                f"eps={epsilon:.3f}, "
                f"loss_single={last_loss_single}, "
                f"loss_double={last_loss_double}, total_steps={total_steps}"
            )

    env_single.close()
    env_double.close()
    print("[TRAIN] interleaved training finished.")

    return episode_rewards_single, episode_rewards_double


# def train_interleaved(
#     env_cfg: EnvConfig,
#     reward_cfg: RewardConfig,
#     training_cfg: Dict,
#     device: torch.device,
#     single_DQN: torch.nn.Module,
#     target_single: torch.nn.Module,
#     double_DQN: torch.nn.Module,
#     target_double: torch.nn.Module,
#     replay_buffer_single: ReplayBuffer,
#     replay_buffer_double: ReplayBuffer,
# ) -> Tuple[List[float], List[float]]:
#     """
#     Main interleaved training loop from the notebook, cleaned up & modular.
#     """
#     batch_size = training_cfg["batch_size"]
#     training_iters = training_cfg["training_iters"]
#     gamma = training_cfg["gamma"]
#     target_update_freq = training_cfg["target_update_frequency"]
#
#     lr = training_cfg["learning_rate"]
#     optimizer_single = optim.Adam(single_DQN.parameters(), lr=lr)
#     optimizer_double = optim.Adam(double_DQN.parameters(), lr=lr)
#
#     env_single = make_training_env(env_cfg, reward_cfg)
#     env_double = make_training_env(env_cfg, reward_cfg)
#
#     obs_single, _ = env_single.reset()
#     obs_double, _ = env_double.reset()
#
#     ep_reward_s = 0.0
#     ep_reward_d = 0.0
#     ep_len_s = 0
#     ep_len_d = 0
#
#     episode_rewards_single: List[float] = []
#     episode_rewards_double: List[float] = []
#
#     total_steps = 0
#     last_loss_single = None
#     last_loss_double = None
#
#     run_id = uuid.uuid4().hex
#     run_ts = time.time()
#     print(f"[TRAIN] run_id={run_id}, ts={run_ts}, iters={training_iters}")
#
#     for t in range(training_iters):
#         # --- collect from single learner env ---
#         a_s = env_single.action_space.sample()  # still random actions, like notebook
#         n_s, r_s, term_s, trunc_s, info_s = env_single.step(a_s)
#         done_s = term_s or trunc_s
#         replay_buffer_single.push(obs_single, a_s, r_s, n_s, done_s)
#         ep_reward_s += float(r_s)
#         ep_len_s += 1
#         obs_single = n_s
#         if done_s:
#             episode_rewards_single.append(ep_reward_s)
#             ep_reward_s = 0.0
#             ep_len_s = 0
#             obs_single, _ = env_single.reset()
#
#         # --- collect from double learner env ---
#         a_d = env_double.action_space.sample()
#         n_d, r_d, term_d, trunc_d, info_d = env_double.step(a_d)
#         done_d = term_d or trunc_d
#         replay_buffer_double.push(obs_double, a_d, r_d, n_d, done_d)
#         ep_reward_d += float(r_d)
#         ep_len_d += 1
#         obs_double = n_d
#         if done_d:
#             episode_rewards_double.append(ep_reward_d)
#             ep_reward_d = 0.0
#             ep_len_d = 0
#             obs_double, _ = env_double.reset()
#
#         # --- learner updates ---
#         if len(replay_buffer_single) >= batch_size:
#             batch_s = replay_buffer_single.sample(batch_size)
#             optimizer_single.zero_grad()
#             loss_s = compute_loss(batch_s, single_DQN, target_single, gamma)
#             loss_s.backward()
#             optimizer_single.step()
#             last_loss_single = float(loss_s.detach().cpu().item())
#
#         if len(replay_buffer_double) >= batch_size:
#             batch_d = replay_buffer_double.sample(batch_size)
#             optimizer_double.zero_grad()
#             loss_d = compute_loss_double_dqn(batch_d, double_DQN, target_double, gamma)
#             loss_d.backward()
#             optimizer_double.step()
#             last_loss_double = float(loss_d.detach().cpu().item())
#
#         total_steps += 1
#
#         if total_steps % target_update_freq == 0:
#             target_single.load_state_dict(single_DQN.state_dict())
#             target_double.load_state_dict(double_DQN.state_dict())
#
#         if (t + 1) % 500 == 0 or t == 0:
#             print(
#                 f"[TRAIN] iter={t+1}/{training_iters}, "
#                 f"loss_single={last_loss_single}, "
#                 f"loss_double={last_loss_double}, total_steps={total_steps}"
#             )
#
#     env_single.close()
#     env_double.close()
#     print("[TRAIN] interleaved training finished.")
#
#     return episode_rewards_single, episode_rewards_double


def save_models(
    cfg: Dict,
    single_DQN: torch.nn.Module,
    double_DQN: torch.nn.Module,
) -> None:
    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    single_path = os.path.join(models_dir, "single_dqn_model.pth")
    double_path = os.path.join(models_dir, "double_dqn_model.pth")

    torch.save(single_DQN.state_dict(), single_path)
    torch.save(double_DQN.state_dict(), double_path)

    print(f"[SAVE] single DQN -> {single_path}")
    print(f"[SAVE] double DQN -> {double_path}")


def main():
    cfg = load_config("config.yaml")
    device = get_device(cfg)
    print("[SETUP] device:", device)

    env_cfg = env_config_from_dict(cfg)
    reward_cfg = reward_config_from_dict(cfg)
    training_cfg = cfg["training"]

    # Base env (for shape + action space)
    tmp_env = make_training_env(env_cfg, reward_cfg)
    input_shape = tmp_env.observation_space.shape  # (C, H, W)
    num_actions = tmp_env.action_space.n
    print("[SETUP] input_shape:", input_shape, "num_actions:", num_actions)

    single_DQN, target_single, double_DQN, target_double = build_dqn_pair(
        input_shape, num_actions, device
    )

    # Initial quick evaluation + videos (optional)
    evaluate_and_record(
        single_DQN,
        env_cfg,
        video_folder=cfg["paths"]["video_single_init"],
        name_prefix="single_DQN_episode_0",
        device=device,
    )
    evaluate_and_record(
        double_DQN,
        env_cfg,
        video_folder=cfg["paths"]["video_double_init"],
        name_prefix="double_DQN_episode_0",
        device=device,
    )

    # Populate separate replay buffers
    buf_single, buf_double = populate_replay_buffers(env_cfg, reward_cfg, training_cfg, device)

    # Train
    ep_rewards_single, ep_rewards_double = train_interleaved(
        env_cfg,
        reward_cfg,
        training_cfg,
        device,
        single_DQN,
        target_single,
        double_DQN,
        target_double,
        buf_single,
        buf_double,
    )

    # Save models
    save_models(cfg, single_DQN, double_DQN)

    # Final eval + videos
    evaluate_and_record(
        single_DQN,
        env_cfg,
        video_folder=cfg["paths"]["video_single_final"],
        name_prefix="single_DQN_episode_final",
        device=device,
    )
    evaluate_and_record(
        double_DQN,
        env_cfg,
        video_folder=cfg["paths"]["video_double_final"],
        name_prefix="double_DQN_episode_final",
        device=device,
    )

    # Optionally: dump rewards to disk for later plotting
    np.save(os.path.join(cfg["paths"]["models_dir"], "episode_rewards_single.npy"), ep_rewards_single)
    np.save(os.path.join(cfg["paths"]["models_dir"], "episode_rewards_double.npy"), ep_rewards_double)


if __name__ == "__main__":
    main()
