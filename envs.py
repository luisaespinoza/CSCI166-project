from __future__ import annotations
from PIL import Image
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
import numpy as np
from gymnasium import Wrapper
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    RecordEpisodeStatistics,
    RecordVideo,
)



@dataclass
class EnvConfig:
    gym_id: str
    frameskip: int
    grayscale_obs: bool
    screen_size: int
    frame_stack: int

    vec_game: str
    vec_num_envs: int
    vec_frameskip: int
    vec_grayscale: bool
    vec_stack_num: int
    vec_img_height: int
    vec_img_width: int
    vec_maxpool: bool
    vec_reward_clipping: bool
    vec_noop_max: int
    vec_use_fire_reset: bool
    vec_episodic_life: bool

    recordings_dir: str
    recordings_dir_custom: str


@dataclass
class RewardConfig:
    reward_scale: float
    score_weight: float
    survival_reward: float
    termination_penalty: float


class SimpleAtariPreprocessing(gym.Wrapper):
    """
    Minimal drop-in replacement for gymnasium.wrappers.AtariPreprocessing
    that does NOT depend on cv2 / opencv.

    - Resizes to (screen_size, screen_size)
    - Optional grayscale
    - No fancy max-pooling over frames, but good enough for a class project.
    """

    def __init__(
        self,
        env,
        screen_size: int = 84,
        grayscale_obs: bool = True,
        scale_obs: bool = False,  # kept for API compatibility, but ignored
        **kwargs,
    ):
        super().__init__(env)
        self.screen_size = screen_size
        self.grayscale_obs = grayscale_obs

        if grayscale_obs:
            obs_shape = (screen_size, screen_size)
        else:
            obs_shape = (screen_size, screen_size, 3)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )

    def _process_obs(self, obs):
        # obs is (H, W, C) from Atari env: uint8 RGB
        img = Image.fromarray(obs)

        # Resize
        img = img.resize((self.screen_size, self.screen_size), Image.BILINEAR)

        # Grayscale if requested
        if self.grayscale_obs:
            img = img.convert("L")   # single channel
            arr = np.array(img, dtype=np.uint8)
        else:
            arr = np.array(img, dtype=np.uint8)

        return arr

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._process_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_obs(obs)
        return obs, reward, terminated, truncated, info
def env_config_from_dict(cfg: Dict) -> EnvConfig:
    env = cfg["env"]
    paths = cfg["paths"]
    return EnvConfig(
        gym_id=env["gym_id"],
        frameskip=env["frameskip"],
        grayscale_obs=env["grayscale_obs"],
        screen_size=env["screen_size"],
        frame_stack=env["frame_stack"],
        vec_game=env["vec_game"],
        vec_num_envs=env["vec_num_envs"],
        vec_frameskip=env["vec_frameskip"],
        vec_grayscale=env["vec_grayscale"],
        vec_stack_num=env["vec_stack_num"],
        vec_img_height=env["vec_img_height"],
        vec_img_width=env["vec_img_width"],
        vec_maxpool=env["vec_maxpool"],
        vec_reward_clipping=env["vec_reward_clipping"],
        vec_noop_max=env["vec_noop_max"],
        vec_use_fire_reset=env["vec_use_fire_reset"],
        vec_episodic_life=env["vec_episodic_life"],
        recordings_dir=paths["recordings_dir"],
        recordings_dir_custom=paths["recordings_dir_custom"],
    )


def reward_config_from_dict(cfg: Dict) -> RewardConfig:
    rw = cfg["reward_wrapper"]
    return RewardConfig(
        reward_scale=rw["reward_scale"],
        score_weight=rw["score_weight"],
        survival_reward=rw["survival_reward"],
        termination_penalty=rw["termination_penalty"],
    )


class CustomRewardWrapper(Wrapper):
    """
    Reward shaping for Tetris:

    - base ALE reward is the score delta from the env (`reward`)
    - add a small per-step survival bonus
    - add a moderate penalty on terminal, but not huge
    """

    def __init__(
            self,
            env: gym.Env,
            reward_scale: float = 1.0,
            score_weight: float = 1.0,
            survival_reward: float = 0.1,
            termination_penalty: float = -20.0,
    ):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.score_weight = score_weight
        self.survival_reward = survival_reward
        self.termination_penalty = termination_penalty

        self.last_score = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_score = info.get("score", self.last_score)
        score_increase = max(0, current_score - self.last_score)

        # Base env reward is usually already score_delta, but we'll treat it generically
        modified_reward = self.reward_scale * reward

        # Reward making progress
        modified_reward += self.score_weight * score_increase

        # Reward surviving one more step
        modified_reward += self.survival_reward

        # Penalize terminal, but not so hard it dwarfs everything else
        if terminated:
            modified_reward += self.termination_penalty

        self.last_score = current_score
        return obs, modified_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_score = info.get("score", 0)
        return obs, info



def make_training_env(env_cfg: EnvConfig, reward_cfg: RewardConfig) -> gym.Env:
    """
    Single Atari env with preprocessing, frame-stack, custom reward.
    """
    env = gym.make(env_cfg.gym_id, render_mode="rgb_array", frameskip=env_cfg.frameskip)
    # env = AtariPreprocessing(
    #     env,
    #     grayscale_obs=env_cfg.grayscale_obs,
    #     terminal_on_life_loss=False,
    #     screen_size=env_cfg.screen_size,
    # )
    env = SimpleAtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
        # frame_skip is ignored in SimpleAtariPreprocessing, but you can still pass it in kwargs if you like.
    )
    env = FrameStackObservation(env, stack_size=env_cfg.frame_stack)
    env = RecordEpisodeStatistics(env)
    env = CustomRewardWrapper(
        env,
        reward_scale=reward_cfg.reward_scale,
        score_weight=reward_cfg.score_weight,
        survival_reward=reward_cfg.survival_reward,
        termination_penalty=reward_cfg.termination_penalty,
    )
    return env


def make_recording_env(
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig | None,
    video_folder: str,
    name_prefix: str,
) -> gym.Env:
    """
    Env used for video recording (optionally with custom reward).
    """
    os.makedirs(video_folder, exist_ok=True)

    env = gym.make(env_cfg.gym_id, render_mode="rgb_array", frameskip=env_cfg.frameskip)
    # env = AtariPreprocessing(
    #     env,
    #     grayscale_obs=env_cfg.grayscale_obs,
    #     terminal_on_life_loss=False,
    #     screen_size=env_cfg.screen_size,
    # )
    env = SimpleAtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
        # frame_skip is ignored in SimpleAtariPreprocessing, but you can still pass it in kwargs if you like.
    )
    env = FrameStackObservation(env, stack_size=env_cfg.frame_stack)

    if reward_cfg is not None:
        env = CustomRewardWrapper(
            env,
            reward_scale=reward_cfg.reward_scale,
            score_weight=reward_cfg.score_weight,
            survival_reward=reward_cfg.survival_reward,
            termination_penalty=reward_cfg.termination_penalty,
        )

    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep: ep == 0,
        name_prefix=name_prefix,
    )
    return env


def make_vector_envs(env_cfg: EnvConfig, reward_cfg: RewardConfig):
    """
    Two independent vectorized envs built using Gymnasium's SyncVectorEnv.
    No dependency on ale_py.AtariVectorEnv.

    Each vector env runs `vec_num_envs` copies of the training env in parallel.
    """

    def make_single_env():
        # Reuse the same preprocessing + reward wrapper as training
        return make_training_env(env_cfg, reward_cfg)

    # One set of envs for the "single" learner
    vec_single = SyncVectorEnv(
        [make_single_env for _ in range(env_cfg.vec_num_envs)]
    )

    # And a completely separate set for the "double" learner
    vec_double = SyncVectorEnv(
        [make_single_env for _ in range(env_cfg.vec_num_envs)]
    )

    return vec_single, vec_double

def evaluate_and_record(
    model,
    env_cfg: EnvConfig,
    video_folder: str,
    name_prefix: str,
    device,
) -> None:
    """
    Your evaluate_and_record helper, but parameterized and re-usable.
    """
    env = make_recording_env(env_cfg, reward_cfg=None, video_folder=video_folder, name_prefix=name_prefix)
    model.eval()
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        import torch  # local import to avoid circulars

        with torch.no_grad():
            state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = model(state_t).argmax(dim=1).item()
        obs, r, term, trunc, info = env.step(action)
        total_reward += float(r)
        steps += 1
        if term or trunc:
            break

    env.close()
    print(
        f"[EVAL] {name_prefix}: reward={total_reward:.2f}, "
        f"steps={steps}, video_dir={video_folder}"
    )
