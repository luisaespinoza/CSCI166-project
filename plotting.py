# python
import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def rolling_avg(arr: List[float], w: int) -> np.ndarray:
    if len(arr) == 0:
        return np.array([])
    if len(arr) < w:
        return np.array([np.mean(arr[: i + 1]) for i in range(len(arr))])
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def plot_learning_curves(
    episode_rewards_single,
    episode_rewards_double,
    window_size: int = 100,
    save_path: str | None = None,
):
    ra_single = rolling_avg(episode_rewards_single, window_size)
    ra_double = rolling_avg(episode_rewards_double, window_size)

    print(
        f"[PLOT] single episodes={len(episode_rewards_single)}, "
        f"double episodes={len(episode_rewards_double)}"
    )

    plt.figure(figsize=(12, 5))

    # Single
    plt.subplot(1, 2, 1)
    if len(ra_single) > 0:
        x_s = np.arange(window_size - 1, window_size - 1 + len(ra_single))
        plt.plot(x_s, ra_single, label="single_DQN (avg)")
    else:
        plt.text(
            0.5,
            0.5,
            "No single_DQN rewards",
            horizontalalignment="center",
            transform=plt.gca().transAxes,
        )
    plt.title("single_DQN Learning Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.grid(True)

    # Double
    plt.subplot(1, 2, 2)
    if len(ra_double) > 0:
        x_d = np.arange(window_size - 1, window_size - 1 + len(ra_double))
        plt.plot(x_d, ra_double, label="double_DQN (avg)")
    else:
        plt.text(
            0.5,
            0.5,
            "No double_DQN rewards",
            horizontalalignment="center",
            transform=plt.gca().transAxes,
        )
    plt.title("double_DQN Learning Curve")
    plt.xlabel("Episodes")
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)  # works when parent is '.'
        plt.savefig(p, dpi=150)
        print(f"[PLOT] saved to {p}")
    else:
        plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent


    data_dir = project_root / "models"
    single_rewards_path = data_dir / "episode_rewards_single.npy"
    double_rewards_path = data_dir / "episode_rewards_double.npy"

    missing = [p for p in (single_rewards_path, double_rewards_path) if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] Missing file: `{p}` (expected under `50k_steps/models`).")
        print(f"[INFO] Current working directory: `{Path.cwd()}`")
        sys.exit(1)

    single_rewards = np.load(single_rewards_path).tolist()
    double_rewards = np.load(double_rewards_path).tolist()

    # provide an explicit save path (project root) to avoid empty dirname issues
    save_file = data_dir /"learning_curve.png"
    plot_learning_curves(
        single_rewards,
        double_rewards,
        window_size=20,
        save_path=str(save_file),
    )
