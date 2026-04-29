import os
from typing import Tuple, Union

import numpy as np
import torch


def read_history(file_path: str) -> np.ndarray:
    """Parse a per-line dict log into a (T, D) numpy array.

    Each line is a Python-style repr of {joint_name: np.array([v], dtype=float32)}.
    Joint order is fixed by the first line.
    """
    rows, keys = [], None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = eval(
                line.replace("array", "np.array").replace("dtype=float32", "dtype=np.float32")
            )
            if keys is None:
                keys = list(entry.keys())
            rows.append([float(entry[k]) for k in keys])
    return np.asarray(rows, dtype=np.float32)


def calc_atv(
    actions: torch.Tensor,
    dt: float = 1 / 30,
    get_max: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Mean (and optionally peak) speed per trajectory.

    Args:
        actions: (B, T, D) joint positions in radians.
        dt:      timestep in seconds.
    Returns:
        atv:     (B,)  mean speed.
        atv_max: (B,)  peak speed, only if `get_max`.
    """
    assert actions.dim() == 3, actions.shape
    vel = (actions[:, 1:] - actions[:, :-1]) / dt        # (B, T-1, D)
    speed = torch.norm(vel, dim=-1)                      # (B, T-1)
    atv = speed.mean(dim=-1)                             # (B,)
    if get_max:
        return atv, speed.max(dim=-1).values
    return atv


def calc_jerk_rms(
    states: torch.Tensor,
    dt: float = 1 / 30,
    get_max: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """RMS (and optionally peak) jerk magnitude per trajectory.

    Args:
        states: (B, T, D) joint positions in radians.
        dt:     timestep in seconds.
    Returns:
        rms:     (B,) RMS jerk magnitude.
        rms_max: (B,) peak jerk magnitude, only if `get_max`.
    """
    assert states.dim() == 3, states.shape
    vel = (states[:, 1:] - states[:, :-1]) / dt
    acc = (vel[:, 1:] - vel[:, :-1]) / dt
    jerk = (acc[:, 1:] - acc[:, :-1]) / dt               # (B, T-3, D)
    jerk_mag = torch.norm(jerk, dim=-1)                  # (B, T-3)
    rms = torch.sqrt((jerk_mag ** 2).mean(dim=-1))       # (B,)
    if get_max:
        return rms, jerk_mag.max(dim=-1).values
    return rms


def my_calc_jerk_rms(long_action):
    """
    long_action: torch.Tensor of shape (C, T, D)
    M: number of top high-jerk (least smooth) actions to return

    Returns:
        top_actions: (M, T, D)
        top_indices: (M,)
        jerk_rms: (C,)
    """
    assert long_action.ndim == 3
    C, T, D = long_action.shape
    assert T >= 4, "Need at least 4 timesteps to compute jerk"

    # Normalize long_action
    std = long_action.std(dim=(0, 1), keepdim=True)
    norm_long_action = long_action / (std + 1e-6)

    # Compute jerk (C, T-3, D-1)  (giữ nguyên logic của bạn)
    jerk = (
        norm_long_action[:, 3:, :-1]
        - 3 * norm_long_action[:, 2:-1, :-1]
        + 3 * norm_long_action[:, 1:-2, :-1]
        - norm_long_action[:, :-3, :-1]
    )

    # ||j||^2 over action dim → (C, T-3)
    jerk_sq = (jerk ** 2).sum(dim=-1)

    # mean over time → (C,)
    jerk_mean = jerk_sq.mean(dim=1)

    # RMS → (C,)
    jerk_rms = torch.sqrt(jerk_mean + 1e-8)

    return jerk_rms


def _load_trajectory(path: str) -> torch.Tensor:
    """Load a history .txt (degrees) and return (1, T, D) tensor in radians."""
    arr = read_history(path)                             # (T, D), degrees
    return (torch.from_numpy(arr) * (np.pi / 180.0)).unsqueeze(0)


def calc_atv_jerk(path: str, *args, **kwargs):
    actions = _load_trajectory(os.path.join(path, "action_history.txt"))
    states = _load_trajectory(os.path.join(path, "state_history.txt"))
    atv = calc_atv(actions, *args, **kwargs)
    jerk_rms = calc_jerk_rms(states, *args, **kwargs)
    return atv, jerk_rms

def calc_atv_only(path: str, args):
    actions = _load_trajectory(path) # 1,300,7
    if args.add_noise:
        actions += torch.randn_like(actions) * 0.01
    atv = calc_atv(actions, args.dt, args.get_max)
    return atv

def calc_jerk_only(path: str, args):
    actions = _load_trajectory(path) # 1,300,7
    print(actions.shape)
    actions = actions[:, :20]
    if args.add_noise:
        actions += torch.randn_like(actions) * 0.01
    jerk_rms = my_calc_jerk_rms(actions)
    return jerk_rms



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute ATV and Jerk RMS from trajectory logs")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to folder containing action_history.txt and state_history.txt",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1 / 30,
        help="Timestep (seconds), default = 1/30",
    )
    parser.add_argument(
        "--get_max",
        action="store_true",
        help="Also return max speed and max jerk",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Add noise to actions",
    )

    args = parser.parse_args()

    # Compute metrics
    atv = calc_atv_only(args.path, args)
    jerk = calc_jerk_only(args.path, args)

    # Print results
    if args.get_max:
        atv_mean, atv_max = atv
        jerk_mean, jerk_max = jerk

        print(f"ATV (mean): {atv_mean.item():.6f}")
        print(f"ATV (max):  {atv_max.item():.6f}")
        print(f"Jerk RMS (mean): {jerk_mean.item():.6f}")
        print(f"Jerk RMS (max):  {jerk_max.item():.6f}")
    else:
        print(f"ATV: {atv.item():.6f}")
        print(f"Jerk RMS: {jerk.item():.6f}")

# python measure_smoothness.py --path action_history/local/chunk_3.txt --add_noise


for loop in range(10):
    action_chunk = model(obs) # (1,50,7)

    action_chunk = action_chunk[:,:20] # our current motion set up use long_ah=20
    action_chunk_noise = action_chunk + torch.randn_like(action_chunk) * 0.01

    jerk_score = compute_jerk(action_chunk)
    jerk_score_add_noise = compute_jerk(action_chunk_noise)

    processed_action = post_processing(action_chunk)
    new_obs = env.step(processed_action[:,:10]) # assume we only execute first 10 actions to env
    obs = new_obs

