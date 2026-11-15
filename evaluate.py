"""Evaluate checkpoints or an untrained random agent with identical episode metrics."""
import argparse
import json
from pathlib import Path

import crafter
import numpy as np
import torch

from agent.network import QNetwork
from metrics import summarize, write_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Exploration probability during evaluation (0 = greedy policy)")
    parser.add_argument("--out", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--trace", action="store_true", help="Record action and achievement steps for behavior review")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "episodes.jsonl"
    if path.exists():
        raise FileExistsError(f"Evaluation output already exists: {path}")
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = QNetwork().to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    records = []
    for episode in range(args.episodes):
        env = crafter.Env(seed=args.seed + episode)
        if args.video and episode < 3:
            env = crafter.Recorder(env, str(args.out / "videos"), save_stats=False,
                                   save_video=True, save_episode=False)
        obs = env.reset()
        done, total, length = False, 0.0, 0
        previous_achievements = {}
        while not done:
            if model is None or rng.random() < args.epsilon:
                action = int(rng.integers(17))
            else:
                with torch.no_grad():
                    action = int(model(torch.as_tensor(obs[None], device=device)).argmax().item())
            obs, reward, done, info = env.step(action)
            total += float(reward)
            length += 1
            if args.trace and episode < 3:
                unlocked = [key for key, count in info["achievements"].items()
                            if count > previous_achievements.get(key, 0)]
                write_episode(args.out / "trace.jsonl", {"episode": episode, "step": length,
                              "action": action, "reward": float(reward), "unlocked": unlocked,
                              "inventory": info["inventory"]})
                previous_achievements = dict(info["achievements"])
        record = {"seed": args.seed + episode, "reward": total, "length": length}
        record.update({f"achievement_{key}": int(value)
                       for key, value in info["achievements"].items()})
        records.append(record)
        write_episode(path, record)
        print(f"episode={episode + 1}/{args.episodes} reward={total:.2f} length={length}", flush=True)
    result = summarize(records)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
