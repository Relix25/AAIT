"""Crafter metrics computed from complete episodes, never from partial steps."""
import csv
import json
import math
from pathlib import Path

import numpy as np


def summarize(episodes):
    if not episodes:
        return {"episodes": 0, "reward_mean": None, "reward_std": None, "score": None, "success": {}}
    rewards = np.array([float(ep["reward"]) for ep in episodes])
    keys = sorted({key for ep in episodes for key in ep if key.startswith("achievement_")})
    success = {key.removeprefix("achievement_"): 100.0 * np.mean([float(ep.get(key, 0)) > 0 for ep in episodes]) for key in keys}
    # Crafter defines 22 achievements. Missing ones must count as zero.
    score = math.exp(sum(math.log1p(rate) for rate in success.values()) / 22) - 1
    return {"episodes": len(episodes), "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
            "length_mean": float(np.mean([float(ep["length"]) for ep in episodes])),
            "score": score, "success": success}


def read_episodes(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_episode(path, episode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(episode, ensure_ascii=False) + "\n")


def write_scalar(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
