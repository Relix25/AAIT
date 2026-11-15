"""Aggregate independent evaluation seeds without pooling away seed variance."""
import argparse
import json
from pathlib import Path

import numpy as np

from metrics import read_episodes, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+", help="One evaluation folder per training seed")
    parser.add_argument("--out", type=Path, default=Path("runs/final_report.json"))
    args = parser.parse_args()
    rows = []
    for folder in args.runs:
        stats = summarize(read_episodes(folder / "episodes.jsonl"))
        if not stats["episodes"]:
            raise ValueError(f"No complete episodes: {folder}")
        rows.append({"run": str(folder), **stats})
    names = sorted({name for row in rows for name in row["success"]})
    values = lambda key: np.array([row[key] for row in rows], dtype=float)
    result = {
        "training_seeds": len(rows),
        "episodes_per_seed": [row["episodes"] for row in rows],
        "reward_mean": float(values("reward_mean").mean()),
        "reward_sd_between_seeds": float(values("reward_mean").std(ddof=1)) if len(rows) > 1 else None,
        "score_mean": float(values("score").mean()),
        "score_sd_between_seeds": float(values("score").std(ddof=1)) if len(rows) > 1 else None,
        "success_mean": {name: float(np.mean([row["success"].get(name, 0) for row in rows])) for name in names},
        "per_seed": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("per_seed", "success_mean")}, indent=2))


if __name__ == "__main__":
    main()
