"""Export concise, slide-ready figures from recorded runs."""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import read_episodes, summarize


def display_name(name):
    return {"double_nstep": "Double DQN + n-step", "final_150k": "Agent (2 seeds)",
            "epsilon0_final": "Agent greedy (2 seeds)",
            "double_nstep_1m_g4": "Double DQN + n-step",
            "final_1m_g4": "Agent (2 seeds)"}.get(name, name)


def save_objective_figure(out_dir: Path):
    """Render clean LaTeX/mathtext training objective card for Slide 4."""
    bg = "#0F172A"
    equations = [
        ("1. Action selection (Double DQN)",
         r"$a^* = \arg\max_{a}\; Q_{\mathrm{online}}(o_{t+3},\, a)$", 0.69),
        ("2. 3-step TD target",
         r"$y_t = R_t^{(3)} + \Gamma_t\, Q_{\mathrm{target}}(o_{t+3},\, a^*)$", 0.50),
        ("3. Huber TD loss",
         r"$\mathcal{L}(\theta) = \mathbb{E}_{\mathrm{replay}}\left[\,\mathrm{Huber}\left(y_t - Q_{\mathrm{online}}(o_t,\, a_t)\right)\,\right]$", 0.31),
    ]
    with plt.rc_context({"mathtext.fontset": "stixsans"}):
        fig = plt.figure(figsize=(10, 2.75), facecolor=bg)
        ax = fig.add_axes([0, 0, 1, 1], facecolor=bg)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

        ax.text(0.035, 0.87, "TRAINING OBJECTIVE", color="#38BDF8",
                fontsize=11, fontweight="bold", va="center", ha="left",
                family="sans-serif")

        for label, formula, y in equations:
            ax.text(0.035, y, label, color="#94A3B8", fontsize=12.5,
                    fontweight="bold", va="center", ha="left", family="sans-serif")
            ax.text(0.38, y, formula, color="#FFFFFF", fontsize=15.5,
                    va="center", ha="left")

        ax.plot([0.035, 0.965], [0.20, 0.20], color="#334155", linewidth=1.0)

        footnote_math = (
            r"$R_t^{(3)} = r_{t+1} + \gamma\, r_{t+2} + \gamma^2\, r_{t+3}"
            r"\qquad\cdot\qquad "
            r"\Gamma_t = \gamma^h \cdot m_t$"
        )
        footnote_note = (
            r"$h = 3$" " (or " r"$h = 1, 2$"
            " for the last two transitions of an episode);   "
            r"$m_t = 0$" " on terminal death, " r"$m_t = 1$"
            " on 10k time-limit truncation"
        )
        ax.text(0.035, 0.125, footnote_math, color="#E2E8F0", fontsize=11.0,
                va="center", ha="left", family="sans-serif")
        ax.text(0.035, 0.05, footnote_note, color="#94A3B8", fontsize=10.0,
                va="center", ha="left", family="sans-serif")

        fig.savefig(out_dir / "training_objective.png", dpi=300, facecolor=bg)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="*", default=[])
    parser.add_argument("--eval", type=Path, nargs="*", default=[])
    parser.add_argument("--eval-curve", type=Path, nargs="*", default=[],
                        help="Checkpoint evaluation folders with a 50k/100k/... token in the name")
    parser.add_argument("--scalars", type=Path, nargs="*", default=[],
                        help="Training run folders with scalars.csv (loss and mean Q-value per 1k steps)")
    parser.add_argument("--random", type=Path)
    parser.add_argument("--max-step", type=int, help="Limit training curves to the selected budget")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    save_objective_figure(args.out)
    random_stats = summarize(read_episodes(args.random / "episodes.jsonl")) if args.random else None

    if args.train:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        series = {}
        for folder in args.train:
            episodes = read_episodes(folder / "episodes.jsonl")
            if not episodes:
                continue
            frame = pd.DataFrame(episodes)
            if "step" not in frame:
                continue
            if args.max_step is not None:
                frame = frame[frame["step"] <= args.max_step]
            frame["bin"] = (frame["step"] // 10000) * 10000
            name = re.sub(r"_seed\d+$", "", folder.name)
            series.setdefault(name, []).append(frame.groupby("bin")["reward"].mean())
        for name, seeds in series.items():
            aligned = pd.concat(seeds, axis=1)
            mean = aligned.mean(axis=1)
            ax.plot(mean.index, mean, label=f"{display_name(name)} ({len(seeds)} seed{'s' if len(seeds)>1 else ''})")
            if len(seeds) > 1:
                sd = aligned.std(axis=1)
                ax.fill_between(mean.index, mean - sd, mean + sd, alpha=0.18)
        if random_stats and random_stats["episodes"]:
            ax.axhline(random_stats["reward_mean"], color="black", linestyle="--", label="Random")
        ax.set(xlabel="Environment steps", ylabel="Mean episodic reward (10k-step bins)",
               title="Training performance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / "training_reward.png", dpi=180)
        plt.close(fig)

    if args.eval:
        grouped = {}
        for folder in args.eval:
            stats = summarize(read_episodes(folder / "episodes.jsonl"))
            if stats["episodes"]:
                name = re.sub(r"_seed\d+$", "", folder.name)
                grouped.setdefault(name, []).append(stats)
        groups = []
        for name, seed_stats in grouped.items():
            success_names = sorted({key for stats in seed_stats for key in stats["success"]})
            rewards = np.array([stats["reward_mean"] for stats in seed_stats])
            groups.append((display_name(name), {"reward_mean": float(rewards.mean()),
                                  "reward_error": float(rewards.std(ddof=1)) if len(rewards) > 1
                                                  else seed_stats[0]["reward_std"] / np.sqrt(seed_stats[0]["episodes"]),
                                  "success": {key: float(np.mean([stats["success"].get(key, 0)
                                                                   for stats in seed_stats])) for key in success_names}}))
        if random_stats and random_stats["episodes"]:
            groups.insert(0, ("Random", {**random_stats,
                                         "reward_error": 0.0}))
        if groups:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar([x[0] for x in groups], [x[1]["reward_mean"] for x in groups],
                   yerr=[x[1]["reward_error"] for x in groups], capsize=4)
            ax.set(ylabel="Mean episodic reward", title="Fresh-episode evaluation")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            fig.savefig(args.out / "evaluation_reward.png", dpi=180)
            plt.close(fig)

            names = sorted({name for _, stat in groups for name in stat["success"]})
            height = max(5, 0.34 * len(names))
            fig, ax = plt.subplots(figsize=(9, height))
            y = np.arange(len(names))
            width = 0.8 / len(groups)
            for i, (label, stat) in enumerate(groups):
                ax.barh(y + (i - (len(groups)-1)/2) * width,
                        [stat["success"].get(name, 0) for name in names], width, label=label)
            ax.set_yticks(y, [name.replace("_", " ") for name in names])
            ax.set(xlabel="Episodes with achievement (%)", title="Achievement success rates")
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.out / "achievements.png", dpi=180)
            plt.close(fig)

    if args.eval_curve:
        grouped_points = {}
        for folder in args.eval_curve:
            match = re.search(r"(?:^|_)(\d+)([km])?(?:_|$)", folder.name)
            if not match:
                raise ValueError(f"Folder needs a checkpoint step in its name: {folder}")
            step = int(match.group(1)) * {"k": 1000, "m": 1_000_000}.get(match.group(2), 1)
            stats = summarize(read_episodes(folder / "episodes.jsonl"))
            if stats["episodes"]:
                grouped_points.setdefault(step, []).append(stats)
        points = sorted(grouped_points.items())
        if points:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            steps = [step for step, _ in points]
            for ax, key, title in zip(axes, ("reward_mean", "score"),
                                      ("Mean episodic reward", "Crafter score (%)")):
                means = np.array([np.mean([stats[key] for stats in seed_stats])
                                  for _, seed_stats in points])
                ax.plot(steps, means, marker="o", label="Agent mean")
                if all(len(seed_stats) > 1 for _, seed_stats in points):
                    sd = np.array([np.std([stats[key] for stats in seed_stats], ddof=1)
                                   for _, seed_stats in points])
                    ax.fill_between(steps, means - sd, means + sd, alpha=0.18,
                                    label="±1 SD between seeds")
                if random_stats and random_stats["episodes"]:
                    ax.axhline(random_stats[key], color="black", linestyle="--", label="Random")
                ax.set(xlabel="Checkpoint step", title=title)
                ax.legend()
            fig.tight_layout()
            fig.savefig(args.out / "evaluation_curve.png", dpi=180)
            plt.close(fig)

    if args.scalars:
        frames = []
        for folder in args.scalars:
            path = folder / "scalars.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path).dropna(subset=["loss", "q_mean"])
            if args.max_step is not None:
                frame = frame[frame["step"] <= args.max_step]
            if len(frame):
                frames.append(frame.set_index("step")[["q_mean", "loss"]])
        if frames:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            label = f"Agent mean ({len(frames)} seed{'s' if len(frames) > 1 else ''})"
            for ax, key, title in zip(axes, ("q_mean", "loss"),
                                      ("Mean predicted Q-value (sampled batches)", "Huber TD loss")):
                aligned = pd.concat([frame[key] for frame in frames], axis=1)
                mean = aligned.mean(axis=1)
                ax.plot(mean.index, mean, label=label)
                if len(frames) > 1:
                    sd = aligned.std(axis=1)
                    ax.fill_between(mean.index, mean - sd, mean + sd, alpha=0.18,
                                    label="±1 SD between seeds")
                ax.set(xlabel="Environment steps", title=title)
                ax.legend()
            fig.tight_layout()
            fig.savefig(args.out / "training_diagnostics.png", dpi=180)
            plt.close(fig)


if __name__ == "__main__":
    main()
