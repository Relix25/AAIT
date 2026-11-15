"""Train the selected hand-written Double DQN + n-step Crafter agent."""
import argparse
import json
import random
import time
from pathlib import Path

import crafter
import numpy as np
import torch
from torch import nn

from agent.network import QNetwork
from agent.replay import NStepAccumulator, ReplayBuffer
from metrics import write_episode, write_scalar


def parse_args():
    parser = argparse.ArgumentParser(description="Train a value-based Crafter agent")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", default="double_nstep")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--algorithm", choices=["dqn", "double"], default="double")
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-frequency", type=int, default=2_000)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=150_000)
    parser.add_argument("--checkpoint-frequency", type=int, default=100_000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def optimize(online, target, optimizer, replay, args, rng, device):
    obs, actions, rewards, next_obs, discounts = replay.sample(args.batch_size, rng)
    obs = torch.as_tensor(obs, device=device)
    actions = torch.as_tensor(actions, device=device).long()
    rewards = torch.as_tensor(rewards, device=device)
    next_obs = torch.as_tensor(next_obs, device=device)
    discounts = torch.as_tensor(discounts, device=device)
    q = online(obs).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        if args.algorithm == "double":
            next_actions = online(next_obs).argmax(dim=1, keepdim=True)
            next_q = target(next_obs).gather(1, next_actions).squeeze(1)
        else:
            next_q = target(next_obs).max(dim=1).values
        td_target = rewards + discounts * next_q
    loss = nn.functional.smooth_l1_loss(q, td_target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10)
    optimizer.step()
    return float(loss.item()), float(q.mean().item())


def main():
    args = parse_args()
    if args.n_step < 1 or args.buffer_size < args.batch_size or args.epsilon_decay_steps < 1:
        raise ValueError("Invalid n-step, buffer, or epsilon schedule")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    run = Path(args.out) / f"{args.run_name}_seed{args.seed}"
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Run already exists; choose a new --out or --run-name: {run}")
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    env = crafter.Env(seed=args.seed)
    online, target = QNetwork().to(device), QNetwork().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.learning_rate, eps=1.5e-4)
    replay = ReplayBuffer(args.buffer_size)
    nstep = NStepAccumulator(args.n_step, args.gamma)
    rng = np.random.default_rng(args.seed)
    obs = env.reset()
    episode_reward, episode_length, episode_id = 0.0, 0, 0
    losses, q_values = [], []
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        epsilon = 1 - (1 - args.epsilon_end) * min(step / args.epsilon_decay_steps, 1)
        if step <= args.learning_starts or rng.random() < epsilon:
            action = int(rng.integers(17))
        else:
            with torch.no_grad():
                action = int(online(torch.as_tensor(obs[None], device=device)).argmax().item())
        next_obs, reward, done, info = env.step(action)
        # Death is terminal; the 10k-step time limit is a truncation.
        bootstrap_mask = float(info.get("discount", 0.0 if done else 1.0))
        for item in nstep.add(obs, action, reward, next_obs, done, bootstrap_mask):
            replay.add(*item)
        episode_reward += float(reward)
        episode_length += 1
        obs = next_obs
        if done:
            episode_id += 1
            record = {"step": step, "episode": episode_id, "reward": episode_reward,
                      "length": episode_length}
            record.update({f"achievement_{key}": int(value)
                           for key, value in info["achievements"].items()})
            write_episode(run / "episodes.jsonl", record)
            obs = env.reset()
            episode_reward, episode_length = 0.0, 0
        if step > args.learning_starts and step % args.train_frequency == 0 and replay.size >= args.batch_size:
            loss, q_mean = optimize(online, target, optimizer, replay, args, rng, device)
            losses.append(loss)
            q_values.append(q_mean)
        if step % args.target_frequency == 0:
            target.load_state_dict(online.state_dict())
        if step % 1000 == 0:
            row = {"step": step, "loss": np.mean(losses) if losses else "",
                   "q_mean": np.mean(q_values) if q_values else "", "epsilon": epsilon,
                   "steps_per_second": step / max(time.monotonic() - start, 1e-6)}
            write_scalar(run / "scalars.csv", row)
            print(json.dumps(row), flush=True)
            losses.clear()
            q_values.clear()
        if step % args.checkpoint_frequency == 0 or step == args.steps:
            torch.save({"model": online.state_dict(), "config": vars(args), "step": step},
                       run / f"checkpoint_{step}.pt")
    print(f"Training complete: {run}", flush=True)


if __name__ == "__main__":
    main()
