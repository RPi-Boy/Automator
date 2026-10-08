"""Train SAC to reach random workspace targets within a 1 cm success radius (ArmReachEnv).

  python scripts/train_sac_reach.py --timesteps 400000 --out models/sac_reach
Evaluates the best checkpoint on fresh seeds and writes models/sac_reach_metrics.json
(success rate, final distance, steps to success) so reports quote measured numbers only.
Runs on CPU or a Colab GPU: the simulator is the bottleneck, so a GPU only speeds up the network updates.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.reach_env import ArmReachEnv  # noqa: E402


def evaluate(model, radius, episodes, seed0=10_000):
    env = ArmReachEnv(success_radius=radius)
    finals, steps, ok = [], [], 0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
            done = term or trunc
        finals.append(info["dist"] * 1000)
        ok += int(info["success"])
        if info["success"]:
            steps.append(env.steps)
    env.close()
    return {"episodes": episodes, "success_rate": ok / episodes, "final_error_mm_mean": float(np.mean(finals)),
            "final_error_mm_p95": float(np.percentile(finals, 95)),
            "steps_to_success_mean": float(np.mean(steps)) if steps else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=400_000)
    ap.add_argument("--radius", type=float, default=0.01)
    ap.add_argument("--out", default="models/sac_reach")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=100)
    args = ap.parse_args()

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    os.makedirs(args.out, exist_ok=True)
    train_env = Monitor(ArmReachEnv(success_radius=args.radius, seed=args.seed))
    eval_env = Monitor(ArmReachEnv(success_radius=args.radius, seed=args.seed + 1))
    cb = EvalCallback(eval_env, best_model_save_path=args.out, log_path=os.path.join(args.out, "logs"),
                      eval_freq=10_000, n_eval_episodes=20, deterministic=True)
    model = SAC("MlpPolicy", train_env, learning_rate=3e-4, buffer_size=300_000, batch_size=256, gamma=0.98,
                learning_starts=5_000, ent_coef="auto", seed=args.seed, verbose=1,
                policy_kwargs=dict(net_arch=[256, 256]))
    model.learn(total_timesteps=args.timesteps, callback=cb)
    model.save(os.path.join(args.out, "final_model"))

    best = SAC.load(os.path.join(args.out, "best_model"))
    metrics = {"success_radius_m": args.radius, "timesteps": args.timesteps,
               "best_checkpoint": evaluate(best, args.radius, args.eval_episodes)}
    print(json.dumps(metrics, indent=2))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
