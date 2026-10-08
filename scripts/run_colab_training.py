import argparse
import os
import sys
import torch
import pybullet as p
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.arm_env import ArmGraspEnv


def main():
    parser = argparse.ArgumentParser(description="Google Colab Headless SAC Training Launcher")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps")
    parser.add_argument("--save-path", type=str, default="models/best_sac_policy.zip", help="Path for policy checkpoint zip")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("[Colab Launcher] Initializing Soft Actor-Critic (SAC) Training on Colab Platform...")
    print(f"  - Target Timesteps: {args.timesteps}")
    print(f"  - Model Checkpoint: {args.save_path}")
    print(f"  - PyTorch Version: {torch.__version__}")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - Active GPU Device: {torch.cuda.get_device_name(0)}")

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("[Colab Launcher Error] stable-baselines3 not installed. Please run 'pip install stable-baselines3 pybullet gymnasium'")
        sys.exit(1)

    train_env = Monitor(ArmGraspEnv(render_mode=None))
    eval_env = Monitor(ArmGraspEnv(render_mode=None))

    save_dir = os.path.dirname(os.path.abspath(args.save_path))
    os.makedirs(save_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=os.path.join(save_dir, "logs"),
        eval_freq=2500,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=200000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        verbose=1,
        seed=args.seed,
    )

    print("[Colab Launcher] Starting SAC policy training with EvalCallback anti-overfitting protection...")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)

    model.save(args.save_path)
    print(f"[Colab Launcher] Training complete! Saved trained SAC policy to '{args.save_path}'.")


if __name__ == "__main__":
    main()
