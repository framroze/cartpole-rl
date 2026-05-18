"""
CartPole RL — Entry Point

Usage:
    python main.py train    Train the PPO policy
    python main.py view     View policy in MuJoCo viewer
"""
import os
import sys

os.environ['MUJOCO_GL'] = 'egl'


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "train":
        from src.training.train import train, evaluate
        model = train()
        evaluate(model)

    elif command == "view":
        from src.visualization.viewer import run
        run()

    else:
        print(f"Unknown command: '{command}'")
        print("Usage: python main.py [train|view]")
        sys.exit(1)


if __name__ == "__main__":
    main()
