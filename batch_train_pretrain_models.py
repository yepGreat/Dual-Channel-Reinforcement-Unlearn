import subprocess
import os

SEEDS = range(8, 99, 10)
TRAIN_STEPS_RANGE = range(20, 41, 2)

total = 0
for seed in SEEDS:
    for train_steps in TRAIN_STEPS_RANGE:
        test_steps = train_steps + 10
        total += 1

        print(f"[{total}/110] Training seed={seed}, steps={train_steps}")

        cmd = [
            "python", "SPU_script.py",
            "--train_normal",
            "--seed", str(seed),
            "--train_max_steps", str(train_steps),
            "--test_max_steps", str(test_steps),
            "--game_type", "grid_world",
            "--n_maps", "50",
        ]

        subprocess.run(cmd)

print(f"\nCompleted: {total} models trained")