import subprocess
import os


TRAIN_STEPS_RANGE = range(20, 31, 2)

total = 0
seed = 28

for train_steps in TRAIN_STEPS_RANGE:
    test_steps = train_steps + 5
    total += 1

    print(f"[{total}/{len(TRAIN_STEPS_RANGE)}] Training seed={seed}, steps={train_steps}")

    cmd = [
        "python", "base_line_model.py",
        "--train_retrain_full",
        "--seed", str(seed),
        "--train_max_steps", str(train_steps),
        "--test_max_steps", str(test_steps),
        "--game_type", "grid_world",
        "--n_maps", "50",
    ]

    subprocess.run(cmd)

print(f"\nCompleted: {total} models trained")