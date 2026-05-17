import os
import shutil
import json
import torch
import numpy as np
from utils import *
from init import *
from Hyperparameter_control import *

SEEDS = range(8, 99, 10)
TRAIN_RANGE = range(20, 41, 2)
GAME_TYPE = "grid_world"
N_MAPS = 50
STANDARD_EPOCH = 50
size = 8
n_ob = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(seed, train_steps, test_steps):
    try:
        model_dir = f"./model/{GAME_TYPE}/Pretrain_model_standard_epoch_{STANDARD_EPOCH}/"
        map_file = f"map_data/{GAME_TYPE}/{N_MAPS}_maps_seed_{seed}.json"
        model_name = f"{GAME_TYPE}_Pretrain_Model_seed_{seed}_train_steps_{train_steps}.pkl"

        if not os.path.exists(os.path.join(model_dir, model_name)):
            return None

        env, agent, _ = init_module(map_file, model_name, model_dir, size, n_ob, [3], device)

        total_steps = 0
        map_count = 0
        all_step = 0
        steps = 0
        map_id = -1

        for i in range(STANDARD_EPOCH * N_MAPS):
            all_step += steps

            if i % STANDARD_EPOCH == 0:
                if map_id >= 0:
                    total_steps += all_step / STANDARD_EPOCH
                    map_count += 1
                map_id += 1
                state = env.reset(map_index=map_id, game_type=GAME_TYPE)
                all_step = 0
            else:
                state = env.reset(map_index=map_id, game_type=GAME_TYPE)

            done = False
            steps = 0
            while not done and steps < test_steps:
                action, _ = agent.get_action(state, epsilon=0.05)
                state, _, done, _ = env.step(action)
                steps += 1

        if map_id >= 0:
            total_steps += all_step / STANDARD_EPOCH
            map_count += 1

        return total_steps / map_count if map_count > 0 else None

    except Exception as e:
        return None


def main():
    print("Finding best pretrain models...")

    all_results = []

    for seed in SEEDS:
        print(f"Seed {seed}:", end=" ")

        results = []
        for train_steps in TRAIN_RANGE:
            test_steps = train_steps + 10
            avg_steps = test_model(seed, train_steps, test_steps)
            if avg_steps:
                results.append((train_steps, avg_steps))
                print(f"{train_steps}→{avg_steps:.1f}", end=" ")

        if results:
            best_train, best_avg = min(results, key=lambda x: x[1])
            print(f"Best: {best_train} ({best_avg:.2f})")

            src = f"./model/{GAME_TYPE}/Pretrain_model_standard_epoch_{STANDARD_EPOCH}/" \
                  f"{GAME_TYPE}_Pretrain_Model_seed_{seed}_train_steps_{best_train}.pkl"
            dst = f"./model/{GAME_TYPE}/Pretrain_model_standard_epoch_{STANDARD_EPOCH}/" \
                  f"{GAME_TYPE}_Pretrain_Model_seed_{seed}_best.pkl"
            shutil.copy2(src, dst)

            all_results.append({
                'seed': seed,
                'best_train_steps': best_train,
                'best_avg_steps': best_avg,
                'all_results': [(t, s) for t, s in results]
            })
        else:
            print("No valid results")

    with open("best_models_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\nSummary:")
    for r in all_results:
        print(f"  Seed {r['seed']:2d}: train={r['best_train_steps']:2d}, steps={r['best_avg_steps']:5.2f}")
    print("Done")


if __name__ == "__main__":
    main()