import subprocess
import os
import time
import json
import itertools
import argparse


def create_parameter_grid():
    """Create parameter grid for hyperparameter search.

    DFRU combines SPU (reward_scale) and SSP (poison_intensity) with unlearn_epoch:
    - unlearn_epoch: 5 to 40, step 5 -> [5, 10, 15, 20, 25, 30, 35, 40]
    - reward_scale: 1 to 10, step 2 -> [1, 3, 5, 7, 9]
    - poison_intensity: 10 to 80, step 10 -> [10, 20, 30, 40, 50, 60, 70, 80]
    """
    unlearn_epochs = range(5, 41, 5)  # 5, 10, 15, 20, 25, 30, 35, 40
    reward_scales = range(1, 11, 2)  # 1, 3, 5, 7, 9
    poison_intensities = range(10, 81, 10)  # 10, 20, 30, 40, 50, 60, 70, 80
    return list(itertools.product(unlearn_epochs, reward_scales, poison_intensities))


def run_single_training(unlearn_epoch, reward_scale, poison_intensity, base_args, tem_dfru_max_steps):
    """Run single training experiment with specified hyperparameters."""
    cmd = [
        'python', 'dfru.py',
        '--unlearn_epoch', str(unlearn_epoch),
        '--reward_scale', str(reward_scale),
        '--poison_intensity', str(poison_intensity),
        '--game_type', base_args.get('game_type', 'grid_world'),
        '--n_maps', str(base_args.get('n_maps', 50)),
        '--unlearn_maps_ratio', str(base_args.get('unlearn_maps_ratio', 30)),
        '--train_unlearn',
        '--tem_dfru_max_steps', str(tem_dfru_max_steps),
    ]

    unlearn_obstacles = base_args.get('unlearn_obstacles_type', [3])
    for obs_type in unlearn_obstacles:
        cmd.extend(['--unlearn_obstacles_type', str(obs_type)])

    print(f"Training: epoch={unlearn_epoch}, reward_scale={reward_scale}, poison_intensity={poison_intensity}")
    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        training_time = time.time() - start_time
        print(f"Completed in {training_time:.2f}s")

        return {
            'status': 'success',
            'unlearn_epoch': unlearn_epoch,
            'reward_scale': reward_scale,
            'poison_intensity': poison_intensity,
            'training_time': training_time,
            'stdout': result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout
        }

    except subprocess.CalledProcessError as e:
        print(f"Failed: {e.stderr}")
        return {
            'status': 'failed',
            'unlearn_epoch': unlearn_epoch,
            'reward_scale': reward_scale,
            'poison_intensity': poison_intensity,
            'error': str(e),
            'stderr': e.stderr
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'status': 'failed',
            'unlearn_epoch': unlearn_epoch,
            'reward_scale': reward_scale,
            'poison_intensity': poison_intensity,
            'error': str(e)
        }


def main():
    """Batch training DFRU models with different hyperparameters."""
    parser = argparse.ArgumentParser(description='Batch DFRU model training')
    parser.add_argument('--tem_dfru_max_steps', type=int, default=25,
                        help='Maximum steps for evaluation (default: 25)')
    args = parser.parse_args()

    base_args = {
        'standard_epoch': 50,
        'game_type': 'grid_world',
        'n_maps': 50,
        'seed': 28,
        'unlearn_maps_ratio': 30,
        'unlearn_obstacles_type': [3],
    }

    param_grid = create_parameter_grid()
    total = len(param_grid)
    print(f"Starting {total} experiments")
    print(f"Grid: 8 unlearn_epochs x 5 reward_scales x 8 poison_intensities = {total} combinations")
    print(f"  - unlearn_epoch: 5-40 (step 5)")
    print(f"  - reward_scale: 1-9 (step 2)")
    print(f"  - poison_intensity: 10-80 (step 10)")

    all_results = []
    start_time = time.time()

    for idx, (unlearn_epoch, reward_scale, poison_intensity) in enumerate(param_grid, 1):
        print(f"\n[{idx}/{total}] ", end='')
        result = run_single_training(unlearn_epoch, reward_scale, poison_intensity, base_args, args.tem_dfru_max_steps)
        all_results.append(result)
        time.sleep(1)

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/3600:.2f}h")

    # Save results summary
    results_dir = f"./result/grid_world/DFRU_Model_standard_epoch_{base_args['standard_epoch']}/tem_dfru_max_steps_{args.tem_dfru_max_steps}"
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, "batch_training_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'total_experiments': total,
            'total_time_hours': total_time/3600,
            'results': all_results
        }, f, indent=2)
    print(f"Results saved to {results_file}")

    return all_results


if __name__ == "__main__":
    results = main()
