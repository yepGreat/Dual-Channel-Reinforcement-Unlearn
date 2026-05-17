from datetime import datetime
from init import *
from Hyperparameter_control import *
import pandas as pd


def check_model_exists(model_path):
    return os.path.exists(model_path)


def verify_pretrain_model(model_info, config):
    try:
        pretrain_model_dir = model_info['model_dir']
        pretrain_model_name = model_info['model_name']
        pretrain_model_path = model_info['model_path']
        full_map_dir = model_info['full_map_dir']
        full_map_name = model_info['full_map_name']
        full_map_path = model_info['full_map_path']
        seed = model_info['seed']
        seed_control(seed)

        env_original, agent_original, _ = init_module(
            map_file_path=full_map_path,
            filename=pretrain_model_name,
            model_dir=pretrain_model_dir,
            size=config['size'],
            n_ob=config['n_ob'],
            unlearn_obstacles_type=None,
            device=config['device']
        )

        original_results = verify_single_model_with_collision_stats(
            env_original,
            agent_original,
            config['game_type'],
            config['n_maps'],
            config['max_steps_test'],
            config['test_episodes_per_map']
        )

        result = {
            'model_type': 'pretrain',
            'seed': seed,
            'standard_epoch': config['standard_epoch'],
            'status': 'success',
        }

        for category, metrics in original_results.items():
            result[f'original_{category}_avg_steps'] = metrics['avg_steps']
            result[f'original_{category}_avg_reward'] = metrics['avg_reward']
            result[f'original_{category}_collision_summary'] = metrics['collision_summary']

        return result

    except Exception as e:
        return {
            'model_type': 'pretrain',
            'seed': seed,
            'standard_epoch': config['standard_epoch'],
            'status': 'error',
            'error_message': str(e),
            'verification_time': 0
        }


def verify_single_model_with_collision_stats(env, agent, game_type, n_maps, max_steps_test, test_episodes_per_map):
    results = {'all': []}
    collision_stats = {'all': {'type_1': 0, 'type_2': 0, 'type_3': 0, 'type_4': 0, 'type_boundary': 0}}

    for map_id in range(n_maps):
        map_steps_list = []
        map_rewards_list = []
        map_collision_counts = {'type_1': 0, 'type_2': 0, 'type_3': 0, 'type_4': 0, 'type_boundary': 0}

        for episode in range(test_episodes_per_map):
            state = env.reset(map_index=map_id, game_type=game_type)
            done = False
            steps = 0
            total_reward = 0

            while not done and steps < max_steps_test:
                action, _ = agent.get_action(state, epsilon=0.05)

                x, y = env.agent_location
                if action == 0:
                    expected_location = (x, y + 1)
                elif action == 1:
                    expected_location = (x, y - 1)
                elif action == 2:
                    expected_location = (x - 1, y)
                elif action == 3:
                    expected_location = (x + 1, y)
                else:
                    expected_location = (x, y)

                collision_type = check_collision_type(expected_location, env.obstacles_location, env.size)

                if collision_type == 1:
                    map_collision_counts['type_1'] += 1
                elif collision_type == 2:
                    map_collision_counts['type_2'] += 1
                elif collision_type == 3:
                    map_collision_counts['type_3'] += 1
                elif collision_type == 4:
                    map_collision_counts['type_4'] += 1
                elif collision_type == 5:
                    map_collision_counts['type_boundary'] += 1

                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                steps += 1

            map_steps_list.append(steps)
            map_rewards_list.append(total_reward)

        avg_steps = np.mean(map_steps_list)
        avg_reward = np.mean(map_rewards_list)
        results['all'].append([avg_steps, avg_reward])

        for collision_type in map_collision_counts:
            collision_stats['all'][collision_type] += map_collision_counts[collision_type]

    summary_results = {}
    for category in ['all']:
        summary_results[category] = calculate_summary(results[category], collision_stats[category])

    return summary_results


def calculate_summary(data, collision_data):
    if not data:
        return {'avg_steps': 0, 'avg_reward': 0, 'collision_summary': '0&0&0&0&0'}

    avg_steps = np.mean([item[0] for item in data])
    avg_reward = np.mean([item[1] for item in data])
    collision_summary = f"{collision_data['type_1']}&{collision_data['type_2']}&{collision_data['type_3']}&{collision_data['type_4']}&{collision_data['type_boundary']}"

    return {
        'avg_steps': format_number(avg_steps),
        'avg_reward': format_number(avg_reward),
        'collision_summary': collision_summary
    }


def format_number(value, decimals=3):
    if isinstance(value, (int, float)):
        return round(float(value), decimals)
    return value


def check_collision_type(agent_location, obstacles_dict, grid_size=10):
    x, y = agent_location

    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        return 5

    if agent_location in obstacles_dict['type_1']:
        return 1
    elif agent_location in obstacles_dict['type_2']:
        return 2
    elif agent_location in obstacles_dict['type_3']:
        return 3
    elif agent_location in obstacles_dict['type_4']:
        return 4
    elif agent_location in obstacles_dict['type_boundary']:
        return 5
    else:
        return 0


def save_verification_summary(results, config):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df = pd.DataFrame(results)

    column_order = [
        'model_type', 'seed', 'standard_epoch', 'status', 'verification_time',
        'original_all_avg_steps', 'original_all_avg_reward', 'original_all_collision_summary',
    ]

    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]

    csv_file = os.path.join(config['result_dir'], f"result.csv")
    df.to_csv(csv_file, index=False, float_format='%.3f')


def main():
    SEEDS = range(8, 99, 10)

    config = {
        'game_type': 'grid_world',
        'n_maps': 50,
        'standard_epoch': 50,
        'size': 10,
        'n_ob': 10,
        'test_episodes_per_map': 25,
        'max_steps_test': 40,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    config['result_dir'] = f"./result/{config['game_type']}/get_best_pretrain_model/"
    init_files(config['result_dir'])

    print("Step 1: Checking models...")

    pretrain_models_info = []
    for seed in SEEDS:
        model_dir = f"./model/{config['game_type']}/Pretrain_model_standard_epoch_{config['standard_epoch']}/"
        model_name = f"{config['game_type']}_Pretrain_Model_seed_{seed}_best.pkl"
        model_path = os.path.join(model_dir, model_name)

        full_map_dir = f"./map_data/{config['game_type']}/"
        full_map_name = f"{config['n_maps']}_maps_seed_{seed}.json"
        full_map_path = os.path.join(full_map_dir, full_map_name)

        model_exists = check_model_exists(model_path)
        full_map_exists = check_model_exists(full_map_path)

        pretrain_models_info.append({
            'seed': seed,
            'model_dir': model_dir,
            'model_name': model_name,
            'model_path': model_path,
            'model_exists': model_exists,
            'full_map_dir': full_map_dir,
            'full_map_name': full_map_name,
            'full_map_path': full_map_path,
            'full_map_exists': full_map_exists,
        })

    pretrain_exists_count = sum(1 for m in pretrain_models_info if m['model_exists'])
    print(f"Models found: {pretrain_exists_count}/{len(SEEDS)}")

    if pretrain_exists_count != len(SEEDS):
        print("Error: Models not found")
        return None

    print("\nStep 2: Verifying models...")

    all_results = []
    successful_count = 0
    error_count = 0

    for model_info in pretrain_models_info:
        result = verify_pretrain_model(model_info, config)
        print(f"Seed {model_info['seed']} -> {result['status']}")

        if result:
            all_results.append(result)
            if result['status'] == 'success':
                successful_count += 1
            else:
                error_count += 1
        else:
            error_count += 1

    print("\nStep 3: Saving results...")
    save_verification_summary(all_results, config)
    print("Done")

    return all_results




if __name__ == "__main__":
    main()