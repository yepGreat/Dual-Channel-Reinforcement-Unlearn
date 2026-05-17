import time
import pandas as pd
import itertools
from init import *


def format_number(value, decimals=3):
    """Format number to specified decimal places."""
    if isinstance(value, (int, float)):
        return round(float(value), decimals)
    return value


def create_parameter_grid(tem_dfru_max_steps):
    """Create parameter grid for DFRU verification.

    DFRU has three parameters:
    - unlearn_epoch: 5 to 40, step 5
    - reward_scale: 1 to 9, step 2
    - poison_intensity: 10 to 80, step 10
    """
    unlearn_epochs = range(5, 41, 5)
    reward_scales = range(1, 11, 2)
    poison_intensities = range(10, 81, 10)
    param_grid = [(epoch, scale, intensity, tem_dfru_max_steps)
                  for epoch, scale, intensity in itertools.product(unlearn_epochs, reward_scales, poison_intensities)]
    return param_grid


def check_collision_type(agent_location, obstacles_dict, grid_size=10):
    """Check collision type at agent location."""
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


def check_near_target_obstacle(agent_location, obstacles_dict, unlearn_obstacles_type, grid_size=10):
    """
    Check if there are target obstacle types within 3x3 range of agent location.

    Args:
        agent_location: Current agent position (x, y)
        obstacles_dict: Dictionary containing all obstacle positions by type
        unlearn_obstacles_type: List of obstacle types to check (e.g., [3])
        grid_size: Size of the grid

    Returns:
        bool: True if any target obstacle is within 3x3 range, False otherwise
    """
    x, y = agent_location

    # Collect all target obstacles
    target_obstacles = set()
    for obs_type in unlearn_obstacles_type:
        type_key = f'type_{obs_type}'
        if type_key in obstacles_dict:
            target_obstacles.update(obstacles_dict[type_key])

    # Check 3x3 range around agent (including agent's position)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            check_pos = (x + dx, y + dy)
            if check_pos in target_obstacles:
                return True

    return False


def verify_single_model_with_full_metrics(env, agent, game_type, n_maps, max_steps_test,
                                          test_episodes_per_map, unlearn_map_flags,
                                          unlearn_obstacles_type):
    """Verify model and collect full metrics: steps, rewards, collisions, and perplexity."""
    unlearn_indices = [i for i, flag in enumerate(unlearn_map_flags) if flag]
    retain_indices = [i for i, flag in enumerate(unlearn_map_flags) if not flag]

    results = {
        'all': [],
        'unlearn': [],
        'retain': []
    }

    collision_stats = {
        'all': {'type_1': 0, 'type_2': 0, 'type_3': 0, 'type_4': 0, 'type_boundary': 0},
        'unlearn': {'type_1': 0, 'type_2': 0, 'type_3': 0, 'type_4': 0, 'type_boundary': 0},
        'retain': {'type_1': 0, 'type_2': 0, 'type_3': 0, 'type_4': 0, 'type_boundary': 0}
    }

    perplexity_stats = {
        'all': [],
        'unlearn': [],
        'retain': []
    }

    for map_id in range(n_maps):
        map_steps_list = []
        map_rewards_list = []
        map_perplexity_list = []
        map_collision_counts = {
            'type_1': 0, 'type_2': 0, 'type_3': 0, 'type_4': 0, 'type_boundary': 0
        }

        for episode in range(test_episodes_per_map):
            state = env.reset(map_index=map_id, game_type=game_type)
            done = False
            steps = 0
            total_reward = 0
            episode_perplexity_list = []

            while not done and steps < max_steps_test:
                action, perplexity = agent.get_action(state, epsilon=0.01)

                # Only record perplexity when target obstacles are within 3x3 range
                if check_near_target_obstacle(env.agent_location, env.obstacles_location,
                                              unlearn_obstacles_type, env.size):
                    episode_perplexity_list.append(perplexity)

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
            if episode_perplexity_list:
                map_perplexity_list.append(np.mean(episode_perplexity_list))

        avg_steps = np.mean(map_steps_list)
        avg_reward = np.mean(map_rewards_list)
        avg_perplexity = np.mean(map_perplexity_list) if map_perplexity_list else 0

        results['all'].append([avg_steps, avg_reward, avg_perplexity])
        perplexity_stats['all'].append(avg_perplexity)

        if map_id in unlearn_indices:
            results['unlearn'].append([avg_steps, avg_reward, avg_perplexity])
            perplexity_stats['unlearn'].append(avg_perplexity)
            for collision_type in map_collision_counts:
                collision_stats['unlearn'][collision_type] += map_collision_counts[collision_type]
        else:
            results['retain'].append([avg_steps, avg_reward, avg_perplexity])
            perplexity_stats['retain'].append(avg_perplexity)
            for collision_type in map_collision_counts:
                collision_stats['retain'][collision_type] += map_collision_counts[collision_type]

        for collision_type in map_collision_counts:
            collision_stats['all'][collision_type] += map_collision_counts[collision_type]

    def calculate_summary(data, collision_data, perplexity_list):
        if not data:
            return {
                'avg_steps': 0,
                'avg_reward': 0,
                'collision_summary': '0&0&0&0&0',
                'unlearn_ratio': '0.00',
                'avg_perplexity': 0
            }

        avg_steps = np.mean([item[0] for item in data])
        avg_reward = np.mean([item[1] for item in data])
        avg_perplexity = np.mean(perplexity_list) if perplexity_list else 0
        collision_summary = f"{collision_data['type_1']}&{collision_data['type_2']}&{collision_data['type_3']}&{collision_data['type_4']}&{collision_data['type_boundary']}"

        total_collisions = (collision_data['type_1'] + collision_data['type_2'] +
                           collision_data['type_3'] + collision_data['type_4'] +
                           collision_data['type_boundary'])

        if total_collisions > 0:
            unlearn_collision_count = sum(collision_data[f'type_{t}'] for t in unlearn_obstacles_type)
            unlearn_ratio = (unlearn_collision_count / total_collisions) * 100
        else:
            unlearn_ratio = 0.0

        return {
            'avg_steps': format_number(avg_steps),
            'avg_reward': format_number(avg_reward),
            'collision_summary': collision_summary,
            'unlearn_ratio': format_number(unlearn_ratio, 2),
            'avg_perplexity': format_number(avg_perplexity)
        }

    summary_results = {}
    for category in ['all', 'unlearn', 'retain']:
        summary_results[category] = calculate_summary(
            results[category],
            collision_stats[category],
            perplexity_stats[category]
        )

    return summary_results


def verify_Pretrain_model(seed, standard_epoch, max_steps_test, config):
    """Verify single Pretrain model."""
    print(f"  Pretrain: seed={seed}, epoch={standard_epoch}, test_steps={max_steps_test}")

    try:
        Pretrain_model_dir = f"./model/{config['game_type']}/Pretrain_model_standard_epoch_{standard_epoch}/"
        map_file = f"map_data/{config['game_type']}/{config['n_maps']}_maps_seed_{seed}.json"
        clean_map_file = f"map_data/{config['game_type']}/{config['n_maps']}_clean_maps_without_type_{'_'.join(map(str, config['unlearn_obstacles_type']))}_ratio_{config['unlearn_maps_ratio']}_seed_{seed}.json"
        unlearn_flags_file = f"map_data/{config['game_type']}/unlearn_flags_ratio_{config['unlearn_maps_ratio']}_seed_{seed}.pkl"

        if not all(os.path.exists(f) for f in [map_file, clean_map_file, unlearn_flags_file]):
            return None

        Pretrain_model_file = f"{config['game_type']}_Pretrain_Model_seed_{seed}_best.pkl"
        Pretrain_model_path = os.path.join(Pretrain_model_dir, Pretrain_model_file)

        if not os.path.exists(Pretrain_model_path):
            return None

        original_seed = config.get('current_seed')
        seed_control(seed)

        with open(unlearn_flags_file, 'rb') as f:
            unlearn_data = pickle.load(f)
            unlearn_map_flags = unlearn_data['unlearn_map_flags']

        start_time = time.time()

        env_original, agent_original, _ = init_module(
            map_file_path=map_file,
            filename=Pretrain_model_file,
            model_dir=Pretrain_model_dir,
            size=config['size'],
            n_ob=config['n_ob'],
            unlearn_obstacles_type=config['unlearn_obstacles_type'],
            device=config['device']
        )

        original_results = verify_single_model_with_full_metrics(
            env_original, agent_original, config['game_type'], config['n_maps'],
            max_steps_test, config['test_episodes_per_map'],
            unlearn_map_flags, config['unlearn_obstacles_type']
        )

        env_clean, agent_clean, _ = init_module(
            map_file_path=clean_map_file,
            filename=Pretrain_model_file,
            model_dir=Pretrain_model_dir,
            size=config['size'],
            n_ob=config['n_ob'],
            unlearn_obstacles_type=config['unlearn_obstacles_type'],
            device=config['device']
        )

        clean_results = verify_single_model_with_full_metrics(
            env_clean, agent_clean, config['game_type'], config['n_maps'],
            max_steps_test, config['test_episodes_per_map'],
            unlearn_map_flags, config['unlearn_obstacles_type']
        )

        end_time = time.time()

        if original_seed is not None:
            seed_control(original_seed)

        result = {
            'model_type': 'Pretrain',
            'seed': seed,
            'standard_epoch': standard_epoch,
            'unlearn_epoch': 'N/A',
            'reward_scale': 'N/A',
            'poison_intensity': 'N/A',
            'tem_dfru_max_steps': 'N/A',
            'status': 'success',
            'verification_time': format_number(end_time - start_time, 2)
        }

        for category in ['all', 'unlearn', 'retain']:
            result[f'original_{category}_avg_steps'] = original_results[category]['avg_steps']
            result[f'original_{category}_avg_reward'] = original_results[category]['avg_reward']
            result[f'original_{category}_collision_summary'] = original_results[category]['collision_summary']
            result[f'original_{category}_unlearn_ratio'] = original_results[category]['unlearn_ratio']
            result[f'original_{category}_avg_perplexity'] = original_results[category]['avg_perplexity']

            result[f'clean_{category}_avg_steps'] = clean_results[category]['avg_steps']
            result[f'clean_{category}_avg_reward'] = clean_results[category]['avg_reward']
            result[f'clean_{category}_collision_summary'] = clean_results[category]['collision_summary']
            result[f'clean_{category}_unlearn_ratio'] = clean_results[category]['unlearn_ratio']
            result[f'clean_{category}_avg_perplexity'] = clean_results[category]['avg_perplexity']

        print(f"    Done in {end_time - start_time:.2f}s")

        return result

    except Exception as e:
        print(f"    Error: {str(e)}")
        return {
            'model_type': 'Pretrain',
            'seed': seed,
            'standard_epoch': standard_epoch,
            'unlearn_epoch': 'N/A',
            'reward_scale': 'N/A',
            'poison_intensity': 'N/A',
            'tem_dfru_max_steps': 'N/A',
            'status': 'error',
            'error_message': str(e),
            'verification_time': 0
        }


def verify_Nontranslfs_model(seed, standard_epoch, unlearn_maps_ratio, max_steps_test, config):
    """Verify single Nontranslfs model."""
    print(f"  Nontranslfs: seed={seed}, epoch={standard_epoch}, ratio={unlearn_maps_ratio}, test_steps={max_steps_test}")

    try:
        Nontranslfs_model_dir = f"./model/{config['game_type']}/Nontranslfs_Model_standard_epoch_{standard_epoch}/"
        map_file = f"map_data/{config['game_type']}/{config['n_maps']}_maps_seed_{seed}.json"
        clean_map_file = f"map_data/{config['game_type']}/{config['n_maps']}_clean_maps_without_type_{'_'.join(map(str, config['unlearn_obstacles_type']))}_ratio_{unlearn_maps_ratio}_seed_{seed}.json"
        unlearn_flags_file = f"map_data/{config['game_type']}/unlearn_flags_ratio_{unlearn_maps_ratio}_seed_{seed}.pkl"

        if not all(os.path.exists(f) for f in [map_file, clean_map_file, unlearn_flags_file]):
            return None

        Nontranslfs_model_file = f"{config['game_type']}_Nontranslfs_Model_ratio_{unlearn_maps_ratio}_types_{'_'.join(map(str, config['unlearn_obstacles_type']))}_best.pkl"
        Nontranslfs_model_path = os.path.join(Nontranslfs_model_dir, Nontranslfs_model_file)

        if not os.path.exists(Nontranslfs_model_path):
            return None

        original_seed = config.get('current_seed')
        seed_control(seed)

        with open(unlearn_flags_file, 'rb') as f:
            unlearn_data = pickle.load(f)
            unlearn_map_flags = unlearn_data['unlearn_map_flags']

        start_time = time.time()

        env_original, agent_original, _ = init_module(
            map_file_path=map_file,
            filename=Nontranslfs_model_file,
            model_dir=Nontranslfs_model_dir,
            size=config['size'],
            n_ob=config['n_ob'],
            unlearn_obstacles_type=config['unlearn_obstacles_type'],
            device=config['device']
        )

        original_results = verify_single_model_with_full_metrics(
            env_original, agent_original, config['game_type'], config['n_maps'],
            max_steps_test, config['test_episodes_per_map'],
            unlearn_map_flags, config['unlearn_obstacles_type']
        )

        env_clean, agent_clean, _ = init_module(
            map_file_path=clean_map_file,
            filename=Nontranslfs_model_file,
            model_dir=Nontranslfs_model_dir,
            size=config['size'],
            n_ob=config['n_ob'],
            unlearn_obstacles_type=config['unlearn_obstacles_type'],
            device=config['device']
        )

        clean_results = verify_single_model_with_full_metrics(
            env_clean, agent_clean, config['game_type'], config['n_maps'],
            max_steps_test, config['test_episodes_per_map'],
            unlearn_map_flags, config['unlearn_obstacles_type']
        )

        end_time = time.time()

        if original_seed is not None:
            seed_control(original_seed)

        result = {
            'model_type': 'Nontranslfs',
            'seed': seed,
            'standard_epoch': standard_epoch,
            'unlearn_epoch': 'N/A',
            'reward_scale': 'N/A',
            'poison_intensity': 'N/A',
            'tem_dfru_max_steps': 'N/A',
            'status': 'success',
            'verification_time': format_number(end_time - start_time, 2)
        }

        for category in ['all', 'unlearn', 'retain']:
            result[f'original_{category}_avg_steps'] = original_results[category]['avg_steps']
            result[f'original_{category}_avg_reward'] = original_results[category]['avg_reward']
            result[f'original_{category}_collision_summary'] = original_results[category]['collision_summary']
            result[f'original_{category}_unlearn_ratio'] = original_results[category]['unlearn_ratio']
            result[f'original_{category}_avg_perplexity'] = original_results[category]['avg_perplexity']

            result[f'clean_{category}_avg_steps'] = clean_results[category]['avg_steps']
            result[f'clean_{category}_avg_reward'] = clean_results[category]['avg_reward']
            result[f'clean_{category}_collision_summary'] = clean_results[category]['collision_summary']
            result[f'clean_{category}_unlearn_ratio'] = clean_results[category]['unlearn_ratio']
            result[f'clean_{category}_avg_perplexity'] = clean_results[category]['avg_perplexity']

        print(f"    Done in {end_time - start_time:.2f}s")

        return result

    except Exception as e:
        print(f"    Error: {str(e)}")
        return {
            'model_type': 'Nontranslfs',
            'seed': seed,
            'standard_epoch': standard_epoch,
            'unlearn_epoch': 'N/A',
            'reward_scale': 'N/A',
            'poison_intensity': 'N/A',
            'tem_dfru_max_steps': 'N/A',
            'status': 'error',
            'error_message': str(e),
            'verification_time': 0
        }


def verify_Retrain_model(seed, standard_epoch, unlearn_maps_ratio, max_steps_test, config):
    """Verify single Retrain model."""
    print(f"  Retrain: seed={seed}, epoch={standard_epoch}, ratio={unlearn_maps_ratio}, test_steps={max_steps_test}")

    try:
        Retrain_model_dir = f"./model/{config['game_type']}/Retrain_Model_standard_epoch_{standard_epoch}/"
        map_file = f"map_data/{config['game_type']}/{config['n_maps']}_maps_seed_{seed}.json"
        clean_map_file = f"map_data/{config['game_type']}/{config['n_maps']}_clean_maps_without_type_{'_'.join(map(str, config['unlearn_obstacles_type']))}_ratio_{unlearn_maps_ratio}_seed_{seed}.json"
        unlearn_flags_file = f"map_data/{config['game_type']}/unlearn_flags_ratio_{unlearn_maps_ratio}_seed_{seed}.pkl"

        if not all(os.path.exists(f) for f in [map_file, clean_map_file, unlearn_flags_file]):
            return None

        Retrain_model_file = f"{config['game_type']}_Retrain_Model_ratio_{unlearn_maps_ratio}_types_{'_'.join(map(str, config['unlearn_obstacles_type']))}_best.pkl"
        Retrain_model_path = os.path.join(Retrain_model_dir, Retrain_model_file)

        if not os.path.exists(Retrain_model_path):
            return None

        original_seed = config.get('current_seed')
        seed_control(seed)

        with open(unlearn_flags_file, 'rb') as f:
            unlearn_data = pickle.load(f)
            unlearn_map_flags = unlearn_data['unlearn_map_flags']

        start_time = time.time()

        env_original, agent_original, _ = init_module(
            map_file_path=map_file,
            filename=Retrain_model_file,
            model_dir=Retrain_model_dir,
            size=config['size'],
            n_ob=config['n_ob'],
            unlearn_obstacles_type=config['unlearn_obstacles_type'],
            device=config['device']
        )

        original_results = verify_single_model_with_full_metrics(
            env_original, agent_original, config['game_type'], config['n_maps'],
            max_steps_test, config['test_episodes_per_map'],
            unlearn_map_flags, config['unlearn_obstacles_type']
        )

        env_clean, agent_clean, _ = init_module(
            map_file_path=clean_map_file,
            filename=Retrain_model_file,
            model_dir=Retrain_model_dir,
            size=config['size'],
            n_ob=config['n_ob'],
            unlearn_obstacles_type=config['unlearn_obstacles_type'],
            device=config['device']
        )

        clean_results = verify_single_model_with_full_metrics(
            env_clean, agent_clean, config['game_type'], config['n_maps'],
            max_steps_test, config['test_episodes_per_map'],
            unlearn_map_flags, config['unlearn_obstacles_type']
        )

        end_time = time.time()

        if original_seed is not None:
            seed_control(original_seed)

        result = {
            'model_type': 'Retrain',
            'seed': seed,
            'standard_epoch': standard_epoch,
            'unlearn_epoch': 'N/A',
            'reward_scale': 'N/A',
            'poison_intensity': 'N/A',
            'tem_dfru_max_steps': 'N/A',
            'status': 'success',
            'verification_time': format_number(end_time - start_time, 2)
        }

        for category in ['all', 'unlearn', 'retain']:
            result[f'original_{category}_avg_steps'] = original_results[category]['avg_steps']
            result[f'original_{category}_avg_reward'] = original_results[category]['avg_reward']
            result[f'original_{category}_collision_summary'] = original_results[category]['collision_summary']
            result[f'original_{category}_unlearn_ratio'] = original_results[category]['unlearn_ratio']
            result[f'original_{category}_avg_perplexity'] = original_results[category]['avg_perplexity']

            result[f'clean_{category}_avg_steps'] = clean_results[category]['avg_steps']
            result[f'clean_{category}_avg_reward'] = clean_results[category]['avg_reward']
            result[f'clean_{category}_collision_summary'] = clean_results[category]['collision_summary']
            result[f'clean_{category}_unlearn_ratio'] = clean_results[category]['unlearn_ratio']
            result[f'clean_{category}_avg_perplexity'] = clean_results[category]['avg_perplexity']

        print(f"    Done in {end_time - start_time:.2f}s")

        return result

    except Exception as e:
        print(f"    Error: {str(e)}")
        return {
            'model_type': 'Retrain',
            'seed': seed,
            'standard_epoch': standard_epoch,
            'unlearn_epoch': 'N/A',
            'reward_scale': 'N/A',
            'poison_intensity': 'N/A',
            'tem_dfru_max_steps': 'N/A',
            'status': 'error',
            'error_message': str(e),
            'verification_time': 0
        }


def verify_dfru_model(unlearn_epoch, reward_scale, poison_intensity, tem_dfru_max_steps, max_steps_test, config):
    """Verify single DFRU model."""
    print(f"  DFRU: epoch={unlearn_epoch}, scale={reward_scale}, intensity={poison_intensity}, steps={tem_dfru_max_steps}")

    try:
        seed = config['seed']
        standard_epoch = config['standard_epoch']

        unlearned_model_dir = (f"./model/{config['game_type']}/DFRU_Model_standard_epoch_{standard_epoch}"
                               f"/tem_dfru_max_steps_{tem_dfru_max_steps}")

        map_file = f"map_data/{config['game_type']}/{config['n_maps']}_maps_seed_{seed}.json"
        clean_map_file = (f"map_data/{config['game_type']}/{config['n_maps']}_clean_maps"
                          f"_without_type_{'_'.join(map(str, config['unlearn_obstacles_type']))}"
                          f"_ratio_{config['unlearn_maps_ratio']}_seed_{seed}.json")
        unlearn_flags_file = f"map_data/{config['game_type']}/unlearn_flags_ratio_{config['unlearn_maps_ratio']}_seed_{seed}.pkl"

        if not all(os.path.exists(f) for f in [map_file, clean_map_file, unlearn_flags_file]):
            return {
                'model_type': 'DFRU',
                'seed': seed,
                'standard_epoch': standard_epoch,
                'unlearn_epoch': unlearn_epoch,
                'reward_scale': reward_scale,
                'poison_intensity': poison_intensity,
                'tem_dfru_max_steps': tem_dfru_max_steps,
                'status': 'map_files_not_found',
                'verification_time': 0
            }

        dfru_model_file = (f"{config['game_type']}_DFRU_Model_epoch_{unlearn_epoch}_ratio_{config['unlearn_maps_ratio']}"
                           f"_types_{'_'.join(map(str, config['unlearn_obstacles_type']))}"
                           f"_reward_scale_{reward_scale}_intensity_{poison_intensity}.pkl")
        dfru_model_path = os.path.join(unlearned_model_dir, dfru_model_file)

        if not os.path.exists(dfru_model_path):
            return {
                'model_type': 'DFRU',
                'seed': seed,
                'standard_epoch': standard_epoch,
                'unlearn_epoch': unlearn_epoch,
                'reward_scale': reward_scale,
                'poison_intensity': poison_intensity,
                'tem_dfru_max_steps': tem_dfru_max_steps,
                'status': 'not_found',
                'verification_time': 0
            }

        with open(unlearn_flags_file, 'rb') as f:
            unlearn_data = pickle.load(f)
            unlearn_map_flags = unlearn_data['unlearn_map_flags']

        start_time = time.time()

        env_original, agent_original, _ = init_module(
            map_file_path=map_file,
            filename=dfru_model_file,
            model_dir=unlearned_model_dir,
            size=config['size'],
            n_ob=config['n_ob'],
            unlearn_obstacles_type=config['unlearn_obstacles_type'],
            device=config['device']
        )

        original_results = verify_single_model_with_full_metrics(
            env_original, agent_original, config['game_type'], config['n_maps'],
            max_steps_test, config['test_episodes_per_map'],
            unlearn_map_flags, config['unlearn_obstacles_type']
        )

        env_clean, agent_clean, _ = init_module(
            map_file_path=clean_map_file,
            filename=dfru_model_file,
            model_dir=unlearned_model_dir,
            size=config['size'],
            n_ob=config['n_ob'],
            unlearn_obstacles_type=config['unlearn_obstacles_type'],
            device=config['device']
        )

        clean_results = verify_single_model_with_full_metrics(
            env_clean, agent_clean, config['game_type'], config['n_maps'],
            max_steps_test, config['test_episodes_per_map'],
            unlearn_map_flags, config['unlearn_obstacles_type']
        )

        end_time = time.time()

        result = {
            'model_type': 'DFRU',
            'seed': seed,
            'standard_epoch': standard_epoch,
            'unlearn_epoch': unlearn_epoch,
            'reward_scale': reward_scale,
            'poison_intensity': poison_intensity,
            'tem_dfru_max_steps': tem_dfru_max_steps,
            'status': 'success',
            'verification_time': format_number(end_time - start_time, 2)
        }

        for category in ['all', 'unlearn', 'retain']:
            result[f'original_{category}_avg_steps'] = original_results[category]['avg_steps']
            result[f'original_{category}_avg_reward'] = original_results[category]['avg_reward']
            result[f'original_{category}_collision_summary'] = original_results[category]['collision_summary']
            result[f'original_{category}_unlearn_ratio'] = original_results[category]['unlearn_ratio']
            result[f'original_{category}_avg_perplexity'] = original_results[category]['avg_perplexity']

            result[f'clean_{category}_avg_steps'] = clean_results[category]['avg_steps']
            result[f'clean_{category}_avg_reward'] = clean_results[category]['avg_reward']
            result[f'clean_{category}_collision_summary'] = clean_results[category]['collision_summary']
            result[f'clean_{category}_unlearn_ratio'] = clean_results[category]['unlearn_ratio']
            result[f'clean_{category}_avg_perplexity'] = clean_results[category]['avg_perplexity']

        print(f"  Done in {end_time - start_time:.2f}s")

        return result

    except Exception as e:
        print(f"Error verifying DFRU model: {str(e)}")
        return {
            'model_type': 'DFRU',
            'seed': config.get('seed', 'unknown'),
            'standard_epoch': config.get('standard_epoch', 'unknown'),
            'unlearn_epoch': unlearn_epoch,
            'reward_scale': reward_scale,
            'poison_intensity': poison_intensity,
            'tem_dfru_max_steps': tem_dfru_max_steps,
            'status': 'error',
            'error_message': str(e),
            'verification_time': 0
        }


def save_verification_results(results, config, suffix='all'):
    """Save verification results to Excel and CSV."""
    if not results:
        print(f"No results to save for {suffix}")
        return

    df = pd.DataFrame(results)

    column_order = [
        'model_type', 'seed', 'standard_epoch', 'unlearn_epoch', 'reward_scale', 'poison_intensity', 'tem_dfru_max_steps', 'status',
        'verification_time',
        'original_all_avg_steps', 'original_all_avg_reward',
        'original_all_collision_summary', 'original_all_unlearn_ratio', 'original_all_avg_perplexity',
        'original_unlearn_avg_steps', 'original_unlearn_avg_reward',
        'original_unlearn_collision_summary', 'original_unlearn_unlearn_ratio', 'original_unlearn_avg_perplexity',
        'original_retain_avg_steps', 'original_retain_avg_reward',
        'original_retain_collision_summary', 'original_retain_unlearn_ratio', 'original_retain_avg_perplexity',
        'clean_all_avg_steps', 'clean_all_avg_reward',
        'clean_all_collision_summary', 'clean_all_unlearn_ratio', 'clean_all_avg_perplexity',
        'clean_unlearn_avg_steps', 'clean_unlearn_avg_reward',
        'clean_unlearn_collision_summary', 'clean_unlearn_unlearn_ratio', 'clean_unlearn_avg_perplexity',
        'clean_retain_avg_steps', 'clean_retain_avg_reward',
        'clean_retain_collision_summary', 'clean_retain_unlearn_ratio', 'clean_retain_avg_perplexity'
    ]

    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]

    # Save Excel
    excel_file = os.path.join(config['verify_result_dir'], f"dfru_verification_{suffix}.xlsx")
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)
        worksheet = writer.sheets['Results']

        # Import alignment for cell formatting
        from openpyxl.styles import Alignment

        # Set column width to 7 characters and center align all cells
        for column in worksheet.columns:
            column_letter = column[0].column_letter
            worksheet.column_dimensions[column_letter].width = 7

            # Center align all cells in this column
            for cell in column:
                cell.alignment = Alignment(horizontal='center', vertical='center')

    print(f"Excel saved: {excel_file}")

    # Save CSV
    csv_file = os.path.join(config['verify_result_dir'], f"dfru_verification_{suffix}.csv")
    df.to_csv(csv_file, index=False, float_format='%.3f')
    print(f"CSV saved: {csv_file}")


def main():
    """Batch verification of all DFRU models with threshold filtering."""
    parser = argparse.ArgumentParser(description='Batch verify DFRU models with full metrics')
    parser.add_argument('--tem_dfru_max_steps', type=int, default=21,
                        help='Maximum steps for DFRU training (default: 21)')
    parser.add_argument('--test_episodes_per_map', type=int, default=20,
                        help='Number of test episodes per map (default: 20)')
    parser.add_argument('--threshold_ratio', type=float, default=0.95,
                        help='Threshold ratio for filtering (default: 0.95, means original_all_avg_steps < tem_dfru_max_steps * 0.95)')
    args = parser.parse_args()

    config = {
        'game_type': 'grid_world',
        'n_maps': 50,
        'standard_epoch': 50,
        'unlearn_maps_ratio': 30,
        'unlearn_obstacles_type': [3],
        'seed': 28,
        'size': 10,
        'n_ob': 10,
        'test_episodes_per_map': args.test_episodes_per_map,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    seed_control(config['seed'])
    config['current_seed'] = config['seed']

    config['verify_result_dir'] = f"./result/{config['game_type']}/DFRU_Model_standard_epoch_{config['standard_epoch']}/tem_dfru_max_steps_{args.tem_dfru_max_steps}/"
    init_files(config['verify_result_dir'])

    max_steps_test = args.tem_dfru_max_steps + 5

    # Create parameter grid for DFRU models
    param_grid = create_parameter_grid(args.tem_dfru_max_steps)
    total = len(param_grid) + 3  # +3 for Pretrain, Retrain, Nontranslfs

    print(f"\n{'='*80}")
    print(f"Starting unified verification for tem_dfru_max_steps={args.tem_dfru_max_steps}")
    print(f"Test episodes per map: {args.test_episodes_per_map}")
    print(f"Max test steps: {max_steps_test}")
    print(f"Total models to verify: {total} (1 Pretrain + 1 Retrain + 1 Nontranslfs + {len(param_grid)} DFRU)")
    print(f"DFRU grid: 8 epochs x 5 reward_scales x 8 poison_intensities = {len(param_grid)} combinations")
    print(f"{'='*80}\n")

    all_results = []
    successful, not_found, error = 0, 0, 0
    start_time = time.time()

    # Verify Pretrain model
    print(f"[1/{total}] Verifying Pretrain model...")
    pretrain_result = verify_Pretrain_model(
        config['seed'], config['standard_epoch'], max_steps_test, config
    )
    if pretrain_result:
        all_results.append(pretrain_result)
        if pretrain_result['status'] == 'success':
            successful += 1
        else:
            error += 1
    else:
        error += 1

    # Verify Retrain model
    print(f"\n[2/{total}] Verifying Retrain model...")
    retrain_result = verify_Retrain_model(
        config['seed'], config['standard_epoch'], config['unlearn_maps_ratio'], max_steps_test, config
    )
    if retrain_result:
        all_results.append(retrain_result)
        if retrain_result['status'] == 'success':
            successful += 1
        else:
            error += 1
    else:
        error += 1

    # Verify Nontranslfs model
    print(f"\n[3/{total}] Verifying Nontranslfs model...")
    nontranslfs_result = verify_Nontranslfs_model(
        config['seed'], config['standard_epoch'], config['unlearn_maps_ratio'], max_steps_test, config
    )
    if nontranslfs_result:
        all_results.append(nontranslfs_result)
        if nontranslfs_result['status'] == 'success':
            successful += 1
        else:
            error += 1
    else:
        error += 1

    # Verify all DFRU models
    print(f"\n{'='*80}")
    print(f"Verifying {len(param_grid)} DFRU models...")
    print(f"{'='*80}\n")

    for idx, (unlearn_epoch, reward_scale, poison_intensity, tem_steps) in enumerate(param_grid, 4):
        print(f"[{idx}/{total}] epoch={unlearn_epoch}, scale={reward_scale}, intensity={poison_intensity}, steps={tem_steps}")
        result = verify_dfru_model(
            unlearn_epoch, reward_scale, poison_intensity, tem_steps, max_steps_test, config
        )
        all_results.append(result)

        if result['status'] == 'success':
            successful += 1
        elif result['status'] == 'not_found':
            not_found += 1
        else:
            error += 1

    total_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Verification completed in {total_time/60:.1f} minutes")
    print(f"Success: {successful}/{total} | Not found: {not_found} | Errors: {error}")
    print(f"{'='*80}\n")

    # Save all results
    save_verification_results(all_results, config, suffix='all')

    # Filter and save results based on threshold
    successful_dfru = [r for r in all_results if r['status'] == 'success' and r['model_type'] == 'DFRU']
    threshold = max_steps_test * args.threshold_ratio

    filtered_results = []
    # Add Pretrain, Retrain and Nontranslfs first
    for r in all_results:
        if r['model_type'] in ['Pretrain', 'Retrain', 'Nontranslfs'] and r['status'] == 'success':
            filtered_results.append(r)

    # Filter DFRU models
    for r in successful_dfru:
        if r['original_all_avg_steps'] < threshold:
            filtered_results.append(r)

    print(f"\nFiltering results:")
    print(f"Threshold: original_all_avg_steps < {threshold:.2f}")
    print(f"Filtered DFRU models: {len(filtered_results)-3}/{len(successful_dfru)} passed")
    print(f"Total filtered models (including Pretrain/Retrain/Nontranslfs): {len(filtered_results)}")

    if filtered_results:
        save_verification_results(filtered_results, config, suffix='filtered')

    return all_results


if __name__ == "__main__":
    results = main()
