from init import *
from utils import *
from Hyperparameter_control import*

parser = argparse.ArgumentParser(description='DFRU: Dual Fine-grained Reinforcement Unlearning')
parser.add_argument('--unlearn_epoch', type=int, default=50)
parser.add_argument('--standard_epoch', type=int, default=50)

parser.add_argument('--game_type', type=str, default='grid_world',
                    choices=['grid_world', 'aircraft_landing'])
parser.add_argument('--n_maps', type=int, default=50)
parser.add_argument('--seed', type=int, default=28)
parser.add_argument('--unlearn_obstacles_type', type=int, nargs='+', default=[3],
                    choices=[1, 2, 3, 4])
parser.add_argument('--unlearn_maps_ratio', type=int, default=30)
parser.add_argument('--reward_scale', type=int, default=5)
parser.add_argument('--poison_intensity', type=int, default=20)
parser.add_argument('--tem_dfru_max_steps', type=int, default=25)
parser.add_argument('--train_max_steps', type=int, default=30)
parser.add_argument('--test_max_steps', type=int, default=40)
parser.add_argument('--train_unlearn', action='store_true')
parser.add_argument('--test_unlearn', action='store_true')
parser.add_argument('--use_script_defaults', action='store_true')


args = parser.parse_args()

game_type = args.game_type
n_maps = args.n_maps
unlearn_epoch = args.unlearn_epoch
standard_epoch = args.standard_epoch
unlearn_maps_ratio = args.unlearn_maps_ratio
unlearn_obstacles_type = args.unlearn_obstacles_type
reward_scale = args.reward_scale
poison_intensity = args.poison_intensity
tem_dfru_max_steps = args.tem_dfru_max_steps

size = 10
n_ob = 10
env = Simple2DEnvironment(size, n_ob)

if args.seed is not None:
    seed_control(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Pretrain_model_dir = f"./model/{game_type}/Pretrain_model_standard_epoch_{standard_epoch}/"
unlearn_model_dir = f"./model/{game_type}/DFRU_Model_standard_epoch_{standard_epoch}/tem_dfru_max_steps_{tem_dfru_max_steps}"
Pretrain_result_dir = f"./result/{game_type}/Pretrain_model_standard_epoch_{standard_epoch}/"
unlearn_result_dir = f"./result/{game_type}/DFRU_Model_standard_epoch_{standard_epoch}/tem_dfru_max_steps_{tem_dfru_max_steps}"

init_files(Pretrain_model_dir)
init_files(unlearn_model_dir)
init_files(Pretrain_result_dir)
init_files(unlearn_result_dir)

max_steps_train_unlearn = get_default_max_steps(game_type, 'train_DFRU') if hasattr(get_default_max_steps, 'train_DFRU') else get_default_max_steps(game_type, 'train_SPU')
max_steps_test_unlearn = get_default_max_steps(game_type, 'test_DFRU') if hasattr(get_default_max_steps, 'test_DFRU') else get_default_max_steps(game_type, 'test_SPU')

if args.train_max_steps is not None:
    max_steps_train_pretrain_model = args.train_max_steps
else:
    max_steps_train_pretrain_model = get_default_max_steps(args.game_type, 'train_normal')

if args.test_max_steps is not None:
    max_steps_test_pretrain_model = args.test_max_steps
else:
    max_steps_test_pretrain_model = get_default_max_steps(args.game_type, 'test_normal')

unlearn_flags_file = f"map_data/{game_type}/unlearn_flags_ratio_{unlearn_maps_ratio}_seed_{args.seed}.pkl"
map_file = f"map_data/{game_type}/{n_maps}_maps_seed_{args.seed}.json"
clean_map_file = f"map_data/{game_type}/{n_maps}_clean_maps_without_type_{'_'.join(map(str, unlearn_obstacles_type))}_ratio_{unlearn_maps_ratio}_seed_{args.seed}.json"

unlearn_map_flags, unlearn_indices, n_unlearn_maps = init_unlearn_flags(
    unlearn_flags_file, game_type, n_maps, unlearn_maps_ratio, unlearn_obstacles_type, args.seed
)

env = init_map_file(env, map_file, game_type, n_maps, size, n_ob)
init_clean_maps_with_flags(map_file, clean_map_file, unlearn_obstacles_type, unlearn_map_flags)

episodes = standard_epoch * n_maps
batch_size = 32

script_run_train_unlearn_model = False
script_run_test_unlearn_model = False

if args.use_script_defaults:
    run_train_unlearn_model = script_run_train_unlearn_model
    run_test_unlearn_model = script_run_test_unlearn_model
else:
    if any([args.train_unlearn, args.test_unlearn]):
        run_train_unlearn_model = args.train_unlearn
        run_test_unlearn_model = args.test_unlearn
    else:
        run_train_unlearn_model = script_run_train_unlearn_model
        run_test_unlearn_model = script_run_test_unlearn_model

def targeted_poisoning(state, intensity=0.2):
    """
    Targeted state poisoning method for DFRU unlearning (from SSP)

    Args:
        state: Current state vector [x, y, surrounding 8 directions]
        intensity: Poisoning intensity (0-1)

    Returns:
        state_poisoned: Poisoned state vector
    """
    state_poisoned = state.copy()

    for i in range(2, 10):
        if state[i] == 3:
            if np.random.random() < intensity:
                state_poisoned[i] = np.random.choice([0])

    num_shuffle = 2
    indices = list(range(2, 10))
    chosen_indices = np.random.choice(indices, num_shuffle, replace=False)
    values = [state_poisoned[idx] for idx in chosen_indices]
    np.random.shuffle(values)
    for idx, val in zip(chosen_indices, values):
        state_poisoned[idx] = val

    return state_poisoned

if run_train_unlearn_model:
    env, agent, buffer = init_module(map_file_path=map_file,
                                     filename=f"{game_type}_Pretrain_Model_seed_{args.seed}_best.pkl",
                                     model_dir=Pretrain_model_dir,
                                     size=size,
                                     n_ob=n_ob,
                                     unlearn_obstacles_type=unlearn_obstacles_type,
                                     device=device)

    all_step = 0
    steps = 0
    epsilon = 1.0
    each_map = unlearn_epoch
    map_id = -1
    total_reward = 0
    Train_DFRU_Model_rewards = []
    unlearn_map_indices = unlearn_indices

    episode_count = 0
    for map_idx in unlearn_map_indices:
        for _ in range(each_map):
            state = env.reset(map_index=map_idx, game_type=game_type)

            if episode_count % each_map == 0:
                all_step = 0
                epsilon = 1.0
                total_reward = 0

            done = False
            steps = 0
            while not done and steps < tem_dfru_max_steps:
                # SSP: Apply targeted poisoning to state
                state = targeted_poisoning(state, intensity=poison_intensity/100)
                action, _ = agent.get_action(state, epsilon)
                # SPU: Use adversarial reward with reward_scale
                next_state, reward, done, _ = env.step_Adversarial_Reward(action, reward_scale=reward_scale)
                buffer.push(state, action, reward, next_state, done)
                agent.update(batch_size)

                total_reward += reward
                state = next_state
                steps += 1

            all_step = all_step + steps
            episode_count += 1

            if episode_count % each_map == 0:
                print(f'Map {map_idx} Reward: {total_reward / each_map:.2f} Steps: {all_step / each_map:.2f}')

            Train_DFRU_Model_rewards.append(total_reward)
            epsilon *= 0.995

    save_model(agent,
               f"{game_type}_DFRU_Model"
               f"_epoch_{unlearn_epoch}"
               f"_ratio_{unlearn_maps_ratio}"
               f"_types_{'_'.join(map(str, unlearn_obstacles_type))}"
               f"_reward_scale_{reward_scale}"
               f"_intensity_{poison_intensity}"
               f".pkl",
               {
                   'unlearn_epoch': unlearn_epoch,
                   'unlearn_obstacles_type': unlearn_obstacles_type,
                   'training_episodes': each_map,
                   'final_epsilon': epsilon,
                   'total_maps': len(unlearn_map_indices),
                   'reward_scale': reward_scale,
                   'poison_intensity': poison_intensity,
               }, model_dir=unlearn_model_dir)

    with open(os.path.join(unlearn_result_dir, f"{game_type}_Train_DFRU_Model_cRewards" 
                                               f"_epoch_{unlearn_epoch}"
                                               f"_ratio_{unlearn_maps_ratio}"
                                               f"_types_{'_'.join(map(str, unlearn_obstacles_type))}"
                                               f"_reward_scale_{reward_scale}"
                                               f"_intensity_{poison_intensity}.pkl"), "wb") as f:
        pickle.dump(Train_DFRU_Model_rewards, f)

if run_test_unlearn_model:
    env, agent, buffer = init_module(map_file_path=map_file,
                                     filename=f"{game_type}_DFRU_Model"
                                              f"_epoch_{unlearn_epoch}"
                                              f"_ratio_{unlearn_maps_ratio}"
                                              f"_types_{'_'.join(map(str, unlearn_obstacles_type))}"
                                              f"_reward_scale_{reward_scale}"
                                              f"_intensity_{poison_intensity}"
                                              f".pkl",
                                     model_dir=unlearn_model_dir,
                                     size=size,
                                     n_ob=n_ob,
                                     unlearn_obstacles_type=unlearn_obstacles_type,
                                     device=device)

    n_episodes = episodes
    all_step = 0
    steps = 0
    epsilon = 1.0
    each_map = standard_epoch
    map_id = -1
    total_reward = 0
    previous_r_truth_unlearn_sum = 0
    Test_DFRU_Model_r_truth = []
    Test_DFRU_Model_rewards = []

    for i_episode in tqdm(range(n_episodes)):
        all_step = all_step + steps
        if i_episode % each_map == 0:
            map_id = map_id + 1
            state = env.reset(map_index=map_id, game_type=game_type)

            all_step = 0
            epsilon = 1.0
            total_reward = 0
        else:
            state = env.reset(map_index=map_id, game_type=game_type)
        done = False
        steps = 0
        previous_r_truth_unlearn_sum = 0
        while not done and steps < max_steps_test_unlearn:
            action, r_truth = agent.get_action(state, epsilon=0.05)

            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            previous_r_truth_unlearn_sum += r_truth
            state = next_state
            steps += 1
        Test_DFRU_Model_r_truth.append(previous_r_truth_unlearn_sum / 3)
        Test_DFRU_Model_rewards.append(total_reward)

        if (i_episode + 1) % each_map == 0:
            print(f'Map {map_id} Reward: {total_reward / each_map:.2f} R_truth: {previous_r_truth_unlearn_sum/3:.2f} Steps: {all_step / each_map:.2f}')

    with open(os.path.join(unlearn_result_dir, f"{game_type}_Test_DFRU_Model_RTruth_epoch_{unlearn_epoch}_ratio_{unlearn_maps_ratio}"
                                       f"_types_{'_'.join(map(str, unlearn_obstacles_type))}_reward_scale_{reward_scale}_intensity_{poison_intensity}.pkl"), "wb") as f:
        pickle.dump(Test_DFRU_Model_r_truth, f)

    with open(os.path.join(unlearn_result_dir, f"{game_type}_Test_DFRU_Model_cRewards_epoch_{unlearn_epoch}_ratio_{unlearn_maps_ratio}"
                                       f"_types_{'_'.join(map(str, unlearn_obstacles_type))}_reward_scale_{reward_scale}_intensity_{poison_intensity}.pkl"), "wb") as f:
        pickle.dump(Test_DFRU_Model_rewards, f)
