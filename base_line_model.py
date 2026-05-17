from init import *
from utils import *
from Hyperparameter_control import *

parser = argparse.ArgumentParser(description='Baseline Models for Reinforcement Unlearning')
parser.add_argument('--unlearn_epoch', type=int, default=50)
parser.add_argument('--standard_epoch', type=int, default=50)
parser.add_argument('--game_type', type=str, default='grid_world',
                    choices=['grid_world', 'aircraft_landing'])
parser.add_argument('--n_maps', type=int, default=50)
parser.add_argument('--seed', type=int, default=28)
parser.add_argument('--unlearn_obstacles_type', type=int, nargs='+', default=[2],
                    choices=[1, 2, 3, 4])
parser.add_argument('--unlearn_maps_ratio', type=int, default=20)
parser.add_argument('--train_retrain_full', action='store_true')
parser.add_argument('--test_retrain_full', action='store_true')
parser.add_argument('--train_Nontranslfs', action='store_true')
parser.add_argument('--test_Nontranslfs', action='store_true')
parser.add_argument('--use_script_defaults', action='store_true')
parser.add_argument('--train_max_steps', type=int, default=30)
parser.add_argument('--test_max_steps', type=int, default=40)

args = parser.parse_args()

game_type = args.game_type
n_maps = args.n_maps
unlearn_epoch = args.unlearn_epoch
standard_epoch = args.standard_epoch

unlearn_maps_ratio = args.unlearn_maps_ratio
unlearn_obstacles_type = args.unlearn_obstacles_type

if args.train_max_steps is not None:
    max_steps_train_retrain = args.train_max_steps
else:
    max_steps_train_retrain = get_default_max_steps(args.game_type, 'train_retrain')

if args.test_max_steps is not None:
    max_steps_test_retrain = args.test_max_steps
else:
    max_steps_test_retrain = get_default_max_steps(args.game_type, 'ttest_retrain')

size = 10
n_ob = 10
env = Simple2DEnvironment(size, n_ob)

if args.seed is not None:
    seed_control(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# retrain_model_dir = f"./model/{game_type}/Retrain_Model_ratio_{unlearn_maps_ratio}_types_{'_'.join(map(str, unlearn_obstacles_type))}"
retrain_model_dir = f"./model/{game_type}/Retrain_Model_standard_epoch_{standard_epoch}/"
retrain_model_result_dir = f"./result/{game_type}/Retrain_Model_standard_epoch_{standard_epoch}/"

Nontranslfs_model_dir = f"./model/{game_type}/Nontranslfs_Model"
Nontranslfs_result_dir = f"./result/{game_type}/Nontranslfs_Model"

init_files(retrain_model_dir)
init_files(retrain_model_result_dir)
init_files(Nontranslfs_model_dir)
init_files(Nontranslfs_result_dir)

max_steps_train_Nontranslfs = get_default_max_steps(game_type, 'train_Nontranslfs')
max_steps_test_Nontranslfs = get_default_max_steps(game_type, 'test_Nontranslfs')

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

script_run_Train_Retrain_With_Clean_Maps_Model = False
script_run_Test_Retrain_With_Clean_Maps_Model = False
script_run_train_Nontranslfs_model = False
script_run_test_Nontranslfs_model = False

if args.use_script_defaults:
    run_Train_Retrain_With_Clean_Maps_Model = script_run_Train_Retrain_With_Clean_Maps_Model
    run_Test_Retrain_With_Clean_Maps_Model = script_run_Test_Retrain_With_Clean_Maps_Model
    run_train_Nontranslfs_model = script_run_train_Nontranslfs_model
    run_test_Nontranslfs_model = script_run_test_Nontranslfs_model
else:
    if any([args.train_retrain_full, args.test_retrain_full, args.train_Nontranslfs, args.test_Nontranslfs]):
        run_Train_Retrain_With_Clean_Maps_Model = args.train_retrain_full
        run_Test_Retrain_With_Clean_Maps_Model = args.test_retrain_full
        run_train_Nontranslfs_model = args.train_Nontranslfs
        run_test_Nontranslfs_model = args.test_Nontranslfs
    else:
        run_Train_Retrain_With_Clean_Maps_Model = script_run_Train_Retrain_With_Clean_Maps_Model
        run_Test_Retrain_With_Clean_Maps_Model = script_run_Test_Retrain_With_Clean_Maps_Model
        run_train_Nontranslfs_model = script_run_train_Nontranslfs_model
        run_test_Nontranslfs_model = script_run_test_Nontranslfs_model

if run_Train_Retrain_With_Clean_Maps_Model:
    env, agent, buffer = init_module(map_file_path=clean_map_file,
                                     filename=None,
                                     model_dir=retrain_model_dir,
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
    Train_Retrain_With_Clean_Maps_Model_rewards = []

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
        while not done and steps < max_steps_train_retrain:
            action, _ = agent.get_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            agent.update(batch_size)

            total_reward += reward
            state = next_state
            steps += 1

        if (i_episode + 1) % each_map == 0:
            print(f'Map {map_id} Reward: {total_reward / each_map:.2f} Steps: {all_step / each_map:.2f}')

        Train_Retrain_With_Clean_Maps_Model_rewards.append(total_reward)
        epsilon *= 0.995

    save_model(agent,
               f"{game_type}_Retrain_Model"
               f"_ratio_{unlearn_maps_ratio}"
               f"_types_{'_'.join(map(str, unlearn_obstacles_type))}"
               f"_train_steps_{max_steps_train_retrain}"
               f".pkl",
               {
                   'ignored_obstacles_type': unlearn_obstacles_type,
                   'training_episodes': n_episodes,
                   'final_epsilon': epsilon,
                   'total_maps': n_maps
               }, model_dir=retrain_model_dir)

    with open(os.path.join(retrain_model_result_dir, f"{game_type}_Train_Retrain_Model_cRewards"
                                       f"_ratio_{unlearn_maps_ratio}"
                                       f"_types_{'_'.join(map(str, unlearn_obstacles_type))}"
                                       f"_train_steps_{max_steps_train_retrain}"
                                       f".pkl"), "wb") as f:
        pickle.dump(Train_Retrain_With_Clean_Maps_Model_rewards, f)

if run_Test_Retrain_With_Clean_Maps_Model:
    env, agent, buffer = init_module(map_file_path=map_file,
                                     filename=f"{game_type}_Retrain_Model"
                                              f"_ratio_{unlearn_maps_ratio}"
                                              f"_types_{'_'.join(map(str, unlearn_obstacles_type))}"
                                              f"_train_steps_{max_steps_train_retrain}"
                                              f".pkl",
                                     model_dir=retrain_model_dir,
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
    previous_r_truth_retrain_full_sum = 0
    Test_Retrain_With_Clean_Maps_Model_r_truth = []
    Test_Retrain_With_Clean_Maps_Model_rewards = []

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
        previous_r_truth_retrain_full_sum = 0
        while not done and steps < max_steps_test_retrain:
            action, r_truth = agent.get_action(state, epsilon=0.05)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            previous_r_truth_retrain_full_sum += r_truth
            state = next_state
            steps += 1

        Test_Retrain_With_Clean_Maps_Model_r_truth.append(previous_r_truth_retrain_full_sum / 3)
        Test_Retrain_With_Clean_Maps_Model_rewards.append(total_reward)

        if (i_episode + 1) % each_map == 0:
            print(f'Map {map_id} Reward: {total_reward / each_map:.2f} R_truth: {previous_r_truth_retrain_full_sum/3:.2f} Steps: {all_step / each_map:.2f}')

    with open(os.path.join(retrain_model_result_dir, f"{game_type}_Test_Retrain_Model_RTruth"
                                       f"_ratio_{unlearn_maps_ratio}"
                                       f"_types_{'_'.join(map(str, unlearn_obstacles_type))}"
                                       f"_train_steps_{max_steps_train_retrain}"
                                       f".pkl"), "wb") as f:
        pickle.dump(Test_Retrain_With_Clean_Maps_Model_r_truth, f)

    with open(os.path.join(retrain_model_result_dir, f"{game_type}_Test_Retrain_Model_cRewards"
                                       f"_ratio_{unlearn_maps_ratio}"
                                       f"_types_{'_'.join(map(str, unlearn_obstacles_type))}"
                                       f"_train_steps_{max_steps_train_retrain}"
                                       f".pkl"), "wb") as f:
        pickle.dump(Test_Retrain_With_Clean_Maps_Model_rewards, f)

if run_train_Nontranslfs_model:
    env, agent, buffer = init_module(map_file_path=clean_map_file,
                                     filename=None,
                                     model_dir=Nontranslfs_model_dir,
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
    Train_Nontranslfs_Model_rewards = []

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
        while not done and steps < max_steps_train_Nontranslfs:
            action, _ = agent.get_action(state, epsilon)
            next_state, reward, done, hit_unlearn = env.step_with_collision_tracking(
                action, unlearn_obstacles_type
            )
            buffer.push_with_collision_info(
                state, action, reward, next_state, done, hit_unlearn
            )
            agent.Nontranslfs_update(batch_size)

            total_reward += reward
            state = next_state
            steps += 1

        if (i_episode + 1) % each_map == 0:
            print(f'Map {map_id} Reward: {total_reward / each_map:.2f} Steps: {all_step / each_map:.2f}')

        Train_Nontranslfs_Model_rewards.append(total_reward)
        epsilon *= 0.995

    save_model(agent,
               f"{game_type}_Nontranslfs_Model"
               f"_epoch_{unlearn_epoch}"
               f"_ratio_{unlearn_maps_ratio}"
               f"_types_{'_'.join(map(str, unlearn_obstacles_type))}.pkl",
               {
                   'unlearn_epoch': unlearn_epoch,
                   'unlearn_obstacles_type': unlearn_obstacles_type,
                   'training_episodes': n_episodes,
                   'final_epsilon': epsilon,
                   'total_maps': n_maps
               }, model_dir=Nontranslfs_model_dir)

    with open(os.path.join(Nontranslfs_result_dir, f"{game_type}_Train_Nontranslfs_Model_cRewards"
                                       f"_epoch_{unlearn_epoch}"
                                       f"_ratio_{unlearn_maps_ratio}"
                                       f"_types_{'_'.join(map(str, unlearn_obstacles_type))}.pkl"), "wb") as f:
        pickle.dump(Train_Nontranslfs_Model_rewards, f)

if run_test_Nontranslfs_model:
    env, agent, buffer = init_module(map_file_path=clean_map_file,
                                     filename=f"{game_type}_Nontranslfs_Model"
                                                f"_epoch_{unlearn_epoch}"
                                                f"_ratio_{unlearn_maps_ratio}"
                                                f"_types_{'_'.join(map(str, unlearn_obstacles_type))}.pkl",
                                     model_dir=Nontranslfs_model_dir,
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
    previous_r_truth_Nontranslfs_sum = 0
    Test_Nontranslfs_Model_r_truth = []
    Test_Nontranslfs_Model_rewards = []

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
        previous_r_truth_Nontranslfs_sum = 0
        while not done and steps < max_steps_test_Nontranslfs:
            action, r_truth = agent.get_action(state, epsilon=0.05)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            previous_r_truth_Nontranslfs_sum += r_truth
            state = next_state
            steps += 1

        Test_Nontranslfs_Model_r_truth.append(previous_r_truth_Nontranslfs_sum / 3)
        Test_Nontranslfs_Model_rewards.append(total_reward)

        if (i_episode + 1) % each_map == 0:
            print(f'Map {map_id} Reward: {total_reward / each_map:.2f} R_truth: {previous_r_truth_Nontranslfs_sum/3:.2f} Steps: {all_step / each_map:.2f}')

    with open(os.path.join(Nontranslfs_result_dir,
                           f"{game_type}_Test_Nontranslfs_Model_RTruth"
                           f"_epoch_{unlearn_epoch}"
                           f"_ratio_{unlearn_maps_ratio}"
                           f"_types_{'_'.join(map(str, unlearn_obstacles_type))}.pkl"), "wb") as f:
        pickle.dump(Test_Nontranslfs_Model_r_truth, f)

    with open(os.path.join(Nontranslfs_result_dir,
                           f"{game_type}_Test_Nontranslfs_Model_cRewards"
                           f"_epoch_{unlearn_epoch}"
                           f"_ratio_{unlearn_maps_ratio}"
                           f"_types_{'_'.join(map(str, unlearn_obstacles_type))}.pkl"), "wb") as f:
        pickle.dump(Test_Nontranslfs_Model_rewards, f)
