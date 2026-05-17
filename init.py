from utils import *
from visualization import *
import argparse
from Hyperparameter_control import *
from unlearn_comparison_visualization import *


def init_files(model_dir):
    """
    初始化文件目录，如果目录不存在则创建

    Args:
        model_dir (str): 目标目录路径

    Returns:
        None
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    else:
        print(f"Loaded existing directory from:  {model_dir}")


def init_unlearn_flags(unlearn_flags_file, game_type, n_maps, unlearn_maps_ratio, unlearn_obstacles_type, seed):
    """
    初始化或加载遗忘标志，确定哪些地图需要遗忘
    如果文件不存在，则随机选择指定比例的地图进行遗忘，并保存标志文件
    如果文件存在，则直接加载已有的遗忘标志

    Args:
        unlearn_flags_file (str): 遗忘标志文件路径
        game_type (str): 游戏类型（'grid_world' 或 'aircraft_landing'）
        n_maps (int): 总地图数量
        unlearn_maps_ratio (int): 需要遗忘的地图百分比
        unlearn_obstacles_type (list): 需要遗忘的障碍物类型列表
        seed (int): 随机种子

    Returns:
        tuple: (unlearn_map_flags, unlearn_indices, n_unlearn_maps)
            - unlearn_map_flags (list): 布尔列表，标记每个地图是否需要遗忘
            - unlearn_indices (ndarray): 需要遗忘的地图索引数组
            - n_unlearn_maps (int): 需要遗忘的地图总数
    """
    # 检查是否存在unlearn_flags文件
    if os.path.exists(unlearn_flags_file):
        # 加载已存在的unlearn_flags
        with open(unlearn_flags_file, 'rb') as f:
            saved_data = pickle.load(f)
            unlearn_map_flags = saved_data['unlearn_map_flags']
            unlearn_indices = saved_data['unlearn_indices']
            n_unlearn_maps = saved_data['n_unlearn_maps']
        print(f"Loaded existing unlearn flags from: {unlearn_flags_file}")

    else:
        # 创建新的unlearn_flags
        n_unlearn_maps = int(n_maps * unlearn_maps_ratio / 100)
        unlearn_map_flags = [False] * n_maps
        unlearn_indices = np.random.choice(n_maps, n_unlearn_maps, replace=False)
        for idx in unlearn_indices:
            unlearn_map_flags[idx] = True

        # 保存unlearn_flags
        os.makedirs(os.path.dirname(unlearn_flags_file), exist_ok=True)
        with open(unlearn_flags_file, 'wb') as f:
            pickle.dump({
                'unlearn_map_flags': unlearn_map_flags,
                'unlearn_indices': unlearn_indices,
                'n_unlearn_maps': n_unlearn_maps,
                'unlearn_ratio': unlearn_maps_ratio,
                'seed': seed,
                'n_maps': n_maps
            }, f)
        print(f"Created and saved new unlearn flags to: {unlearn_flags_file}")
        # 选择好要遗忘的障碍物后自动生成对比可视化
        try:
            print("Generating unlearning comparison visualization...")
            visualize_unlearn_comparison_auto(
                game_type=game_type,
                n_maps=n_maps,
                unlearn_obstacles_type=unlearn_obstacles_type,
                unlearn_maps_ratio=unlearn_maps_ratio,
                seed=seed
            )
            print("Comparison visualization completed!")
        except Exception as e:
            print(f"Warning: Failed to generate comparison visualization: {e}")

    print(f"Total maps: {n_maps}, Unlearn ratio: {unlearn_maps_ratio}%, Maps to unlearn: {n_unlearn_maps}")
    print(f"Unlearning maps at indices: {sorted(unlearn_indices.tolist())}")
    return unlearn_map_flags, unlearn_indices, n_unlearn_maps


def init_map_file(env, map_file, game_type, n_maps, size, n_ob):
    """
    初始化地图文件，如果不存在则生成并保存默认地图
    地图包含障碍物位置和目标位置，同时会生成可视化图片

    Args:
        env (Simple2DEnvironment): 环境实例
        map_file (str): 地图文件路径
        game_type (str): 游戏类型（'grid_world' 或 'aircraft_landing'）
        n_maps (int): 需要生成的地图数量

    Returns:
        Simple2DEnvironment: 加载了地图的环境实例
    """
    if os.path.exists(map_file):
        # 文件存在，正常加载
        env.load_maps(map_file)
        print(f"Loaded existing maps from: {map_file}")
    else:
        # 文件不存在，创建文件夹并生成默认地图
        os.makedirs(os.path.dirname(map_file), exist_ok=True)

        # 生成默认地图
        default_maps = []
        for _ in range(n_maps):
            env.reset(game_type=game_type)
            default_maps.append({"obstacles_location": env.obstacles_location, "target_location": env.target_location})

        # 保存默认地图
        env.maps = default_maps
        env.save_maps(map_file)
        print(f"Created and saved maps to: {map_file}")

        # 生成可视化图片
        try:
            print("Generating map visualization...")
            visualize_maps_for_spu(map_file, game_type, n_maps, size, n_ob)
            print("Map visualization completed!")
        except Exception as e:
            print(f"Warning: Failed to generate map visualization: {e}")
            print("You can manually run the visualization later using map_with_image.py")
    return env


def init_clean_maps_with_flags(map_file, clean_map_file, unlearn_obstacles_type, unlearn_map_flags):
    """
    根据遗忘标志创建清洁地图文件
    对于标记为需要遗忘的地图，移除指定类型的障碍物；其他地图保持原样

    Args:
        map_file (str): 原始地图文件路径
        clean_map_file (str): 清洁地图文件输出路径
        unlearn_obstacles_type (list): 需要遗忘的障碍物类型列表 (例如 [1, 3])
        unlearn_map_flags (list): 布尔列表，标记每个地图是否需要遗忘

    Returns:
        None
    """
    if os.path.exists(clean_map_file):
        # 文件存在，后续可以正常加载
        print("Clean map already exists")
    else:
        # 读取原始地图
        with open(map_file, 'r') as f:
            maps = json.load(f)

        # 创建新地图
        new_maps = []
        for idx, map_data in enumerate(maps):
            new_map = {
                'target_location': map_data['target_location']
            }

            # 如果这张地图需要遗忘，移除指定类型障碍物
            if idx < len(unlearn_map_flags) and unlearn_map_flags[idx]:
                for i in range(1, 5):
                    type_key = f'obstacles_type_{i}'
                    if i in unlearn_obstacles_type:
                        new_map[type_key] = []  # 移除这类障碍物
                    else:
                        new_map[type_key] = map_data[type_key]
            else:
                # 不需要遗忘的地图，保持原样
                for i in range(1, 5):
                    type_key = f'obstacles_type_{i}'
                    new_map[type_key] = map_data[type_key]

            # 保留边界
            new_map['obstacles_type_boundary'] = map_data['obstacles_type_boundary']
            new_maps.append(new_map)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(clean_map_file), exist_ok=True)

        # 保存新地图
        with open(clean_map_file, 'w') as f:
            json.dump(new_maps, f, indent=2)

        print(f"Created clean maps with {sum(unlearn_map_flags)} maps cleaned: {clean_map_file}")


def init_module(map_file_path, filename = None, model_dir=None, size=10, n_ob=10, unlearn_obstacles_type=None, device="cuda"):
    """
    为训练或测试模块初始化完整的环境、智能体和经验回放缓冲区
    根据是否提供模型文件决定是从头训练还是加载预训练模型

    Args:
        map_file_path (str): 地图文件路径
        filename (str, optional): 模型文件名，None表示不加载模型从头训练. Defaults to None.
        model_dir (str, optional): 模型文件所在目录. Defaults to None.
        size (int, optional): 环境网格大小. Defaults to 10.
        n_ob (int, optional): 障碍物数量. Defaults to 10.
        unlearn_obstacles_type (list, optional): 需要遗忘的障碍物类型列表. Defaults to None.
        device (str, optional): 计算设备 ('cuda' 或 'cpu'). Defaults to "cuda".

    Returns:
        tuple: (env, agent, buffer)
            - env (Simple2DEnvironment): 初始化完成的环境实例
            - agent (DQNAgent): DQN智能体实例（可能已加载模型）
            - buffer (ReplayBuffer): 经验回放缓冲区实例

    Raises:
        FileNotFoundError: 当指定的模型文件不存在时抛出
    """
    # 创建新环境
    env = Simple2DEnvironment(size, n_ob)
    env.load_maps(map_file_path)

    if unlearn_obstacles_type is not None:
        env.set_unlearn_obstacles(unlearn_obstacles_type)
    else:
        # 明确清空/不设置也行
        env.unlearn_obstacles_types = []

    state_dim = env.observation_state.shape[0]  # 值为 10
    action_dim = env.action_space.n  # n属性表示离散动作空间中可能的动作数量,action_dim的值为 4

    # 创建新agent
    buffer = ReplayBuffer(1000)
    agent = DQNAgent(state_dim, action_dim, buffer, device)

    # 如果需要，加载模型
    if filename:
        if not load_model(agent, filename, model_dir=model_dir, device=device):
            raise FileNotFoundError(f"Model not found: {os.path.join(model_dir, filename)}")
        print(f"Loaded model: {filename}")
    else:
        print("Created fresh agent (no model loaded)")

    return env, agent, buffer