import sys
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
from collections import deque
import random
import json
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import os

# 数据结构变更：将 obstacles_location 从列表改为字典，按类型分类存储
# 地图生成：按10%、20%、30%、40%比例生成不同类型的障碍物
# 保存/加载：适应新的数据结构格式
# 相关方法更新：所有需要访问障碍物的方法都需要适应新结构
# 辅助方法：添加用于后续遗忘实验的便利方法

def save_model(agent, filename, additional_info=None, model_dir=None):
    """保存模型为字典格式"""
    model_dict = {
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'gamma': agent.gamma,
        'action_dim': agent.action_dim
    }
    if additional_info:
        model_dict.update(additional_info)

    filepath = os.path.join(model_dir, filename)
    torch.save(model_dict, filepath)
    print(f"Model saved to: {filepath}")


def load_model(agent, filename, model_dir=None, device=None):
    """加载模型"""
    filepath = os.path.join(model_dir, filename)
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device, weights_only = False)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.gamma = checkpoint['gamma']
        agent.action_dim = checkpoint['action_dim']
        print(f"Model loaded from: {filepath}")
        return True
    else:
        print(f"Model file not found: {filepath}")
        return False


class ReplayBuffer():
    """
    经验回放缓冲区
    """

    def __init__(self, capacity):
        """
        这里使用了deque（双端队列）来存储经验，并设置了最大容量capacity
        当队列达到最大容量时，新添加的项目会自动删除最旧的项目
        param capacity: 最大容量
        """
        self.buffer = deque(maxlen=capacity)

    # state：10，智能体当前坐标(x,y)以及周围8个单元格的信息
    # action： 1，一个整数（0, 1, 2, 3分别代表上、下、左、右）
    # reward：1， 一个数值，表示执行动作后获得的奖励
    # next_state： 10，与state格式相同，是执行动作后的新状态
    # done： 是否结束
    def push(self, state, action, reward, next_state, done):
        # 首先对state和next_state使用np.expand_dims增加一个维度，这是为了保持数据形状的一致性
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        # 然后将五元组(state, action, reward, next_state, done)作为一个整体添加到缓冲区
        self.buffer.append((state, action, reward, next_state, done))

    # 这个方法用于从缓冲区随机抽取一批经验
    def sample(self, batch_size):
        """
        这个方法用于从缓冲区随机抽取一批经验
        """
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # np.concatenate函数将多个数组拼接成一个数组
        # 在这个上下文中，它将批量样本的状态数组合并成一个大数组,这样做的目的是为了方便后续的神经网络批量处理
        return np.concatenate(state), action, reward, np.concatenate(next_state), done  # 这个方法用于从缓冲区随机抽取一批经验

    # 获取缓冲区中的样本数量
    def __len__(self):
        """
        这个方法重载了len()函数，使得可以直接使用len(buffer)来获取缓冲区中的样本数量
        """
        return len(self.buffer)

    # 新增：专门用于Nontranslfs的push方法，记录是否撞到要遗忘的障碍物
    def push_with_collision_info(self, state, action, reward, next_state, done, hit_unlearn_obstacle):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        # 存储6个元素，最后一个是是否撞到要遗忘的障碍物
        self.buffer.append((state, action, reward, next_state, done, hit_unlearn_obstacle))

    # 新增：专门用于Nontranslfs的采样方法
    def sample_with_collision_info(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, hit_unlearn = zip(*samples)
        return (np.concatenate(state), action, reward,
                np.concatenate(next_state), done, hit_unlearn)

class DQN(nn.Module):
    """
    这是一个简单的双层神经网络，用作Q值函数的近似器：

    -输入层接收状态
    -一个隐藏层(128个神经元)
    -输出层输出每个动作的Q值

    """

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(input_dim, 128)
        self.lin2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        # 注意这里没有应用激活函数，因为在DQN算法中，Q值网络的输出通常是原始的Q值估计
        return x


class Simple2DEnvironment(gym.Env):
    """
    这个类定义了一个2D网格环境
    环境中有：智能体(状态)、目标位置、障碍物、边界
    """

    def __init__(self, size=10, n_ob=10):
        super(Simple2DEnvironment, self).__init__()
        """
        定义了一个名为Simple2DEnvironment的类，它继承自gym.Env（OpenAI Gym环境的基类）
        """
        # 初始化方法接受两个参数：size（环境大小，默认为10）和n_ob（障碍物数量，默认为10）
        self.size = size
        self.n_ob = n_ob

        # define state space 定义动作空间为4个离散动作（上、下、左、右）
        self.action_space = spaces.Discrete(4)

        # observation space 定义观察空间为10维的整型数组
        # 观察状态由以下部分组成：前两个值（索引0和1）是智能体的位置坐标 (x, y)
        # 接下来的8个值（索引2到9）是通过 get_surrounding_cells() 方法获取的智能体周围8个方向的信息与buffer里的state一致！
        self.observation_state = spaces.Box(low=-1, high=self.size, shape=(10,), dtype=np.int32)

        # 初始化状态、障碍物和目标为None
        # 定义智能体的坐标(x, y)
        self.agent_location = None
        # 定义存储环境中所有障碍物的位置列表
        self.obstacles_location = {
            'type_1': [],       # 10%
            'type_2': [],       # 20%
            'type_3': [],       # 30%
            'type_4': [],       # 40%
            'type_boundary': [] # 边界障碍物
        }
        # 定义目标位置的坐标(x, y)
        self.target_location = None

        # 创建一个空的地图列表 用于存储多个地图数据的列表，每个地图包含障碍物和目标的位置信息
        self.maps = []

    # 将地图数据保存为JSON格式
    def save_maps(self, filename):
        """保存包含分类障碍物的地图数据"""
        with open(filename, 'w') as f:
            maps = []
            for map_data in self.maps:
                map_info = {
                    'obstacles_type_1': [list(obstacle) for obstacle in map_data["obstacles_location"]['type_1']],
                    'obstacles_type_2': [list(obstacle) for obstacle in map_data["obstacles_location"]['type_2']],
                    'obstacles_type_3': [list(obstacle) for obstacle in map_data["obstacles_location"]['type_3']],
                    'obstacles_type_4': [list(obstacle) for obstacle in map_data["obstacles_location"]['type_4']],
                    'obstacles_type_boundary': [list(obstacle) for obstacle in map_data["obstacles_location"]['type_boundary']],
                    'target_location': list(map_data["target_location"])
                }
                maps.append(map_info)
            json.dump(maps, f, indent=2)

    # 从JSON文件加载地图数据
    def load_maps(self, filename):
        """加载包含分类障碍物的地图数据"""
        with open(filename, 'r') as f:
            maps = json.load(f)
            self.maps = []
            for map_data in maps:
                obstacles_dict = {
                    'type_1': [tuple(obstacle) for obstacle in map_data['obstacles_type_1']],
                    'type_2': [tuple(obstacle) for obstacle in map_data['obstacles_type_2']],
                    'type_3': [tuple(obstacle) for obstacle in map_data['obstacles_type_3']],
                    'type_4': [tuple(obstacle) for obstacle in map_data['obstacles_type_4']],
                    'type_boundary': [tuple(obstacle) for obstacle in map_data['obstacles_type_boundary']],
                }
                self.maps.append({
                    "obstacles_location": obstacles_dict,
                    "target_location": tuple(map_data["target_location"])
                })

    # 更新一下位置，最终得到observation（也就是buffer里面的state）, reward, done, {} 实现了论文里面的迁移函数
    def step(self, action):
        x, y = self.agent_location
        old_state = self.agent_location

        # 移动逻辑保持不变
        if action == 0:  # up
            new_y = min(y + 1, self.size - 1)
            y = new_y
        elif action == 1:  # down
            new_y = max(y - 1, 0)
            y = new_y
        elif action == 2:  # left
            new_x = max(x - 1, 0)
            x = new_x
        elif action == 3:  # right
            new_x = min(x + 1, self.size - 1)
            x = new_x

        self.agent_location = (x, y)

        # 合并所有障碍物进行碰撞检测
        all_obstacles = []
        for obstacle_type in self.obstacles_location.values():
            all_obstacles.extend(obstacle_type)

        # 判断结束条件
        arrive_target = (self.agent_location == self.target_location)
        arrive_land = (self.agent_location[1] == 0 and not arrive_target)
        done = arrive_target or arrive_land

        # 奖励函数 - 与obj_step保持一致
        if arrive_target:
            reward = 100
        elif self.agent_location in all_obstacles:
            reward = -10
        elif arrive_land:
            reward = -100
        else:
            reward = -5

        if self.agent_location in all_obstacles:
            self.agent_location = old_state

        surrounding_info = self.get_surrounding_cells()
        next_state = np.insert(surrounding_info, 0, self.agent_location)

        return next_state, reward, done, {}


    # 区分不同的game_type；传进来的map_index不给参数即初始化地图，如果map_index带参数即读取已有的地图。
    # 最终返回observation（也就是buffer里面的state）

    def reset(self, map_index=None, game_type=None):
        """
        改进的reset方法，确保障碍物不重复
        """
        if map_index is None:
            # 创建新地图时按比例分配障碍物，确保不重复
            if game_type == "grid_world":
                self.agent_location = (int(self.size / 2), self.size - 1)
                self.target_location = (int(self.size / 2), 0)

                # 按比例计算各类型障碍物数量
                n_ob_1 = int(self.n_ob * 0.1)  # 10%
                n_ob_2 = int(self.n_ob * 0.2)  # 20%
                n_ob_3 = int(self.n_ob * 0.3)  # 30%
                n_ob_4 = self.n_ob - n_ob_1 - n_ob_2 - n_ob_3  # 剩余的作为40%

                # 创建禁止位置集合（包括起始位置和目标位置）
                forbidden_positions = {self.agent_location, self.target_location}

                # 生成所有可能的内部位置
                all_possible_positions = []
                for x in range(1, self.size - 1):
                    for y in range(1, self.size - 1):
                        pos = (x, y)
                        if pos not in forbidden_positions:
                            all_possible_positions.append(pos)

                # 检查是否有足够的位置放置所有障碍物
                total_obstacles = n_ob_1 + n_ob_2 + n_ob_3 + n_ob_4
                if total_obstacles > len(all_possible_positions):
                    print(f"警告：可用位置({len(all_possible_positions)})少于所需障碍物数量({total_obstacles})")
                    print(f"将减少障碍物数量以适应可用空间")
                    # 按比例减少障碍物数量
                    scale_factor = len(all_possible_positions) / total_obstacles
                    n_ob_1 = int(n_ob_1 * scale_factor)
                    n_ob_2 = int(n_ob_2 * scale_factor)
                    n_ob_3 = int(n_ob_3 * scale_factor)
                    n_ob_4 = len(all_possible_positions) - n_ob_1 - n_ob_2 - n_ob_3
                    if n_ob_4 < 0:
                        n_ob_4 = 0

                # 随机打乱所有可能位置
                import random
                random.shuffle(all_possible_positions)

                # 依次分配给不同类型的障碍物
                obstacles_dict = {
                    'type_1': all_possible_positions[:n_ob_1],
                    'type_2': all_possible_positions[n_ob_1:n_ob_1 + n_ob_2],
                    'type_3': all_possible_positions[n_ob_1 + n_ob_2:n_ob_1 + n_ob_2 + n_ob_3],
                    'type_4': all_possible_positions[n_ob_1 + n_ob_2 + n_ob_3:n_ob_1 + n_ob_2 + n_ob_3 + n_ob_4]
                }

                # 生成边界障碍物（保持原有逻辑）
                top_border = [(i, -1) for i in range(self.size + 2)]
                bottom_border = [(i, self.size) for i in range(self.size + 2)]
                left_border = [(-1, i) for i in range(-1, self.size + 1)]
                right_border = [(self.size, i) for i in range(-1, self.size + 1)]

                obstacles_dict['type_boundary'] = top_border + bottom_border + left_border + right_border
                self.obstacles_location = obstacles_dict

            elif game_type == "aircraft_landing":
                # aircraft_landing的改进逻辑
                self.agent_location = (np.random.randint(self.size), self.size - 1)
                self.target_location = (np.random.randint(self.size), 0)

                # 按比例分配内部障碍物
                n_ob_1 = int(self.n_ob * 0.1)
                n_ob_2 = int(self.n_ob * 0.2)
                n_ob_3 = int(self.n_ob * 0.3)
                n_ob_4 = self.n_ob - n_ob_1 - n_ob_2 - n_ob_3

                # 创建禁止位置集合
                forbidden_positions = {self.agent_location, self.target_location}

                # 生成所有可能的内部位置
                all_possible_positions = []
                for x in range(1, self.size - 1):
                    for y in range(1, self.size - 1):
                        pos = (x, y)
                        if pos not in forbidden_positions:
                            all_possible_positions.append(pos)

                # 检查是否有足够的位置
                total_obstacles = n_ob_1 + n_ob_2 + n_ob_3 + n_ob_4
                if total_obstacles > len(all_possible_positions):
                    print(f"警告：可用位置({len(all_possible_positions)})少于所需障碍物数量({total_obstacles})")
                    print(f"将减少障碍物数量以适应可用空间")
                    scale_factor = len(all_possible_positions) / total_obstacles
                    n_ob_1 = int(n_ob_1 * scale_factor)
                    n_ob_2 = int(n_ob_2 * scale_factor)
                    n_ob_3 = int(n_ob_3 * scale_factor)
                    n_ob_4 = len(all_possible_positions) - n_ob_1 - n_ob_2 - n_ob_3
                    if n_ob_4 < 0:
                        n_ob_4 = 0

                # 随机打乱并分配位置
                import random
                random.shuffle(all_possible_positions)

                obstacles_dict = {
                    'type_1': all_possible_positions[:n_ob_1],
                    'type_2': all_possible_positions[n_ob_1:n_ob_1 + n_ob_2],
                    'type_3': all_possible_positions[n_ob_1 + n_ob_2:n_ob_1 + n_ob_2 + n_ob_3],
                    'type_4': all_possible_positions[n_ob_1 + n_ob_2 + n_ob_3:n_ob_1 + n_ob_2 + n_ob_3 + n_ob_4]
                }

                # 添加边界
                top_border = [(i, -1) for i in range(self.size + 2)]
                bottom_border = [(i, self.size) for i in range(self.size + 2)]
                left_border = [(-1, i) for i in range(-1, self.size + 1)]
                right_border = [(self.size, i) for i in range(-1, self.size + 1)]

                obstacles_dict['type_boundary'] = top_border + bottom_border + left_border + right_border
                self.obstacles_location = obstacles_dict

            map_data = {"obstacles_location": self.obstacles_location, "target_location": self.target_location}
            self.maps.append(map_data)

            # 打印统计信息
            print(f"生成地图 - 游戏类型: {game_type}")
            print(f"起始位置: {self.agent_location}, 目标位置: {self.target_location}")
            print(f"障碍物统计:")
            for obs_type, obs_list in self.obstacles_location.items():
                if obs_type != 'type_boundary':
                    print(f"  {obs_type}: {len(obs_list)} 个")

        else:
            # 加载现有地图（保持原有逻辑）
            map_data = self.maps[map_index]
            self.agent_location = (int(self.size / 2), self.size - 1) if game_type == "grid_world" else (
                np.random.randint(self.size), self.size - 1)
            self.target_location = map_data['target_location']
            self.obstacles_location = map_data['obstacles_location']

        surrounding_info = self.get_surrounding_cells()
        state = np.insert(surrounding_info, 0, self.agent_location)
        return state


    def get_surrounding_cells(self):
        surrounding = np.full(8, 0)
        x, y = self.agent_location
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

        # 将所有类型的障碍物合并为一个列表进行检查
        all_obstacles = []
        for obstacle_type in self.obstacles_location.values():
            all_obstacles.extend(obstacle_type)


        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                if (new_x, new_y) == self.agent_location:
                    surrounding[i] = 1
                elif (new_x, new_y) == self.target_location:
                    surrounding[i] = 2
                elif (new_x, new_y) in all_obstacles:
                    surrounding[i] = 3
            else:
                surrounding[i] = 3
        return surrounding


# 这些方法共同构成了一个简单的二维环境模拟，可以用于强化学习任务。
# reset 方法初始化环境，get_surrounding_cells 方法提供观察状态，
# 这整个类是一个 OpenAI Gym 环境的实现，用于训练和测试强化学习算法（如本代码中的 DQN 算法）

    def set_unlearn_obstacles(self, unlearn_types):
        """
        设置要遗忘的障碍物类型
        """
        self.unlearn_obstacles_types = [f'type_{t}' for t in unlearn_types]
        print(f"Set unlearn obstacles: {self.unlearn_obstacles_types}")

    def get_obstacles_except_unlearn(self):
        """
        获取除了要遗忘的障碍物之外的所有障碍物
        """
        remaining_obstacles = []
        for obs_type, obs_list in self.obstacles_location.items():
            if obs_type not in getattr(self, 'unlearn_obstacles_types', []):
                remaining_obstacles.extend(obs_list)
        return remaining_obstacles

    def get_unlearn_obstacles(self):
        """
        获取所有需要遗忘的障碍物位置列表

        Returns:
            list: 要遗忘的障碍物位置列表
        """
        unlearn_obstacles = []

        # 检查是否设置了要遗忘的障碍物类型
        if not hasattr(self, 'unlearn_obstacles_types'):
            print("Warning: unlearn_obstacles_types not set, returning empty list")
            return unlearn_obstacles

        # 遍历要遗忘的障碍物类型，收集所有位置
        for obs_type in self.unlearn_obstacles_types:
            if obs_type in self.obstacles_location:
                unlearn_obstacles.extend(self.obstacles_location[obs_type])

        return unlearn_obstacles

    def obj_step(self, action):
        """
        修改版的step方法，忽略指定类型的障碍物
        """
        x, y = self.agent_location
        old_state = self.agent_location

        # 移动逻辑保持不变
        if action == 0:  # up
            new_y = min(y + 1, self.size - 1)
            y = new_y
        elif action == 1:  # down
            new_y = max(y - 1, 0)
            y = new_y
        elif action == 2:  # left
            new_x = max(x - 1, 0)
            x = new_x
        elif action == 3:  # right
            new_x = min(x + 1, self.size - 1)
            x = new_x

        self.agent_location = (x, y)

        # 获取除了要遗忘的障碍物之外的所有障碍物
        active_obstacles = self.get_obstacles_except_unlearn()

        # 10.9增加，获取全部障碍物
        all_obstacles = []
        for obstacle_type in self.obstacles_location.values():
            all_obstacles.extend(obstacle_type)

        # 判断是否到目的地、是否着陆
        arrive_target = (self.agent_location == self.target_location)
        arrive_land = (self.agent_location[1] == 0 and not arrive_target)
        done = arrive_target or arrive_land

        # 只有碰到非遗忘类型的障碍物才算撞击
        if arrive_target:
            reward = 100
        elif self.agent_location in active_obstacles:
            reward = -10
        elif arrive_land:
            reward = -100
        else:
            reward = -5

        # # 只有碰到非遗忘类型的障碍物才回退
        if self.agent_location in active_obstacles:
            self.agent_location = old_state

        # 10.9修改为撞到所有障碍物都回退
        # if self.agent_location in all_obstacles:
        #     self.agent_location = old_state


        surrounding_info = self.get_surrounding_cells_obj()
        next_state = np.insert(surrounding_info, 0, self.agent_location)

        return next_state, reward, done, {}

    def step_Adversarial_Reward(self, action, reward_scale=None):
        """reward_scale
        10.14日，修改版的step方法，增加Adversarial Reward
        """
        x, y = self.agent_location
        old_state = self.agent_location

        # 移动逻辑保持不变
        if action == 0:  # up
            new_y = min(y + 1, self.size - 1)
            y = new_y
        elif action == 1:  # down
            new_y = max(y - 1, 0)
            y = new_y
        elif action == 2:  # left
            new_x = max(x - 1, 0)
            x = new_x
        elif action == 3:  # right
            new_x = min(x + 1, self.size - 1)
            x = new_x

        self.agent_location = (x, y)

        # 获取除了要遗忘的障碍物之外的所有障碍物
        active_obstacles = self.get_obstacles_except_unlearn()

        # 获取要遗忘的障碍物
        unlearn_obstacles = self.get_unlearn_obstacles()

        # 10.9增加，获取全部障碍物
        all_obstacles = []
        for obstacle_type in self.obstacles_location.values():
            all_obstacles.extend(obstacle_type)

        # 判断是否到目的地、是否着陆
        arrive_target = (self.agent_location == self.target_location)
        arrive_land = (self.agent_location[1] == 0 and not arrive_target)
        hit_unlearn = (self.agent_location in unlearn_obstacles)
        hit_active = (self.agent_location in active_obstacles)
        done = arrive_target or arrive_land

        # 只有碰到非遗忘类型的障碍物才算撞击
        if arrive_target:
            reward = 100
        elif hit_unlearn:
            # 关键改进：给予正奖励，主动鼓励"忘记"这些障碍物是危险的
            reward = reward_scale  # 可调整的正奖励
        elif hit_active :
            reward = -8
        elif arrive_land:
            reward = -100
        else:
            reward = -5

        # # 只有碰到非遗忘类型的障碍物才回退
        # if self.agent_location in active_obstacles:
        #     self.agent_location = old_state

        # 10.9修改为撞到所有障碍物都回退
        if self.agent_location in all_obstacles:
            self.agent_location = old_state


        surrounding_info = self.get_surrounding_cells_obj()
        next_state = np.insert(surrounding_info, 0, self.agent_location)

        return next_state, reward, done, {}

    def get_surrounding_cells_obj(self):
        """
        修改版的get_surrounding_cells方法，忽略指定类型的障碍物
        """
        surrounding = np.full(8, 0)
        x, y = self.agent_location
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

        # 获取除了要遗忘的障碍物之外的所有障碍物
        active_obstacles = self.get_obstacles_except_unlearn()

        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                if (new_x, new_y) == self.agent_location:
                    surrounding[i] = 1
                elif (new_x, new_y) == self.target_location:
                    surrounding[i] = 2
                elif (new_x, new_y) in active_obstacles:
                    surrounding[i] = 3
                # 遗忘的障碍物被忽略，视为空白区域
            else:
                surrounding[i] = 3  # 边界
        return surrounding


    def step_with_collision_tracking(self, action, unlearn_types):
        """
        执行动作并跟踪是否撞到要遗忘的障碍物
        返回: next_state, reward, done, hit_unlearn_obstacle
        """
        x, y = self.agent_location
        old_state = self.agent_location

        # 移动逻辑
        if action == 0:  # up
            new_y = min(y + 1, self.size - 1)
            y = new_y
        elif action == 1:  # down
            new_y = max(y - 1, 0)
            y = new_y
        elif action == 2:  # left
            new_x = max(x - 1, 0)
            x = new_x
        elif action == 3:  # right
            new_x = min(x + 1, self.size - 1)
            x = new_x

        self.agent_location = (x, y)

        # 分别获取要遗忘的障碍物和其他障碍物
        unlearn_obstacles = []
        other_obstacles = []
        for obs_type, obs_list in self.obstacles_location.items():
            if obs_type in [f'type_{t}' for t in unlearn_types]:
                unlearn_obstacles.extend(obs_list)
            else:
                other_obstacles.extend(obs_list)

        all_obstacles = unlearn_obstacles + other_obstacles

        # 判断是否撞到要遗忘的障碍物
        hit_unlearn_obstacle = (self.agent_location in unlearn_obstacles)

        # 判断结束条件
        arrive_target = (self.agent_location == self.target_location)
        arrive_land = (self.agent_location[1] == 0 and not arrive_target)
        done = arrive_target or arrive_land

        # 奖励函数
        if arrive_target:
            reward = 100
        elif self.agent_location in all_obstacles:
            reward = -10
        elif arrive_land:
            reward = -100
        else:
            reward = -5

        # 撞到障碍物就回退
        if self.agent_location in all_obstacles:
            self.agent_location = old_state

        surrounding_info = self.get_surrounding_cells()
        next_state = np.insert(surrounding_info, 0, self.agent_location)

        return next_state, reward, done, hit_unlearn_obstacle


class DQNAgent():
    """
    这个类实现了DQN算法的智能体
    初始化DQN模型、优化器等
    从经验回放缓冲区采样并更新网络
    使用epsilon-贪心策略选择动作
    """

    def __init__(self, state_dim, action_dim, replay_buffer, device):

        # 存动作空间维度到 self.action_dim
        # 设置折扣因子 gamma 为 0.99（强化学习中的重要参数，用于平衡即时奖励和未来奖励）
        # 创建 DQN 模型实例并移至指定设备（CPU或GPU）
        # 设置 Adam 优化器，用于更新神经网络的参数
        # 保存经验回放缓冲区的引用
        # 设置损失函数为均方误差（MSE），reduction='mean'表示取平均值

        self.device = device
        self.action_dim = action_dim
        self.gamma = 0.99
        self.model = DQN(state_dim, action_dim).to(self.device)
        # FIXME:只看自己与周围8个状态，视野受限，可以改进
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.replay_buffer = replay_buffer
        self.loss_fn = nn.MSELoss(reduction='mean')


    def update(self, batch_size):
        """
        这个类实现了深度 Q 网络（DQN）算法的核心组件：

        神经网络模型用于近似状态-动作价值函数;经验回放用于打破样本之间的相关性，提高训练稳定性;
        ε-贪婪策略平衡探索与利用;基于时序差分（TD）学习的网络更新
        通过这些组件，智能体能够从与环境的交互中学习最优策略，即在每个状态选择能获得最大累积奖励的动作

        """
        # 首先检查经验回放缓冲区是否有足够的样本，如果没有则直接返回
        if len(self.replay_buffer) < batch_size:
            return
        # 从经验回放缓冲区中随机采样一批数据：状态、动作、奖励、下一状态和是否结束
        # 将这些数据转换为 PyTorch 张量并移至指定设备：
        #     state 和 next_state 转为浮点型张量
        #     action 转为长整型张量
        #     reward 和 done 转为浮点型张量

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = self.loss_fn(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def Nontranslfs_update(self, batch_size):
        """
        改进的Non-transferLFS更新方法，使用准确的撞击信息
        """
        if len(self.replay_buffer) < batch_size:
            return

        # 使用新的采样方法获取撞击信息
        state, action, reward, next_state, done, hit_unlearn = self.replay_buffer.sample_with_collision_info(batch_size)

        # 转换为tensor
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # 根据hit_unlearn标志分组
        group_A_indices = [i for i, hit in enumerate(hit_unlearn) if hit]  # 撞到要遗忘的障碍物
        group_B_indices = [i for i, hit in enumerate(hit_unlearn) if not hit]  # 其他情况

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # 处理A组（要遗忘的样本）- 使用负损失
        if len(group_A_indices) > 0:
            indices_A = torch.LongTensor(group_A_indices).to(self.device)
            state_A = state[indices_A]
            action_A = action[indices_A]
            reward_A = reward[indices_A]
            next_state_A = next_state[indices_A]
            done_A = done[indices_A]

            q_values_A = self.model(state_A)
            next_q_values_A = self.model(next_state_A)
            q_value_A = q_values_A.gather(1, action_A.unsqueeze(1)).squeeze(1)
            next_q_value_A = next_q_values_A.max(1)[0]
            expected_q_value_A = reward_A + self.gamma * next_q_value_A * (1 - done_A)

            loss_A = self.loss_fn(q_value_A, expected_q_value_A.detach())
            total_loss = total_loss - loss_A  # 负损失用于遗忘

        # 处理B组（正常样本）- 使用正损失
        if len(group_B_indices) > 0:
            indices_B = torch.LongTensor(group_B_indices).to(self.device)
            state_B = state[indices_B]
            action_B = action[indices_B]
            reward_B = reward[indices_B]
            next_state_B = next_state[indices_B]
            done_B = done[indices_B]

            q_values_B = self.model(state_B)
            next_q_values_B = self.model(next_state_B)
            q_value_B = q_values_B.gather(1, action_B.unsqueeze(1)).squeeze(1)
            next_q_value_B = next_q_values_B.max(1)[0]
            expected_q_value_B = reward_B + self.gamma * next_q_value_B * (1 - done_B)

            loss_B = self.loss_fn(q_value_B, expected_q_value_B.detach())
            total_loss = total_loss + loss_B  # 正损失用于正常学习

        # 执行反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def get_action(self, state, epsilon=0.1): # 相当于论文里的Π函数

        # 探索-利用平衡：
        # 以 epsilon 的概率随机选择动作（探索）
        # 当随机选择动作时，函数返回随机动作索引和perplexity(困惑度)

        if random.random() < epsilon:
            # 随机探索时，困惑度最大（假设均匀分布）
            max_entropy = np.log(self.action_dim)  # log(4) ≈ 1.386
            return random.randrange(self.action_dim), max_entropy

        # 当选择利用现有知识时：
        # state (输入状态)
        #   ↓ torch.FloatTensor + unsqueeze(0)   # unsqueeze在指定位置添加长度为1的维度的函数。
        # [batch_size=1, state_dim=10] 的张量
        #   ↓ self.model.forward()
        # q_value: [batch_size=1, action_dim=4] 的Q值张量
        #   ↓ F.softmax()
        # softmax概率张量: [batch_size=1, action_dim=4]
        #   ↓ squeeze(0)
        # 概率张量: [action_dim=4]
        #   ↓ tolist()
        # probabilities: [prob_action0, prob_action1, prob_action2, prob_action3]

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 利用阶段：根据Q值选择动作
        q_value = self.model.forward(state) # shape: [1, 4]

        # 计算动作概率分布（softmax）
        probabilities = F.softmax(q_value, dim=1).squeeze(0)  # shape: [4]

        # 计算策略熵（困惑度）
        # Entropy = -Σ p(a) * log(p(a))
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
        perplexity = entropy.item()  # 转换为Python float

        # 选择Q值最大的动作
        action = probabilities.argmax().item()

        return action, perplexity



# def shuffle_three_parts(lst,change):
#     indices = [i for i, x in enumerate(lst[2:], start=2) if x >-5]
#
#     chosen_indices = np.random.choice(indices, change, replace=False)
#
#     np.random.shuffle(chosen_indices)
#
#     lst_new = lst.copy()
#     for i, index in enumerate(sorted(chosen_indices)):
#         lst_new[index] = lst[chosen_indices[i]]
#
#     return lst_new









