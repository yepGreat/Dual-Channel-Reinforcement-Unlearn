from utils import *
import argparse


def get_default_max_steps(game_type, module_type):
    """根据游戏类型和模块类型返回合适的最大步数"""
    if game_type == 'grid_world':
        defaults = {
            'train_normal': 25,
            'test_normal': 40, # seed28:(25,40),best
            # 'train_SPU': 25,
            # 'test_SPU': 40,
            # 'train_SSP': 50,
            # 'test_SSP': 60,
            'train_retrain': 30, # (20,30),best
            'test_retrain': 40,
            # 'train_Nontranslfs': 70,
            # 'test_Nontranslfs': 90,
            # 'verify_normal':40,

        }
    elif game_type == 'aircraft_landing':
        defaults = {
            'train_normal': 100,
            'test_normal': 120,
            'train_SPU': 50,
            'test_SPU': 60,
            'train_SSP': 50,
            'test_SSP': 60,
            'train_retrain': 80,
            'test_retrain': 100,
            'train_Nontranslfs': 100,
            'test_Nontranslfs': 120,
        }
    return defaults.get(module_type, 100)

def seed_control(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
