import gym, json, torch, warnings
from loguru import logger
from config import CONFIG, env_info, mujoco_envs
from buffer import ReplayBuffer
from train import train_ddpg
from agent import DDPGAgent
warnings.filterwarnings('ignore')

from visualizer import visualize_cum_rewards



def main(env_name=None, cuda_device=None):
    config = CONFIG()
    config.random_seed(config.seed)
    config.env_name = env_name if env_name else config.env_name
    if cuda_device:
        config.DEVICE = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
    replay_buffer = ReplayBuffer(capacity=config.capacity)
    
    logger.add(f"logs/{config.env_name}.log", format="{time} - {level} - {message}", rotation="500 MB", compression="zip", enqueue=True)
    config.logger = logger
    env = gym.make(config.env_name)
    
    # Determining the sizes from the environment
    observation_space = env_info[config.env_name]['observation_space']
    action_space = env_info[config.env_name]['action_space']
    observation_space_discrete = env_info[config.env_name]['observation_space_discrete']
    action_space_discrete = env_info[config.env_name]['action_space_discrete']
    max_action = env_info[config.env_name]['action_range'][1]
    max_action = None if not max_action else max_action[0] # since all the max actions are the same
    min_action = env_info[config.env_name]['action_range'][0]
    min_action = None if not min_action else min_action[0] # since all the min actions are the same
    assert not action_space_discrete, "Only continuous action spaces are supported"
    config.observation_space = action_space_discrete
    config.observation_space_discrete = observation_space_discrete
    config.max_action = max_action
    config.min_action = min_action
    config.print_all()
    
    # create agent
    agent = DDPGAgent(
        state_dim=observation_space, 
        action_dim=action_space, 
        replay_buffer=replay_buffer,
        config=config
    )
    
    episode_rewards = train_ddpg(env, agent, config)
    
    with open(f'data/{config.suffix()}_rewards.json', 'w') as f:
        json.dump({config.env_name: episode_rewards}, f, ensure_ascii=False, indent=4)
    
    visualize_cum_rewards(episode_rewards, additional_info=config.suffix())
    visualize_cum_rewards(episode_rewards, additional_info=config.suffix(), smooth_rate=50)

if __name__ == '__main__':
    env_name = mujoco_envs[0]
    cuda_device = 1
    main(env_name, cuda_device)