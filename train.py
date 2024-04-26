
from tqdm import trange
import numpy as np

from agent import DDPGAgent
from buffer import ReplayBuffer

def one_hot_state(state, observation_space):
    s = np.zeros(observation_space)
    s[state] = 1
    return s

def train_ddpg(env, agent, config):
    episode_rewards = []
    for episode in trange(config.episodes):
        state, info = env.reset()
        state = one_hot_state(state, config.observation_space) if config.observation_space_discrete else state
        episode_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, trucated, _ = env.step(action)
            next_state = one_hot_state(next_state, config.observation_space) if config.observation_space_discrete else next_state
            episode_reward += reward
            agent.replay_buffer.push(state, next_state, action, reward, done)
            state = next_state
            agent.train()
            if done or trucated:
                break
        
        episode_rewards.append(episode_reward)
        if (episode + 1) % config.print_interval == 0:
            print(f'Episode {episode + 1}: Total Reward = {episode_reward}')
            
    return episode_rewards

def main():
    state_dim = 3  # Example: Adjust according to specific environment
    action_dim = 1  # Example: Adjust according to specific environment
    max_action = 1  # Example: Adjust according to specific environment

    agent = DDPGAgent(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(max_size=10000)

    # Example for a specific gym environment, adjust as necessary
    env_name = 'MountainCarContinuous-v0'
    rewards = train_ddpg(env_name, agent, replay_buffer)
    print('Training completed. Rewards:', rewards)

if __name__ == '__main__':
    main()
