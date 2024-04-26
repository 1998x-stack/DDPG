
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import Actor, Critic

class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck过程用于生成噪声。

    参数：
        size (int): 噪声的维度。
        mu (float, 可选): 均值，默认为0。
        theta (float, 可选): 控制回归到均值的速度，默认为0.15。
        sigma (float, 可选): 控制噪声强度的标准差，默认为0.2。

    属性：
        mu (ndarray): 均值。
        theta (float): 控制回归到均值的速度。
        sigma (float): 控制噪声强度的标准差。
        state (ndarray): 当前状态。

    方法：
        reset(): 重置状态为均值。
        noise(): 生成噪声。

    示例：
        >>> noise = OrnsteinUhlenbeckNoise(2)
        >>> noise.reset()
        >>> noise.noise()
        array([-0.1, 0.3])
    """
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        """重置状态为均值。"""
        self.state = np.copy(self.mu)

    def noise(self):
        """生成噪声。"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    

class DDPGAgent:
    def __init__(self, state_dim, action_dim, replay_buffer, config):
        actor_lr = config.actor_lr
        critic_lr = config.critic_lr
        self.DEVICE = config.DEVICE
        self.actor = Actor(state_dim, action_dim, config.max_action, config.min_action).to(self.DEVICE)
        self.actor_target = Actor(state_dim, action_dim, config.max_action, config.min_action).to(self.DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = Critic(state_dim, action_dim).to(self.DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(self.DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.max_action = config.max_action
        self.min_action = config.min_action
        self.replay_buffer = replay_buffer
        self.noise = OrnsteinUhlenbeckNoise(action_dim)
        
        self.iterations = config.iterations
        self.batch_size = config.batch_size
        self.discount_factor = config.discount_factor
        self.tau = config.tau

    def select_action(self, state):
        """
        根据当前状态选择动作。

        参数：
            state (ndarray): 当前状态。

        返回：
            action (ndarray): 选择的动作。

        示例：
            >>> agent = DDPGAgent(state_dim, action_dim, max_action, replay_buffer, config)
            >>> state = np.array([0.1, 0.2, 0.3])
            >>> action = agent.select_action(state)
            >>> action
            array([-0.5, 0.4, -0.3])
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.DEVICE)
        action = self.actor(state).cpu().data.numpy().flatten()
        noise = self.noise.noise()
        if self.max_action:
            return np.clip(action + noise, self.min_action, self.max_action)
        elif self.max_action is None and self.min_action is None:
            return action + noise

    def train(self):
        """
        使用DDPG算法训练智能体。

        参数：
            replay_buffer (ReplayBuffer): 经验回放缓冲区。
            iterations (int): 训练迭代次数。
            size (int, 可选): 批次大小，默认为100。
            discount (float, 可选): 折扣因子，默认为0.99。
            tau (float, 可选): 软更新参数，默认为0.005。

        示例：
            >>> agent = DDPGAgent(state_dim, action_dim, max_action, replay_buffer, config)
            >>> agent.train(replay_buffer, iterations=1000, size=64, discount=0.95, tau=0.001)
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        for it in range(self.iterations):
            states, next_states, actions, rewards, dones = self.replay_buffer.sample(self.batch_size)

            states = states.to(self.DEVICE)
            actions = actions.to(self.DEVICE)
            next_states = next_states.to(self.DEVICE)
            rewards = rewards.to(self.DEVICE)
            dones = dones.to(self.DEVICE)

            target_Q = self.critic_target(next_states, self.actor_target(next_states))
            target_Q = rewards + ((1 - dones) * self.discount_factor * target_Q).detach()

            current_Q = self.critic(states, actions)

            critic_loss = F.mse_loss(current_Q, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)