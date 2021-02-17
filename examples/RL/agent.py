import dm_env
import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim
import typing

from .utils.memory import Experience, ReplayMemory, PrioritizedReplayMemory
from .models.qnet_MCdrop import Dqn, DuelDQN


# from qnet import Dqn, DuelDQN


class Agent:
    def __init__(self,
                 action_spec: dm_env.specs.DiscreteArray,
                 observation_spec: dm_env.specs.Array,
                 device: torch.device,
                 settings: dict) -> None:
        """
        Initializes the agent,  constructs the qnet and the q_target, initializes the optimizer and ReplayMemory.
        Args:
            action_spec(dm_env.specs.DiscreteArray): description of the action space of the environment
            observation_spec(dm_env.specs.Array): description of observations form the environment
            device(str): "gpu" or "cpu"
            settings(dict): dictionary with settings
        """
        self.device = device
        action_size = action_spec.num_values
        state_size = np.prod(observation_spec.shape)
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size = settings['batch_size']
        self.noisy_nets = settings['qnet_settings']['noisy_nets']

        if settings["duelling_dqn"]:
            self.qnet = DuelDQN(state_size, action_size, settings['qnet_settings']).to(device)
            self.q_target = DuelDQN(state_size, action_size, settings['qnet_settings']).to(device)
        else:
            self.qnet = Dqn(state_size, action_size, settings['qnet_settings']).to(device)
            self.q_target = Dqn(state_size, action_size, settings['qnet_settings']).to(device)
            self.drop_porb = 0

        self.q_target.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=settings['lr'])

        self.epsilon = settings["epsilon_start"]
        self.decay = settings["epsilon_decay"]
        self.epsilon_min = settings["epsilon_min"]
        self.gamma = settings['gamma']

        self.start_optimization = settings["start_optimization"]
        self.update_qnet_every = settings["update_qnet_every"]
        self.update_target_every = settings["update_target_every"]
        self.number_steps = 0
        self.ddqn = settings["ddqn"]

        # Initialize replay memory
        self.prioritized_replay = settings["prioritized_buffer"]
        if self.prioritized_replay:
            self.memory = PrioritizedReplayMemory(device, settings["buffer_size"], self.gamma, settings["n_steps"],
                                                  settings["alpha"], settings["beta0"], settings["beta_increment"])
        else:
            self.memory = ReplayMemory(device, settings["buffer_size"], self.gamma, settings["n_steps"])
        return

    def select_action(self, timestep: dm_env.TimeStep) -> int:
        """
        Returns an action following an epsilon-greedy policy.
        Args:
            timestep(dm_env.TimeStep): An observation from the environment

        Returns:
            int: The chosen action.
        """
        observation = np.array(timestep.observation).flatten()
        observation = torch.from_numpy(observation).float().to(self.device)
        self.number_steps += 1

        if not self.noisy_nets:
            self.update_epsilon()

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return int(self.qnet.get_max_action(observation))

    def update_epsilon(self) -> None:
        """
        Decays epsilon until self.epsilon_min
        Returns:
            None
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay

    @staticmethod
    def calc_loss(q_observed: torch.Tensor,
                  q_target: torch.Tensor,
                  weights: torch.Tensor) -> typing.Tuple[torch.Tensor, np.float64]:
        """
        Returns the mean weighted MSE loss and the loss for each sample
        Args:
            q_observed(torch.Tensor): calculated q_value
            q_target(torch.Tensor):   target q-value
            weights: weights of the batch samples

        Returns:
            tuple(torch.Tensor, np.float64): mean squared error loss, loss for each indivdual sample
        """
        losses = functional.mse_loss(q_observed, q_target, reduction='none')
        loss = (weights * losses).sum() / weights.sum()
        return loss, losses.cpu().detach().numpy() + 1e-8

    def update(self,
               step: dm_env.TimeStep,
               action: int,
               next_step: dm_env.TimeStep) -> None:
        """
        Adds experience to the replay memory, performs an optimization_step and updates the q_target neural network.
        Args:
            step(dm_env.TimeStep): Current observation from the environment
            action(int): The action that was performed by the agent.
            next_step(dm_env.TimeStep): Next observation from the environment
        Returns:
            None
        """

        observation = np.array(step.observation).flatten()
        next_observation = np.array(next_step.observation).flatten()
        done = next_step.last()
        exp = Experience(observation,
                         action,
                         next_step.reward,
                         next_step.discount,
                         next_observation,
                         0,
                         done
                         )
        self.memory.add(exp)

        if self.memory.number_samples() < self.start_optimization:
            return

        if self.number_steps % self.update_qnet_every == 0:
            s0, a0, n_step_reward, discount, s1, _, dones, indices, weights = self.memory.sample_batch(self.batch_size)
            self.optimization_step(s0, a0, n_step_reward, discount, s1, indices, weights)

        if self.number_steps % self.update_target_every == 0:
            self.q_target.load_state_dict(self.qnet.state_dict())
        return

    def optimization_step(self,
                          s0: torch.Tensor,
                          a0: torch.Tensor,
                          n_step_reward: torch.Tensor,
                          discount: torch.Tensor,
                          s1: torch.Tensor,
                          indices: typing.Optional[torch.Tensor],
                          weights: typing.Optional[torch.Tensor]) -> None:
        """
        Calculates the Bellmann update and updates the qnet.
        Args:
            s0(torch.Tensor): current state
            a0(torch.Tensor): current action
            n_step_reward(torch.Tensor): n-step reward
            discount(torch.Tensor): discount factor
            s1(torch.Tensor): next state
            indices(torch.Tensor): batch indices, needed for prioritized replay. Not used yet.
            weights(torch.Tensor): weights needed for prioritized replay

        Returns:
            None
        """

        with torch.no_grad():
            if self.noisy_nets:
                self.q_target.reset_noise()
                self.qnet.reset_noise()

            # Calculating the target values
            next_q_vals = self.q_target(s1)
            if self.ddqn:
                a1 = torch.argmax(self.qnet(s1), dim=1).unsqueeze(-1)
                next_q_val = next_q_vals.gather(1, a1).squeeze()
            else:
                next_q_val = torch.max(next_q_vals, dim=1).values
            q_target = n_step_reward.squeeze() + self.gamma * discount.squeeze() * next_q_val

        # Getting the observed q-values
        if self.noisy_nets:
            self.qnet.reset_noise()
        q_observed = self.qnet(s0).gather(1, a0.long()).squeeze()

        # Calculating the losses
        if not self.prioritized_replay:
            weights = torch.ones(self.batch_size)
        critic_loss, batch_loss = self.calc_loss(q_observed, q_target, weights.to(self.device))

        # Backpropagation of the gradients
        self.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 5)
        self.optimizer.step()

        # Update replay memory
        self.memory.update_priorities(indices, batch_loss)
        return
