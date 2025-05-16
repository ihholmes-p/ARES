#DQN code is from the below resource:
#https://github.com/yuchen071/DQN-for-LunarLander-v2
"""
@misc{yuchen071_dqn_lunarlander,
  author       = {Yu-Chen Chou},
  title        = {DQN-for-LunarLander-v2},
  year         = {2021},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/yuchen071/DQN-for-LunarLander-v2}},
  note         = {Implementation of reinforcement learning algorithms for the OpenAI Gym environment LunarLander-v2}
}
"""

import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import base64, io

import numpy as np
from collections import deque, namedtuple

env = gym.make('LunarLander-v2')
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

#Possible reward_type values:
#random to take random actions
#delayed fot fully delayed rewards
#immediate for immediate rewards
#inferred for inferred rewards (based on reward function from infile)

def run_ll(infile="", outfile="", reward_type="immediate", epsilon_start=0.9, num_episodes=500, num_to_solve=5, metric=2):
    class QNetwork(nn.Module):
        """Actor (Policy) Model."""

        def __init__(self, state_size, action_size, seed):
            """Initialize parameters and build model.
            Params
            ======
                state_size (int): Dimension of each state
                action_size (int): Dimension of each action
                seed (int): Random seed
            """
            super(QNetwork, self).__init__()
            self.seed = torch.manual_seed(seed)
            self.fc1 = nn.Linear(state_size, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, action_size)
            
        def forward(self, state):
            """Build a network that maps state -> action values."""
            x = self.fc1(state)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            return self.fc3(x)

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class Agent():
        """Interacts with and learns from the environment."""

        def __init__(self, state_size, action_size, seed):
            """Initialize an Agent object.
            
            Params
            ======
                state_size (int): dimension of each state
                action_size (int): dimension of each action
                seed (int): random seed
            """
            self.state_size = state_size
            self.action_size = action_size
            self.seed = random.seed(seed)

            # Q-Network
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

            # Replay memory
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
            # Initialize time step (for updating every UPDATE_EVERY steps)
            self.t_step = 0
        
        def step(self, state, action, reward, next_state, done):
            # Save experience in replay memory
            self.memory.add(state, action, reward, next_state, done)
            
            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

        def act(self, state, eps=0.):
            """Returns actions for given state as per current policy.
            
            Params
            ======
                state (array_like): current state
                eps (float): epsilon, for epsilon-greedy action selection
            """
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            if (reward_type == "random"):
                return random.choice(np.arange(self.action_size))
            # Epsilon-greedy action selection
            if random.random() > eps:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))

        def learn(self, experiences, gamma):
            """Update value parameters using given batch of experience tuples.

            Params
            ======
                experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
            # Obtain random minibatch of tuples from D
            states, actions, rewards, next_states, dones = experiences

            ## Compute and minimize the loss
            ### Extract next maximum estimated value from target network
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            ### Calculate target value from bellman equation
            q_targets = rewards + gamma * q_targets_next * (1 - dones)
            ### Calculate expected value from local network
            q_expected = self.qnetwork_local(states).gather(1, actions)
            
            ### Loss calculation (we used Mean squared error)
            loss = F.mse_loss(q_expected, q_targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

        def soft_update(self, local_model, target_model, tau):
            """Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target

            Params
            ======
                local_model (PyTorch model): weights will be copied from
                target_model (PyTorch model): weights will be copied to
                tau (float): interpolation parameter 
            """
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    class ReplayBuffer:
        """Fixed-size buffer to store experience tuples."""

        def __init__(self, action_size, buffer_size, batch_size, seed):
            """Initialize a ReplayBuffer object.

            Params
            ======
                action_size (int): dimension of each action
                buffer_size (int): maximum size of buffer
                batch_size (int): size of each training batch
                seed (int): random seed
            """
            self.action_size = action_size
            self.memory = deque(maxlen=buffer_size)  
            self.batch_size = batch_size
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
            self.seed = random.seed(seed)
        
        def add(self, state, action, reward, next_state, done):
            """Add a new experience to memory."""
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
        
        def sample(self):
            """Randomly sample a batch of experiences from memory."""
            experiences = random.sample(self.memory, k=self.batch_size)

            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
    
            return (states, actions, rewards, next_states, dones)

        def __len__(self):
            """Return the current size of internal memory."""
            return len(self.memory)



    model = []
    all_rewards = []
    if (reward_type == "inferred"):
        with open(infile, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:  # skip empty lines
                    continue
                s1_str, s2_str, s3_str, s4_str, s5_str, s6_str, s7_str, s8_str, action_str, value_str = line.split(',')
                s1 = float(s1_str) #* 10000000
                s2 = float(s2_str) #* 1000000
                s3 = float(s3_str) #* 100000
                s4 = float(s4_str) #* 10000
                s5 = float(s5_str) #* 1000
                s6 = float(s6_str) #* 100
                s7 = float(s7_str) #* 10
                s8 = float(s8_str) #* 1
                action = float(action_str)
                reward_value = float(value_str)
                model.append([s1, s2, s3, s4, s5, s6, s7, s8, action, reward_value])
                all_rewards.append(reward_value)

        # Compute the global maximum absolute value
        global_max = max(all_rewards)
        global_min = min(all_rewards)
        global_abs_max = max(global_max, abs(global_min))

        # Normalize each reward in the model so that the maximum absolute value becomes 1
        for entry in model:
            entry[9] = entry[9] / global_abs_max

    def get_closest_key(dictionary, key):
        """Return the key in 'dictionary' closest to the supplied 'key'."""
        return min(dictionary.keys(), key=lambda k: abs(k - key))

    from scipy.spatial import KDTree
    # First, build a mapping from action value to model entries
    action_groups = {}
    for entry in model:
        act = entry[8]
        action_groups.setdefault(act, []).append(entry)
    # Then, build a KDTree for each group
    trees = {}
    for act, entries in action_groups.items():
        data = np.array([[e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]] for e in entries])
        trees[act] = KDTree(data)

    def find_closest_state_kdtree_constrained(s):
        """
        s is expected to be a list with at least 5 elements: 
        [state1, state2, state3, state4, action, ...]
        This function finds the model entry (list of 6 values) whose first four
        values are closest to s[0:4] and whose action (index 4) equals s[4].
        """
        desired_action = s[8]
        if desired_action not in trees:
            return None  # no model entries with that action value
        # Query the KD-tree for the desired action using the state part
        state_query = s[0:8]
        
        # multiplier = 1
        # for i in range(len(state_query)):
        #     state_query[i] = state_query[i] * multiplier
        #     multiplier = multiplier * 10

        dist, idx = trees[desired_action].query(state_query, p=metric)
        return action_groups[desired_action][idx]


    def dqn(n_episodes=num_episodes, max_t=1000, eps_start=0.9, eps_end=0.01, eps_decay=0.995):
    #def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Deep Q-Learning.
        
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=10)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        episodes = []
        num_solved = 0
        scores_to_return = [0]
        for i_episode in range(1, n_episodes+1):
            state, info = env.reset()
            score = 0
            episodes.append([])
            sum_reward = 0
            for t in range(max_t):
                action = agent.act(state, eps)
                append_state = torch.tensor([round(s.item(), 4) for s in state], dtype=torch.float32, device=device).unsqueeze(0)
                append_action = torch.tensor([action], dtype=torch.float32, device=device).unsqueeze(0)
                episodes[-1].append(torch.cat((append_state, append_action), dim=1).squeeze().tolist())
                next_state, reward, done, truncated, _ = env.step(action)
                true_reward = reward

                if (reward_type == "inferred"):
                    s = append_state.squeeze(0).tolist()
                    s.append(action.item())
                    closest = find_closest_state_kdtree_constrained(s)
                    reward = closest[9]
                    # if done:
                    #     reward = 1

                if (reward_type == "delayed"):
                    sum_reward += true_reward
                    reward = 0
                    if done:
                        reward = sum_reward

                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += true_reward
                if done or truncated:
                    episodes[-1].append(score)
                    break 
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 10 == 0:
                scores_to_return.append(np.mean(scores_window))
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score over last 10 episodes: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window)>=200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                # break
            if (score >= 200):
                print("solved episode")
                print(score)
                num_solved += 1
                if (num_solved == num_to_solve):
                    if (not outfile == ""):
                        with open(outfile, 'w') as file:
                            for episode in episodes:
                                for step in episode:
                                    if isinstance(step, list):
                                        formatted_step = [f"{value:.2f}" if isinstance(value, float) else str(value) for value in step]
                                        file.write("[" + ", ".join(formatted_step) + "]\n")
                                    else:
                                        file.write(f"{step:.2f}\n")
                                file.write("\n")

                    return True, scores_to_return

        # if (not outfile == ""):
        #     with open(outfile, 'w') as file:
        #         for episode in episodes:
        #             for step in episode:
        #                 if isinstance(step, list):
        #                     formatted_step = [f"{value:.2f}" if isinstance(value, float) else str(value) for value in step]
        #                     file.write("[" + ", ".join(formatted_step) + "]\n")
        #                 else:
        #                     file.write(f"{step:.2f}\n")
        #             file.write("\n")

        return False, scores_to_return

    agent = Agent(state_size=8, action_size=4, seed=0)
    scores = dqn()
    return scores

if __name__ == "__main__":

    # run_ll(infile="", outfile="ll_expert_solve5_output1.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=2000, num_to_solve=5)
    # run_ll(infile="", outfile="ll_expert_solve5_output2.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=2000, num_to_solve=5)
    # run_ll(infile="", outfile="ll_expert_solve5_output3.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=2000, num_to_solve=5)
    # run_ll(infile="", outfile="ll_expert_solve5_output4.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=2000, num_to_solve=5)
    # run_ll(infile="", outfile="ll_expert_solve5_output5.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=2000, num_to_solve=5)
    # run_ll(infile="", outfile="ll_expert_solve5_output6.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=2000, num_to_solve=5)
    # run_ll(infile="", outfile="ll_expert_solve5_output7.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=2000, num_to_solve=5)
    # run_ll(infile="", outfile="ll_expert_solve5_output8.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=2000, num_to_solve=5)
    # run_ll(infile="", outfile="ll_expert_solve5_output9.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=2000, num_to_solve=5)
    # run_ll(infile="", outfile="ll_expert_solve5_output10.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=2000, num_to_solve=5)

    run_ll(infile="ll_expert_solve5_rewards3.txt", outfile="test.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=2000, num_to_solve=5)
    exit()