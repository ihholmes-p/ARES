#The DQN code is from the below source:
"""
Reinforcement Learning (DQN) Tutorial
https://github.com/pytorch/tutorials/blob/main/intermediate_source/reinforcement_q_learning.py
=====================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_
            `Mark Towers <https://github.com/pseudo-rnd-thoughts>`_

"""

import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def run_cartpole(infile="", outfile="", reward_type="immediate", epsilon_start=0.9, num_episodes=200, repeat=False, rndm=False):

    env = gym.make("CartPole-v1")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))


    class ReplayMemory(object):

        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)

        def push(self, *args):
            """Save a transition"""
            self.memory.append(Transition(*args))

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

    class DQN(nn.Module):

        def __init__(self, n_observations, n_actions):
            super(DQN, self).__init__()
            self.layer1 = nn.Linear(n_observations, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, n_actions)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = epsilon_start
    EPS_END = 0.01
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    steps_done = 0

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        eps_threshold = 0.01 + (EPS_START * (0.999 ** steps_done))
        steps_done += 1
        #print(eps_threshold)
        # if (steps_done % 100 == 0):
        #     print(eps_threshold)
        if (rndm == True):
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


    episode_durations = []

    def optimize_model(rewards=None):
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    episodes = []
    gate = 0

    #inferred reward here
    # model = infer_rewards()
    # model = {}

    model = []
    all_rewards = []

    if (reward_type == "inferred"):
        with open(infile, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:  # skip empty lines
                    continue
                s1_str, s2_str, s3_str, s4_str, action_str, value_str = line.split(',')
                s1 = float(s1_str)
                s2 = float(s2_str)
                s3 = float(s3_str)
                s4 = float(s4_str)
                action = float(action_str)
                reward_value = float(value_str)
                model.append([s1, s2, s3, s4, action, reward_value])
                all_rewards.append(reward_value)

        # Compute the global maximum absolute value
        global_max = max(all_rewards)
        global_min = min(all_rewards)
        global_abs_max = max(global_max, abs(global_min))

        # Normalize each reward in the model so that the maximum absolute value becomes 1
        for entry in model:
            entry[5] = entry[5] / global_abs_max




    def get_closest_key(dictionary, key):
        """Return the key in 'dictionary' closest to the supplied 'key'."""
        return min(dictionary.keys(), key=lambda k: abs(k - key))

    from scipy.spatial import KDTree
    # First, build a mapping from action value to model entries
    action_groups = {}
    for entry in model:
        act = entry[4]
        action_groups.setdefault(act, []).append(entry)
    # Then, build a KDTree for each group
    trees = {}
    for act, entries in action_groups.items():
        data = np.array([[e[0], e[1], e[2], e[3]] for e in entries])
        trees[act] = KDTree(data)

    def find_closest_state_kdtree_constrained(s):
        """
        s is expected to be a list with at least 5 elements: 
        [state1, state2, state3, state4, action, ...]
        This function finds the model entry (list of 6 values) whose first four
        values are closest to s[0:4] and whose action (index 4) equals s[4].
        """
        desired_action = s[4]
        if desired_action not in trees:
            return None  # no model entries with that action value
        # Query the KD-tree for the desired action using the state part
        state_query = s[0:4]
        dist, idx = trees[desired_action].query(state_query)
        return action_groups[desired_action][idx]




    episode_durations = []
    batch_reward = 0
    flag_500 = False
    for i_episode in range(num_episodes):
        inferred_episode_reward = 0
        # Initialize the environment and get its state
        state, info = env.reset()

        state = [round(s, 4) for s in state]

        sum_reward = 0
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        if (i_episode > gate):
            episodes.append([])

        for t in count():

            action = select_action(state)

            if (i_episode > gate):
                append_state = torch.tensor([round(s.item(), 4) for s in state.squeeze(0)], dtype=torch.float32, device=device).unsqueeze(0)
                episodes[-1].append(torch.cat((append_state, action), dim=1).squeeze().tolist())

            observation, reward, terminated, truncated, _ = env.step(action.item())

            sum_reward += reward
            batch_reward += reward 

            if (reward_type == "delayed"):
                if (terminated):
                    reward = sum_reward
                else:
                    reward = 0

            if (reward_type == "inferred"):
                s = state.squeeze(0).tolist()
                s.append(action.item())
                closest = find_closest_state_kdtree_constrained(s)
                reward = closest[5]
                inferred_episode_reward += reward

            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if done:
                next_state = None
                if (i_episode > gate):
                    episodes[-1].append(sum_reward)
            else:
                observation = [round(o, 4) for o in observation]
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if (truncated):
                flag_500 = True

            if done:
                episode_durations.append(t + 1)

                if (i_episode % 50 == 0):
                    print(i_episode, batch_reward / 50)
                    batch_reward = 0

                break

        if (flag_500 == True):
            break

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

    if (repeat == False or flag_500 == True):
        return episode_durations, flag_500
    else:
        return run_cartpole(infile=infile, outfile=outfile, reward_type=reward_type, epsilon_start=epsilon_start, num_episodes=num_episodes)
        
    print('Complete')

if __name__ == "__main__":

    print(run_cartpole(infile="", outfile="test.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=500, repeat=False, rndm=False)[0])
    exit()

    print(run_cartpole(infile="", outfile="cartpole_expert_new_output1.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=500, repeat=True, rndm=False)[0])
    print(run_cartpole(infile="", outfile="cartpole_expert_new_output2.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=500, repeat=True, rndm=False)[0])
    print(run_cartpole(infile="", outfile="cartpole_expert_new_output3.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=500, repeat=True, rndm=False)[0])
    print(run_cartpole(infile="", outfile="cartpole_expert_new_output4.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=500, repeat=True, rndm=False)[0])
    print(run_cartpole(infile="", outfile="cartpole_expert_new_output5.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=500, repeat=True, rndm=False)[0])
    print(run_cartpole(infile="", outfile="cartpole_expert_new_output6.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=500, repeat=True, rndm=False)[0])
    print(run_cartpole(infile="", outfile="cartpole_expert_new_output7.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=500, repeat=True, rndm=False)[0])
    print(run_cartpole(infile="", outfile="cartpole_expert_new_output8.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=500, repeat=True, rndm=False)[0])
    print(run_cartpole(infile="", outfile="cartpole_expert_new_output9.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=500, repeat=True, rndm=False)[0])
    print(run_cartpole(infile="", outfile="cartpole_expert_new_output10.txt", reward_type="immediate", epsilon_start=0.9, num_episodes=500, repeat=True, rndm=False)[0])
