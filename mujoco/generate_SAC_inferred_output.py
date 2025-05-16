import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
import warnings
import sys
import argparse
from scipy.spatial import KDTree

global k_neighbors
global p_state
global p_action
global env_name
global state_dim
global act_dim
global total_dim
global n_envs
global n_timesteps

# Suppress all future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def find_closest_states(query_state):
    """
    Find the k closest states in the dataset.
    
    Args:
        query_state: Array or list of state values (11 elements)
        k: Number of neighbors to return
        p: Distance metric (1=Manhattan, 2=Euclidean)
        
    Returns:
        List of indices of model entries corresponding to the closest states
    """
    query_state_array = np.array([query_state])
    dist, idx = state_tree.query(query_state_array, k_neighbors, p_state)
    return idx[0]  # Return indices of k closest states

def find_closest_action_from_states(query_action, state_indices):
    """
    Find the closest action among the given state indices.
    
    Args:
        query_action: Tuple or list of 3 floats representing the action
        state_indices: List of indices into the model array
        p: Distance metric power (1=Manhattan, 2=Euclidean)
        
    Returns:
        The model entry that best matches the action with the specified state constraint
    """
    min_dist = float('inf')
    best_entry = None
    
    if np.isscalar(state_indices):
        state_indices = [state_indices]

    for idx in state_indices:
        entry = model[idx]
        action = entry[state_dim:total_dim]
        # Calculate distance between actions using the specified metric
        if p_action == 1:
            # Manhattan distance
            dist = sum(abs(a1 - a2) for a1, a2 in zip(query_action, action))
        else:
            # Euclidean or other p-norm distance
            dist = sum(abs(a1 - a2) ** p_action for a1, a2 in zip(query_action, action)) ** (1/p_action)
        
        if dist < min_dist:
            min_dist = dist
            best_entry = entry
    
    return best_entry

def find_closest_state_action_reversed(state, action):
    """
    Find the closest state first, then the closest action.
    
    Args:
        state: Array or list of state values (11 elements)
        action: Tuple or list of 3 floats representing the action
        k_neighbors: Number of neighbors to consider
        p_state: Distance metric for state matching
        p_action: Distance metric for action matching
        
    Returns:
        The model entry that best matches the state and action
    """
    # First find the closest k states
    state_indices = find_closest_states(state)
    # Then find the closest action among those states
    return find_closest_action_from_states(action, state_indices)

class InferredRewardWrapper(gym.Wrapper):
    """Wrapper that implements the shaped rewards."""
    
    def __init__(self, env):
        super().__init__(env)
        self.is_vector_env = hasattr(self.env, 'num_envs')
        self.true_rewards = None  # Will store the true rewards from last step
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, rewards, terminated, truncated, infos = self.env.step(action)
        
        # Store the true rewards for logging
        self.true_rewards = rewards.copy() if self.is_vector_env else rewards
        
        # For single environments
        # Extract state from observation and convert action to tuple for KDTree lookup
        state = obs.tolist()[:state_dim]  # Assuming first 11 elements are the state
        act_tuple = tuple(action.tolist() if hasattr(action, 'tolist') else action)
        
        # Infer reward from KDTree model
        inferred_reward = find_closest_state_action_reversed(
            state, act_tuple, 
        )[total_dim]
        
        return obs, inferred_reward, terminated, truncated, infos

class LogAndSaveCallback(BaseCallback):
    def __init__(self, check_freq, outfile, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.episode_count = 0
        self.episode_rewards = []
        self.current_episode_rewards = [0] * n_envs
        self.episode_lists = [[[]] for i in range(n_envs)] #a list that starts with i lists, each of those lists starting with 1 empty list
        self.episode_lists_to_log = [[] for i in range(n_envs)] #a list that starts with i lists
        self.start_flags = [True for i in range(n_envs)]
        self.outfile = outfile
        self.timestep_count = 0

    def _on_rollout_start(self) -> None:
        for i, flag in enumerate(self.start_flags):
            if (self.start_flags[i] == True):
                initial_state = self.training_env.get_original_obs().copy()[i]
                self.episode_lists[i][-1].append([initial_state.tolist()])
                self.start_flags[i] = False

    def _on_step(self) -> bool:
        for act, episode_list in zip(self.locals["actions"], self.episode_lists):
            episode_list[-1][-1][-1].extend(act.tolist())
            self.timestep_count += 1

        for obs, episode_list in zip(self.training_env.get_original_obs(), self.episode_lists):
            episode_list[-1][-1].append(obs.tolist())

        # for i, reward in enumerate(self.locals["rewards"]):
        #     self.current_episode_rewards[i] += reward

        true_rewards = self.training_env.get_attr('true_rewards')
        for i, reward in enumerate(true_rewards):
            self.current_episode_rewards[i] += reward

        # for i, reward in enumerate(self.training_env.get_original_reward()):
        #     self.current_episode_rewards[i] += reward

        # Check if episodes are done
        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_count += 1
                self.episode_rewards.append(self.current_episode_rewards[i])
                self.episode_lists[i][-1][-1].append(self.current_episode_rewards[i])
                self.current_episode_rewards[i] = 0
                self.episode_lists_to_log[i].append(self.episode_lists[i])
                self.start_flags[i] = True
                
        if self.timestep_count % 10000 == 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            print(f"Timestep {self.timestep_count}: Average score over last 100 episodes: {mean_reward:.2f}", flush=True)

        return True

    def _on_training_end(self) -> None:
        """Called when the training ends."""
        print(f"\nTraining completed after {self.episode_count} total episodes")
        print(f"Final average reward (last 1000 episodes): {np.mean(self.episode_rewards[-1000:]):.2f}")
        print(f"Best mean reward: {self.best_mean_reward:.2f}")
        # with open(self.outfile, "w") as f:
        #     # Process each episode
        #     for i in range(n_envs):
        #         for ep_idx, episode in enumerate(self.episode_lists[i][0]):
        #             # Skip if corrupted (doesn't end with a reward value)
        #             if not (len(episode) > 0 and isinstance(episode[-1], (float, np.float64, np.float32))):
        #                 print(f"Skipping corrupted episode {ep_idx+1}")
        #                 continue
                    
        #             # Process steps excluding the last step and the reward
        #             for step_idx in range(len(episode) - 2):
        #                 step = episode[step_idx]
                        
        #                 # Write rounded step data
        #                 if isinstance(step, list):
        #                     # Round each value in the list to 2 decimal places
        #                     rounded_vals = [round(val, 2) if isinstance(val, (float, np.float64, np.float32)) else val for val in step]
        #                     f.write(f"{str(rounded_vals)}\n")
        #                 else:
        #                     # Round individual values
        #                     rounded_val = round(step, 2) if isinstance(step, (float, np.float64, np.float32)) else step
        #                     f.write(f"{str(rounded_val)}\n")
                    
        #             # Write the rounded reward
        #             reward = episode[-1]
        #             rounded_reward = round(reward, 2) if isinstance(reward, (float, np.float64, np.float32)) else reward
        #             f.write(f"{rounded_reward}\n")
                    
        #             # Add blank line between episodes
        #             f.write("\n")

def run_hopper(infile, outfile, rounding):
    """
    Main function that runs the Hopper environment with specified parameters.
    
    Args:
        outfile: Path to output file
        k_neighbors: Number of neighbors for state matching
        p_state: Distance metric for state matching (1=Manhattan, 2=Euclidean)
        p_action: Distance metric for action comparison
        rounding: Action rounding strategy (0=none, 1=nearest 0.1, 2=nearest 0.5, 3=nearest integer)
    """
    global model, all_rewards, state_tree

    model = []
    all_rewards = []
    with open(infile, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:  # skip empty lines
                continue
            
            # Split into individual values
            values = line.split(',')
            
            # Extract state_dim state values
            state_values = [float(values[i]) for i in range(state_dim)]
            
            # Extract 8 action values
            action_values = [float(values[i]) for i in range(state_dim, total_dim)]
            
            # Extract reward value
            reward_value = float(values[total_dim])
            
            # Create entry with all values
            entry = state_values + action_values + [reward_value]
            model.append(entry)
            all_rewards.append(reward_value)

    # Compute the global maximum absolute value
    global_max = max(all_rewards)
    global_min = min(all_rewards)
    global_abs_max = max(global_max, abs(global_min))

    # Normalize each reward in the model so that the maximum absolute value becomes 1
    for entry in model:
        entry[total_dim] = entry[total_dim] / global_abs_max

    # Apply rounding based on parameter
    for entry in model:
        if rounding == 1:  # Round to nearest 0.1
            for i in range(state_dim, total_dim):  # Round all 8 action values
                entry[i] = round(entry[i] * 10) / 10
        elif rounding == 2:  # Round to nearest 0.5
            for i in range(state_dim, total_dim):
                entry[i] = round(entry[i] * 2) / 2
        elif rounding == 3:  # Round to nearest integer
            for i in range(state_dim, total_dim):
                entry[i] = round(entry[i])
        # For rounding == 0, no rounding is applied

    # Create a KD-tree of all states
    all_states = np.array([entry[0:state_dim] for entry in model])
    state_tree = KDTree(all_states)

    # data = [1.20, -0.10, -0.10, -0.00, -0.00, -0.10, -0.50, 0.50, 0.50, 0.50, -0.40, -0.10, -0.50, 0.40, 100]
    # state = [1.20, -0.10, -0.10, -0.00, -0.00, -0.10, -0.50, 0.50, 0.50, 0.50, -0.40]
    # act = [-0.10, -0.50, 0.40]
    # act = tuple(act)
    # estimate = find_closest_state_action_reversed(state, act, 2, 1, 2)
    # print(estimate)
    # exit()

    # Create the base environment
    env = make_vec_env(env_name, n_envs=n_envs, vec_env_cls=SubprocVecEnv, wrapper_class=InferredRewardWrapper)
    #TODO: change n_envs
    #TOOO: change first step rollout start

    normalize = True
    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Create the callback
    callback = LogAndSaveCallback(check_freq=10000, outfile=outfile)
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        device="cuda",
        #learning_starts=10000,
    )

    # Train
    model.learn(total_timesteps=n_timesteps, callback=callback)

    obs = env.reset()
    counter = 0
    while True:
        counter = counter + 1
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render("human")
        if (counter > 100000):
            break

if __name__=="__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run Hopper environment with KDTree reward inference')
    parser.add_argument('infile', type=str, help='Input file path')
    parser.add_argument('outfile', type=str, help='Output file path')
    parser.add_argument('env_name', type=str, help='Gym environment name (e.g., Ant-v5, HalfCheetah-v4)')
    parser.add_argument('n_envs', type=int)
    parser.add_argument('n_timesteps', type=int)
    parser.add_argument('state_dim', type=int)
    parser.add_argument('act_dim', type=int)
    parser.add_argument('k_neighbors', type=int, help='Number of neighbors for state matching')
    parser.add_argument('p_state', type=int, help='Distance metric for state (1=Manhattan, 2=Euclidean)')
    parser.add_argument('p_action', type=float, help='Distance metric for action comparison')
    parser.add_argument('rounding', type=int, choices=[0, 1, 2, 3], 
                        help='Action rounding: 0=none, 1=nearest 0.1, 2=nearest 0.5, 3=nearest integer')
    args = parser.parse_args()
    env_name = args.env_name
    n_envs = args.n_envs
    n_timesteps = args.n_timesteps
    state_dim = args.state_dim
    act_dim = args.act_dim
    total_dim = state_dim + act_dim
    k_neighbors = args.k_neighbors
    p_state = args.p_state
    p_action = args.p_action
    # Call the main function with parsed arguments
    run_hopper(
        infile=args.infile,
        outfile=args.outfile,
        rounding=args.rounding
    )