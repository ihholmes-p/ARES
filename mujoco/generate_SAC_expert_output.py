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
import argparse
from scipy.spatial import KDTree

# Suppress all future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class LogAndSaveCallback(BaseCallback):
    def __init__(self, check_freq, outfile, n_envs, verbose=1):
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
        self.n_envs = n_envs
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

        for i, reward in enumerate(self.training_env.get_original_reward()):
            self.current_episode_rewards[i] += reward

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
        #     for i in range(1):
        #     #for i in range(self.n_envs):
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

def run_environment(outfile, env_name, timesteps, n_envs):
    # Create the environment with the wrapper class
    env = make_vec_env(
        env_name, 
        n_envs=n_envs, 
        vec_env_cls=SubprocVecEnv,
    )

    normalize = True
    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create the callback
    callback = LogAndSaveCallback(check_freq=10000, outfile=outfile, n_envs=n_envs)

    # Create and train the model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        device="cuda",
    )

    # Train
    model.learn(total_timesteps=timesteps, callback=callback)

if __name__=="__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', type=str, help='Output file path for results')
    parser.add_argument('env_name', type=str, help='Environment name (e.g., "Hopper-v4")')
    parser.add_argument('timesteps', type=int, help='Number of timesteps to train for')
    parser.add_argument('n_envs', type=int, help='Number of parallel environments to use')
    
    args = parser.parse_args()
    
    print(f"Running environment {args.env_name} for {args.timesteps} timesteps with {args.n_envs} parallel environments")
    print(f"Output will be saved to {args.outfile}")
    
    # Call the main function with parsed arguments
    run_environment(outfile=args.outfile, env_name=args.env_name, timesteps=args.timesteps, n_envs=args.n_envs)