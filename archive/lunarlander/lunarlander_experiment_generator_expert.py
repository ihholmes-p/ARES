import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ll_gen import *

def exp_gen(epsilon_starts, episode_lengths, num_to_solve, outfile):

    num_trials = 5

    reward_files = [
        "ll_expert_solve5_rewards1.txt",
        "ll_expert_solve5_rewards2.txt",
        "ll_expert_solve5_rewards3.txt",
        "ll_expert_solve5_rewards4.txt",
        "ll_expert_solve5_rewards5.txt",
        "ll_expert_solve5_rewards6.txt",
        "ll_expert_solve5_rewards7.txt",
        "ll_expert_solve5_rewards8.txt",
        "ll_expert_solve5_rewards9.txt",
        "ll_expert_solve5_rewards10.txt",
    ]

    num_reward_files = len(reward_files)

    inferred_experiments = []
    all_avg_scores = []
    for epsilon_start in epsilon_starts:
        for episode_length in episode_lengths:
            inferred_optimals = 0
            for i in range(num_trials):
                for reward_file in reward_files:
                    inferred_trial, avg_scores = run_ll(infile=reward_file, reward_type="inferred", epsilon_start=epsilon_start, num_episodes=episode_length, num_to_solve=num_to_solve)
                    if (inferred_trial == True):
                        inferred_optimals += 1
                    # if (episode_length == 1000):
                    #     _, avg_scores = run_ll(infile=reward_file, reward_type="inferred", epsilon_start=epsilon_start, num_episodes=1000, num_to_solve=2000)
                    #     all_avg_scores.append(avg_scores)
            inferred_experiments.append([epsilon_start, episode_length, inferred_optimals/(num_reward_files * num_trials)])

    with open(outfile, 'w') as file:
        file.write("Delayed Inferred Experiments:\n")
        for experiment in inferred_experiments:
            file.write(f"{experiment}\n")
        # file.write("Average Scores:\n")
        # scores_to_write = []
        # print(all_avg_scores)
        # print(len(all_avg_scores[0]))
        # for i in range(0, 1000, 10):
        #     scores_at_timestep = [score[int(i/10)] for score in all_avg_scores]
        #     avg_score = np.average(scores_at_timestep)
        #     file.write(f"{avg_score}\n")
        #     print(f"Timestep {i}: {avg_score:.4f}")

if __name__ == "__main__":

    import sys

    epsilon_start = float(sys.argv[1])
    episode_length = int(sys.argv[2])
    num_to_solve = int(sys.argv[3])
    outfile = sys.argv[4]
    exp_gen([epsilon_start], [episode_length], num_to_solve, outfile)

