import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from continuous_msetransformer import *
from cartpole_msetransformer_test import *
from cartpole_generator import *

if __name__ == "__main__":

    inferred_trial = run_cartpole(infile="./cartpole_experiments/cartpole_random_rewards1.txt", reward_type="inferred", epsilon_start=0.9, num_episodes=500)
    print(inferred_trial[0])
    exit()

    num_trials = 10

    episode_lengths = [50, 100, 150, 200, 300, 400, 500, 1000]

    epsilon_starts = [0.9, 0.5, 0.1]

    reward_files = [
        "cartpole_expert_rewards1.txt",
        "cartpole_expert_rewards2.txt",
        "cartpole_expert_rewards3.txt",
        "cartpole_expert_rewards4.txt",
        "cartpole_expert_rewards5.txt",
        "cartpole_expert_rewards6.txt",
        "cartpole_expert_rewards7.txt",
        "cartpole_expert_rewards8.txt",
        "cartpole_expert_rewards9.txt",
        "cartpole_expert_rewards10.txt",
    ]

    num_reward_files = len(reward_files)

    #run_cartpole(infile="cartpole_expert_rewards8.txt", outfile="test.txt", reward_type="inferred", epsilon_start=0.1, num_episodes=200)

    #infer_rewards(infile="cartpole_expert_output1.txt", outfile="cartpole_expert_rewards1.txt", epochs=1001)

    inferred_experiments = []
    immediate_experiments = []
    delayed_experiments = []
    for epsilon_start in epsilon_starts:
        for episode_length in episode_lengths:
            inferred_optimals = 0
            immediate_optimals = 0
            delayed_optimals = 0
            for i in range(num_trials):
                for reward_file in reward_files:
                    inferred_trial = run_cartpole(infile=reward_file, reward_type="inferred", epsilon_start=epsilon_start, num_episodes=episode_length)
                    immediate_trial = run_cartpole(infile=reward_file, reward_type="immediate", epsilon_start=epsilon_start, num_episodes=episode_length)
                    delayed_trial = run_cartpole(infile=reward_file, reward_type="delayed", epsilon_start=epsilon_start, num_episodes=episode_length)
                    if (inferred_trial[1] == True):
                        inferred_optimals += 1
                    if (immediate_trial[1] == True):
                        immediate_optimals += 1
                    if (delayed_trial[1] == True):
                        delayed_optimals += 1
            inferred_experiments.append([epsilon_start, episode_length, inferred_optimals/(num_reward_files * num_trials)])
            immediate_experiments.append([epsilon_start, episode_length, immediate_optimals/(num_reward_files * num_trials)])
            delayed_experiments.append([epsilon_start, episode_length, delayed_optimals/(num_reward_files * num_trials)])

    with open("cartpole_experiment_results_1.txt", 'w') as file:
        file.write("Inferred Experiments:\n")
        for experiment in inferred_experiments:
            file.write(f"{experiment}\n")
        file.write("\nImmediate Experiments:\n")
        for experiment in immediate_experiments:
            file.write(f"{experiment}\n")
        file.write("\nDelayed Experiments:\n")
        for experiment in delayed_experiments:
            file.write(f"{experiment}\n")

    exit()

    import matplotlib.pyplot as plt

    exp_09 = [(arr[1], arr[2]) for arr in inferred_experiments if arr[0] == 0.9]
    exp_01 = [(arr[1], arr[2]) for arr in inferred_experiments if arr[0] == 0.1]

    x_09 = [x[0] for x in exp_09]
    y_09 = [x[1] for x in exp_09]
    x_01 = [x[0] for x in exp_01]
    y_01 = [x[1] for x in exp_01]

    plt.plot(x_09, y_09, marker='o', label='epsilon=0.9')
    plt.plot(x_01, y_01, marker='o', label='epsilon=0.1')
    plt.ylim(0.0, 1.0)
    plt.xlabel('Episode Length')
    plt.ylabel('Inferred Optimal Rate')
    plt.legend()
    plt.show()