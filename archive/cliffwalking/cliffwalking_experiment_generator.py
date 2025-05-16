import gymnasium as gym
import numpy as np
import torch
import argparse
import os
from q_agent import *

def run_experiment(experiment_type):
    """
    Run CliffWalking experiment based on specified type and collect statistics.
    
    Parameters:
    experiment_type (str): One of 'delayed', 'immediate', 'expert_inferred', or 'random_inferred'
    """
    optimal_states = [36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    correct_actions = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    env = gym.make("CliffWalking-v0", render_mode='ansi')
    n_actions = 4
    state, info = env.reset()
    n_observations = 48
    gamma = 0.9
    learning_rate = 100

    num_trials = 10

    episode_lengths = [50, 100, 200, 300, 400, 500, 1000]

    epsilon_starts = [0.9]

    # Configure experiment-specific parameters
    if experiment_type == "expert_inferred":
        reward_file_prefix = "cliffwalking_expert_rewards"
        use_model = True
        use_delayed = True
        header = "Expert Inferred Experiments"
    elif experiment_type == "random_inferred":
        reward_file_prefix = "cw_random_rewards"
        use_model = True
        use_delayed = True
        header = "Random Inferred Experiments"
    elif experiment_type == "immediate":
        reward_file_prefix = "blank"
        use_model = False
        use_delayed = False
        header = "Immediate Experiments"
    else:  # Default to delayed
        reward_file_prefix = "blank"
        use_model = False
        use_delayed = True
        header = "Delayed Experiments"

    # Generate list of reward files
    reward_files = [f"{reward_file_prefix}{i}.txt" for i in range(1, 11)]
    num_reward_files = len(reward_files)

    if use_model:
        for reward_file in reward_files:
            if not os.path.exists(reward_file):
                print(f"Error: Reward file {reward_file} not found!")
                print(f"Current working directory: {os.getcwd()}")
                print("Please check if the reward files exist in the correct location.")
                return

    # Create a dictionary to store results
    all_results = {
        "experiment_type": experiment_type,
        "results_by_config": {},
        "overall_success_rate": 0,
        "overall_success_count": 0,
        "overall_failure_count": 0,
        "total_trials": 0
    }
    
    print(f"\n{'='*50}")
    print(f"{header} - RESULTS")
    print(f"{'='*50}")

    # Run the experiments
    for epsilon_start in epsilon_starts:
        for episode_length in episode_lengths:
            config_key = f"epsilon={epsilon_start}, episodes={episode_length}"
            all_results["results_by_config"][config_key] = {
                "successes": [],
                "success_rate": 0,
                "success_count": 0,
                "failure_count": 0,
                "total_trials": 0,
                "results_by_file": {}
            }
            
            print(f"\nConfig: {config_key}")
            print("-" * 40)
            
            for i in range(num_trials):
                for j, reward_file in enumerate(reward_files):
                    print(f"Trial {i+1}, File: {os.path.basename(reward_file)}", end=": ")
                    
                    # Initialize file results if not present
                    file_name = os.path.basename(reward_file)
                    if file_name not in all_results["results_by_config"][config_key]["results_by_file"]:
                        all_results["results_by_config"][config_key]["results_by_file"][file_name] = {
                            "success_count": 0, 
                            "failure_count": 0,
                            "successes": []
                        }
                    
                    # Run training and get success/failure
                    if use_model:
                        _, _, success = train_q_agent(
                            env, episode_length, epsilon_start, 0.99, 0.99,
                            model=reward_file, sarsa=False, de=True, delayed=use_delayed,
                        )
                    else:
                        _, _, success = train_q_agent(
                            env, episode_length, epsilon_start, 0.99, 0.99,
                            sarsa=False, de=True, delayed=use_delayed,
                        )
                    
                    # Update result counters
                    if success:
                        print("SUCCESS")
                        all_results["results_by_config"][config_key]["successes"].append(1)
                        all_results["results_by_config"][config_key]["success_count"] += 1
                        all_results["results_by_config"][config_key]["results_by_file"][file_name]["success_count"] += 1
                        all_results["results_by_config"][config_key]["results_by_file"][file_name]["successes"].append(1)
                        all_results["overall_success_count"] += 1
                    else:
                        print("FAILURE")
                        all_results["results_by_config"][config_key]["successes"].append(0)
                        all_results["results_by_config"][config_key]["failure_count"] += 1
                        all_results["results_by_config"][config_key]["results_by_file"][file_name]["failure_count"] += 1
                        all_results["results_by_config"][config_key]["results_by_file"][file_name]["successes"].append(0)
                        all_results["overall_failure_count"] += 1
                    
                    all_results["results_by_config"][config_key]["total_trials"] += 1
                    all_results["total_trials"] += 1
            
            # Calculate success rate for this configuration
            config_trials = all_results["results_by_config"][config_key]["total_trials"]
            config_successes = all_results["results_by_config"][config_key]["success_count"]
            success_rate = config_successes / config_trials if config_trials > 0 else 0
            all_results["results_by_config"][config_key]["success_rate"] = success_rate
            
            # Calculate statistics
            successes = all_results["results_by_config"][config_key]["successes"]
            mean = np.mean(successes)
            std_dev = np.std(successes)
            
            # Print configuration summary
            print(f"\nConfiguration Summary ({config_key}):")
            print(f"Success rate: {success_rate:.2%} ({config_successes}/{config_trials})")
            print(f"Mean: {mean:.4f}, Std Dev: {std_dev:.4f}")
            
            # # File-specific results
            # print("\nFile-specific results:")
            # for file_name, file_results in all_results["results_by_config"][config_key]["results_by_file"].items():
            #     file_trials = file_results["success_count"] + file_results["failure_count"]
            #     file_success_rate = file_results["success_count"] / file_trials if file_trials > 0 else 0
            #     file_successes = file_results["successes"]
            #     file_mean = np.mean(file_successes) if file_successes else 0
            #     file_std = np.std(file_successes) if file_successes else 0
                
            #     print(f"  {file_name}: {file_success_rate:.2%} success rate, Mean: {file_mean:.4f}, Std: {file_std:.4f}")
    
    # Calculate overall success rate
    overall_success_rate = all_results["overall_success_count"] / all_results["total_trials"] if all_results["total_trials"] > 0 else 0
    all_results["overall_success_rate"] = overall_success_rate
    
    # Print overall summary
    print(f"\n{'='*50}")
    print(f"OVERALL SUMMARY: {header}")
    print(f"{'='*50}")
    print(f"Total trials: {all_results['total_trials']}")
    print(f"Successes: {all_results['overall_success_count']}")
    print(f"Failures: {all_results['overall_failure_count']}")
    print(f"Overall success rate: {overall_success_rate:.2%}")
    
    # Save results to file
    results_dir = "CliffWalking/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/{experiment_type}_results.txt"
    
    with open(results_file, 'w') as f:
        f.write(f"{header} Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Overall success rate: {overall_success_rate:.2%} ({all_results['overall_success_count']}/{all_results['total_trials']})\n")
        f.write(f"Failures: {all_results['overall_failure_count']}\n\n")
        
        for config_key, config_results in all_results["results_by_config"].items():
            f.write(f"Configuration: {config_key}\n")
            f.write(f"{'-'*40}\n")
            f.write(f"Success rate: {config_results['success_rate']:.2%} ({config_results['success_count']}/{config_results['total_trials']})\n")
            f.write(f"Mean: {np.mean(config_results['successes']):.4f}, Std Dev: {np.std(config_results['successes']):.4f}\n\n")
            
            # f.write("File-specific results:\n")
            # for file_name, file_results in config_results["results_by_file"].items():
            #     file_trials = file_results["success_count"] + file_results["failure_count"]
            #     file_success_rate = file_results["success_count"] / file_trials if file_trials > 0 else 0
            #     file_successes = file_results["successes"]
            #     file_mean = np.mean(file_successes) if file_successes else 0
            #     file_std = np.std(file_successes) if file_successes else 0
                
            #     f.write(f"  {file_name}: {file_success_rate:.2%} success rate, Mean: {file_mean:.4f}, Std: {file_std:.4f}\n")
            
            f.write("\n")
    
    print(f"\nResults saved to {results_file}")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CliffWalking experiments')
    parser.add_argument('type', type=str, default='delayed',
                        choices=['delayed', 'immediate', 'expert_inferred', 'random_inferred'],
                        help='Type of experiment to run')
    args = parser.parse_args()
    
    print(f"Running {args.type} experiment...")
    run_experiment(args.type)