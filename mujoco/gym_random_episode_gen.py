import gymnasium as gym
import torch
import numpy as np
import os
import argparse
import multiprocessing
from functools import partial

def generate_episodes_file(file_index, env_name, num_timesteps, data_dir):
    """Generate a single file of random episodes up to num_timesteps"""
    print(f"\nGenerating file {file_index}/10")
    
    # Create a fresh environment for each file
    if env_name == "Ant-v4":
        print("Should probably use v5 for ant?")
    else:
        env = gym.make(env_name)
    
    device = "cpu"  # Always use CPU for multiprocessing
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    episodes = []
    
    # Track total timesteps generated
    total_timesteps = 0
    episode_count = 0
    
    while total_timesteps < num_timesteps:
        if episode_count % 100 == 0:
            print(f"  Process {file_index}: Episode {episode_count}, Timesteps {total_timesteps}/{num_timesteps}")
        
        episode_count += 1
        state, info = env.reset()
        state = torch.tensor([round(s.item(), 2) for s in state], dtype=torch.float32, device=device).unsqueeze(0)
        episodes.append([])
        sum_reward = 0
        episode_timesteps = 0
        flag = True

        while flag and total_timesteps < num_timesteps:
            action = env.action_space.sample()

            append_state = torch.tensor([round(s.item(), 2) for s in state.squeeze(0)], dtype=torch.float32, device=device)
            append_action = torch.tensor([round(a, 1) for a in action], dtype=torch.float32, device=device)
            
            episodes[-1].append(torch.cat((append_state, append_action), dim=0).squeeze().tolist())
            
            # Increment timestep counters
            total_timesteps += 1
            episode_timesteps += 1

            observation, reward, terminated, truncated, info = env.step(action)
            sum_reward += reward
            done = terminated or truncated

            if done:
                next_state = None
                episodes[-1].append(sum_reward)
                flag = False
            else:
                observation = [round(o, 1) for o in observation]
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            state = next_state
        
        # If we exceeded our timestep limit during this episode, discard the last episode
        if total_timesteps > num_timesteps:
            episodes.pop()  # Remove the last incomplete episode
            total_timesteps -= episode_timesteps  # Adjust the counter
            print(f"  Process {file_index}: Discarding last episode to stay under timestep limit")

    print(f"  Process {file_index}: Generated {len(episodes)} episodes with {total_timesteps} timesteps")

    # Generate output filename with index (now includes timesteps instead of episodes)
    filename = f"{env_name}_random_output_{file_index}.txt"
    output_path = f"{data_dir}/{filename}"

    # Write the collected data to the output file
    with open(output_path, 'w') as file:
        for episode in episodes:
            for step in episode:
                if isinstance(step, list):
                    formatted_step = [f"{value:.2f}" if isinstance(value, float) else str(value) for value in step]
                    file.write("[" + ", ".join(formatted_step) + "]\n")
                else:
                    file.write(f"{step:.2f}\n")
            file.write("\n")

    print(f"Data successfully written to {output_path}")
    
    # Close the environment
    env.close()
    
    return file_index, len(episodes), total_timesteps

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate random environment data')
    parser.add_argument('env_name', type=str, help='Environment name (e.g., "Hopper-v4")')
    parser.add_argument('num_timesteps', type=int, help='Number of timesteps to generate')
    parser.add_argument('processes', type=int, help='Number of parallel processes to use')
    args = parser.parse_args()

    # Create the environment from command-line argument
    env_name = args.env_name
    num_timesteps = args.num_timesteps
    num_processes = min(args.processes, 10)  # Cap at 10 processes since we generate 10 files

    print(f"Generating 10 files of approximately {num_timesteps} timesteps each for environment: {env_name}")
    print(f"Using {num_processes} parallel processes")

    # Create directory structure
    base_dir = f"/home/ihholmes/ARES/episode_data/{env_name}_episode_data"
    data_dir = f"{base_dir}/{env_name}_random_data"
    os.makedirs(data_dir, exist_ok=True)

    # Create a process pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Create a partial function with fixed arguments
        worker_func = partial(generate_episodes_file, 
                            env_name=env_name, 
                            num_timesteps=num_timesteps, 
                            data_dir=data_dir)
        
        # Map the function to the file indices (1-10)
        results = pool.map(worker_func, range(1, 11))
    
    # Summarize results
    total_episodes = sum(result[1] for result in results)
    total_steps = sum(result[2] for result in results)
    print("\nAll files generated successfully!")
    print(f"Generated {total_episodes} episodes with {total_steps} timesteps across 10 files")