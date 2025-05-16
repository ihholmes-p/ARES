#!/usr/bin/env python3
# filepath: /home/ihholmes/ARES/cliffwalking_random_episode_gen.py
import gymnasium as gym
import os
import argparse

def generate_random_episodes(num_episodes, output_file):
    """Generate random episodes from the CliffWalking environment"""
    # Create environment
    env = gym.make('CliffWalking-v0')
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Stats tracking
    total_episodes = 0
    valid_episodes = 0
    skipped_episodes = 0
    episodes_data = []
    
    # Generate episodes until we have the requested number
    while valid_episodes < num_episodes:
        if total_episodes % 100 == 0:
            print(f"Generated {valid_episodes}/{num_episodes} valid episodes (skipped {skipped_episodes})")
        
        total_episodes += 1
        state, info = env.reset()
        current_episode = []
        sum_reward = 0
        done = False
        
        # Run one episode
        while not done:
            action = env.action_space.sample()  # Random action (0: up, 1: right, 2: down, 3: left)
            
            # Store state-action pair
            current_episode.append([state, action])
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Custom reward: if next state is the goal (47), set reward to 100
            if observation == 47:
                reward = 100
                
            sum_reward += reward
            done = terminated or truncated
            state = observation
        
        # Add the total reward as the last item in the episode
        current_episode.append(sum_reward)
        
        # Only save episodes with reward >= -1000
        if sum_reward >= -2000:
            episodes_data.append(current_episode)
            valid_episodes += 1
        else:
            skipped_episodes += 1
    
    # Write episodes to file
    with open(output_file, 'w') as file:
        for episode in episodes_data:
            for i, step in enumerate(episode):
                if i < len(episode) - 1:
                    # State-action pair
                    file.write(f"[{step[0]}, {step[1]}]\n")
                else:
                    # Episode reward (last item)
                    file.write(f"{step:.1f}\n")
            file.write("\n")  # Empty line between episodes
    
    print(f"\nGeneration complete!")
    print(f"Generated {valid_episodes} valid episodes")
    print(f"Skipped {skipped_episodes} episodes with rewards > -1000")
    print(f"Data saved to {output_file}")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate random CliffWalking episodes')
    parser.add_argument('num_episodes', type=int, help='Number of episodes to generate per file')
    args = parser.parse_args()

    # Create directory structure
    base_dir = "/home/ihholmes/ARES/episode_data/CliffWalking_episode_data"
    data_dir = f"{base_dir}/CliffWalking_random_data"
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Generating 10 files with {args.num_episodes} random CliffWalking episodes each")
    print("Custom rewards: +100 for reaching the goal, only saving episodes with reward >= -2000")
    
    # Generate 10 files
    for i in range(1, 11):
        output_file = f"{data_dir}/CliffWalking_random_output_{i}.txt"
        print(f"\n--- Generating file {i}/10: {output_file} ---")
        generate_random_episodes(args.num_episodes, output_file)
    
    print("\nAll files generated successfully!")