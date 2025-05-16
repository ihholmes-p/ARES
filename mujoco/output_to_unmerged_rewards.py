import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import csv
import math
from ARES_transformer import *
import random

def get_nested_dict(dictionary, keys):
    """Access or create a path in a nested dictionary.
    
    Args:
        dictionary: The dictionary to traverse
        keys: List of keys forming the path
        create_missing: If True, create missing dictionary levels
        
    Returns:
        The nested dictionary at the specified path
    """
    current = dictionary
    for key in keys[:-1]:  # All but the last key
        if key not in current:
            current[key] = {}
        current = current[key]
    
    last_key = keys[-1]
    if last_key not in current:
        current[last_key] = {}
    
    return current[last_key]

def average_rewards(nested_dict):
    """Recursively average reward lists in a nested dictionary."""
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            average_rewards(value)
        else:  # This is a list of rewards
            nested_dict[key] = sum(value) / len(value)

def write_rewards(file, nested_dict, path=None):
    """Recursively write rewards to file."""
    if path is None:
        path = []
    
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            write_rewards(file, value, path + [key])
        else:  # This is the reward value
            state_str = ", ".join(str(p) for p in path)
            
            if isinstance(key, tuple):
                action_str = ", ".join(str(k) for k in key)
            else:
                action_str = str(key)
            
            file.write(f"{state_str}, {action_str}, {value:.2f}\n")

def save_dict_to_file(data, filename="test.txt"):
    """Save dictionary to a file in a readable format."""
    with open(filename, 'w') as f:
        for state_key, actions in data.items():
            f.write(f"State {state_key}:\n")
            for action, rewards in actions.items():
                f.write(f"  Action {action}: {rewards}\n")
            f.write("\n")

#####################################################################################################

def infer_rewards(infile, outfile, epochs, dimensions, lr, internal_embedding, state_dim, act_dim):

    # Calculate total dimension for state+action pairs
    total_dim = state_dim + act_dim

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    print(f"State dimension: {state_dim}, Action dimension: {act_dim}, Total: {total_dim}")

    episode_buffer = []
    reward_list = []

    length_threshold = 1050

    with open(infile, 'r') as file:
        episode = []
        for line in file:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                data = eval(line)
                episode.append(data)
            elif line:
                data = eval(line)
                episode.append(data)

                if (len(episode) < length_threshold): #TODO: probably not needed
                    episode_buffer.append(episode)
                else:
                    print("Very long episode removed. Double check that this is correct.")
                reward_list.append(data)
                #print(len(episode))
                episode = []

    buffer_for_transformer = []
    j = 0
    for episode in episode_buffer:
        buffer_for_transformer.append([])
        for step in episode:
            if isinstance(step, list):
                for data in step:
                    buffer_for_transformer[-1].append(data)
            else:
                buffer_for_transformer[-1].append(step)
        buffer_for_transformer[-1].append(j)
        j += 1



    episode_buffer = []
    print("episodes:", len(buffer_for_transformer))
    print("avg reward:", sum(reward_list) / len(reward_list))

    # Optional: shuffle the episodes for training
    random.shuffle(buffer_for_transformer)
    # original_count = len(buffer_for_transformer)
    # num_episodes_to_keep = max(1, int(original_count * 0.20))  # Ensure at least 1 episode
    # buffer_for_transformer = buffer_for_transformer[:num_episodes_to_keep]
    # print(f"Reduced dataset: keeping {num_episodes_to_keep} episodes (4% of original {original_count})")

    # Initialize model with dynamic dimensions
    model = GPT(total_dim, 1, dimensions, length_threshold, 1, 0.10, internal_embedding).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()

    torch.set_printoptions(threshold=10000, precision=5)
    torch.set_printoptions(sci_mode=False)
    print("train")
    batch_loss = 0

    for epoch in range(epochs):
        for episode in buffer_for_transformer:
            #take out index
            idx = episode[-1]
            episode = episode[0: len(episode) - 1]
            #set final token to final state instead of return
            episode_length = len(episode)
            episode = torch.tensor(episode).to(device)
            x = episode[0:episode_length - 1].unsqueeze(0)
            y = episode[-1].unsqueeze(0).float()
            out = model(x, total_dim)
            output = out[0]
            w = out[1].squeeze(0).squeeze(0)[-1]

            loss = criterion(output, y)
            batch_loss += loss
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        episode_length = int((len(episode) - 1) / total_dim)
        if (epoch % 20 == 0):
            print("------------EPISODE--------------")
            #print("x:", x)
            print("episode length:", episode_length)
            print("y:", y)
            # print("length:", episode_length)
            print("o:", output)
            # print("weights:", w)
            # print("weights size:", w.size())
            print("batch loss:", batch_loss.item())
            print(epoch)
            # if (epoch >= 2000):
            #     printlist = []
            #     for i in range(episode_length):
            #         out = model(x, total_dim, test=i)
            #         printlist.append(out[0].item())
            #     print("specific result:", printlist)
            print("---------------------------------", flush=True)

        random.shuffle(buffer_for_transformer)
        if (batch_loss < 1):
            break
        batch_loss = 0

    reward_values = {}
    a = []
    for episode in buffer_for_transformer:
        index = episode[-1]
        episode = episode[0: len(episode) - 1]
        episode_length = len(episode)
        episode = torch.tensor(episode).to(device)
        x = episode[0:episode_length - 1].unsqueeze(0)
        y = episode[-1].unsqueeze(0).float()

        episode_length = int((episode_length - 1) / total_dim)

        for i in range(episode_length):
            output = model(x, total_dim, test=i)[0]
            # Extract state and action using dynamic dimensions
            state = x.squeeze()[(total_dim*i):((total_dim*i)+state_dim)]
            action = x.squeeze()[(total_dim*i)+state_dim:(total_dim*i)+total_dim]
            state_values = [round(s.item(), 4) for s in state]
            action = tuple(round(a.item(), 2) for a in action)

            # Create nested dictionary path if needed
            nested_dict = get_nested_dict(reward_values, state_values)
            if action not in nested_dict:
                nested_dict[action] = []

            # Add output to the rewards list
            nested_dict[action].append(output.item())

    # Average the rewards and write to file
    average_rewards(reward_values)

    with open(outfile, 'w') as file:
        write_rewards(file, reward_values)

    print(f"Reward values written to {outfile}")

    return(reward_values)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python3 ARES_data_to_rewards.py <input_file> <output_file> [epochs] [dimensions] [learning_rate] [internal_embedding] [state_dim] [act_dim]")
        sys.exit(1)
        
    infile = sys.argv[1]
    outfile = sys.argv[2]
    epochs = int(sys.argv[3])
    dimensions = int(sys.argv[4])
    lr = float(sys.argv[5])
    internal_embedding = int(sys.argv[6])
    state_dim = int(sys.argv[7])
    act_dim = int(sys.argv[8])
    
    infer_rewards(infile=infile, outfile=outfile, epochs=epochs, dimensions=dimensions, 
                  lr=lr, internal_embedding=internal_embedding, 
                  state_dim=state_dim, act_dim=act_dim)