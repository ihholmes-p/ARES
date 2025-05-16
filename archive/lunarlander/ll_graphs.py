import re
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pandas as pd

# Set plot style
plt.style.use('ggplot')

# Updated directory containing the files
data_dir = "/home/ihholmes/rat/lunarlander_experiments/solved5_experiment_results/"

# Define epsilon values and method types
epsilon_values = [0.9]
methods = ["delayed", "random", "immediate", "expert"]
method_labels = {"delayed": "Delayed Rewards (Baseline)", "random": "Random-Inferred Shaped Rewards", "immediate": "Immediate Rewards (Baseline)", "expert": "Expert-Inferred Shaped Rewards"}

# Update colors dictionary to use capitalized method names that match the DataFrame
colors = {"Delayed Rewards (Baseline)": "blue", "Random-Inferred Shaped Rewards": "green", "Immediate Rewards (Baseline)": "red", "Expert-Inferred Shaped Rewards": "purple"}

# Define episode lengths
episode_lengths = [100, 200, 300, 400, 500, 1000, 2000]
x_values = range(len(episode_lengths))

# Add standard deviation data from your analysis
std_dev_values = {
    "Delayed Rewards (Baseline)": 12.36/25,
    "Random-Inferred Shaped Rewards": 15.41/25,
    "Immediate Rewards (Baseline)": 21.46/25,
    "Expert-Inferred Shaped Rewards": 18.86/25
}

def read_data_file(filepath):
    """Read data from a file, returning list of data rows after the header"""
    data_rows = []
    try:
        with open(filepath, 'r') as f:
            # Skip header
            next(f)
            # Read data rows
            for line in f:
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    # Parse the line into a list
                    row = eval(line)
                    data_rows.append(row)
        return data_rows
    except Exception as e:
        print(f"Warning: Issue reading {filepath}: {e}")
        return []

# Read all data files
all_data = {}
for method in methods:
    all_data[method] = {}
    for epsilon in epsilon_values:
        filepath = f"{data_dir}ll_{method}_{epsilon}_5.txt"
        if os.path.exists(filepath):
            all_data[method][epsilon] = read_data_file(filepath)
        else:
            print(f"File not found: {filepath}")
            all_data[method][epsilon] = []

# Convert data to pandas DataFrame for seaborn
df_list = []
for method in methods:
    for epsilon in epsilon_values:
        if epsilon in all_data[method] and all_data[method][epsilon]:
            for i, row in enumerate(all_data[method][epsilon]):
                df_list.append({
                    'Method': method_labels[method],
                    'Epsilon': epsilon,
                    'Episode Length': episode_lengths[i],
                    'Success Rate': row[2],
                    'StdDev': std_dev_values[method_labels[method]] / 50.0  # Convert to proportion scale
                })

# Create DataFrame
plot_data = pd.DataFrame(df_list)

# Create graphs for each epsilon value using matplotlib with shaded error regions
for epsilon in epsilon_values:
    # Change figure height from 6 to 5 to shrink vertically
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filter data for this epsilon value
    epsilon_data = plot_data[plot_data['Epsilon'] == epsilon]
    
    # Create a categorical mapping for x-axis positions
    # This ensures equal spacing between points visually
    episode_positions = {length: i for i, length in enumerate(episode_lengths)}
    
    # Plot for each method separately to handle error regions properly
    for method in method_labels.values():
        method_data = epsilon_data[epsilon_data['Method'] == method]
        
        if not method_data.empty:
            # Sort data by episode length to ensure proper line plot
            method_data = method_data.sort_values('Episode Length')
            
            # Get x-positions based on the categorical mapping
            x_positions = [episode_positions[length] for length in method_data['Episode Length']]
            
            # Create the main line using positions instead of actual values
            ax.plot(
                x_positions,
                method_data['Success Rate'],
                label=method,
                color=colors[method],
                marker='o',
                linewidth=2.5,
                markersize=8,
                alpha=0.8
            )
            
            # Add shaded region for 1 standard deviation
            ax.fill_between(
                x_positions,
                method_data['Success Rate'] - method_data['StdDev'],
                method_data['Success Rate'] + method_data['StdDev'],
                color=colors[method],
                alpha=0.2
            )
    
    # Title with larger font
    ax.set_title('LunarLander', fontsize=20)
    
    # Axis labels with larger font
    ax.set_xlabel('Episode Length', fontsize=18)
    ax.set_ylabel('Prop. of trials that achieve reward >=200', fontsize=18)
    
    # Set specific y-axis ticks and make them larger
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=16)
    
    # Set x-axis ticks using the position mapping
    ax.set_xticks(range(len(episode_lengths)))
    ax.set_xticklabels([str(length) for length in episode_lengths], fontsize=16)
    
    # Set axis limits - extend y-axis slightly below 0 like in CartPole
    ax.set_ylim(-0.05, 1.0)  # Changed from (0, 1.0) to (-0.05, 1.0)
    ax.set_xlim(-0.1, len(episode_lengths) - 0.9)  # Add a bit of padding
    
    plt.grid(True, alpha=0.3)
    
    # Make legend text larger
    plt.legend(title='Method', fontsize=14, title_fontsize=16, loc='best')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{data_dir}/lunarlander_epsilon_{epsilon}_plot_with_shaded_errors.png", dpi=300)
    plt.show()

print("Graphs with shaded error regions created successfully!")