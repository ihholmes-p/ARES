import numpy as np
import argparse
import multiprocessing
import os
from functools import partial
import time
import uuid

def merge_similar_entries(model, max_state_distance, max_action_distance, p_state, p_action, max_dimension_diff, state_dim, act_dim):
    """
    Merge entries in the model using either per-dimension checks or distance metrics.
    
    Args:
        model: List of entries where each entry is [s1, s2, ..., sN, a1, a2, ..., aM, reward]
        max_state_distance: Maximum distance between state components (used if max_dimension_diff <= 0)
        max_action_distance: Maximum distance between action components (used if max_dimension_diff <= 0)
        p_state: Distance metric parameter for state (p=1 for Manhattan, p=2 for Euclidean)
        p_action: Distance metric parameter for action (p=1 for Manhattan, p=2 for Euclidean)
        max_dimension_diff: Maximum allowed difference per dimension. If > 0, use per-dimension checking
        state_dim: Number of state dimensions
        act_dim: Number of action dimensions
        
    Returns:
        A new list with merged entries
    """
    # Calculate total dimension (state + action, excluding reward)
    total_dim = state_dim + act_dim
    
    # Create a copy of the model to avoid modifying the original
    remaining_entries = model.copy()
    merged_model = []
    max_cluster_size = 0
    
    # Process entries until none remain
    while remaining_entries:
        if len(remaining_entries) % 1000 == 0:
            print(f"Entries left: {len(remaining_entries)}")
            
        # Take the first unprocessed entry as reference
        reference = remaining_entries.pop(0)
        
        # Separate state and action components of reference
        reference_state = np.array(reference[:state_dim])  # State components
        reference_action = np.array(reference[state_dim:total_dim])  # Action components
        
        # Find all entries that meet similarity criteria
        similar_entries = [reference]
        i = 0
        while i < len(remaining_entries):
            entry = remaining_entries[i]
            
            # Separate state and action components
            entry_state = np.array(entry[:state_dim])
            entry_action = np.array(entry[state_dim:total_dim])
            
            is_similar = False
            
            if max_dimension_diff > 0:
                # Check if every dimension is within the threshold
                state_diffs = np.abs(reference_state - entry_state)
                action_diffs = np.abs(reference_action - entry_action)
                
                # Entry is similar if maximum difference in any dimension is below threshold
                max_state_diff = np.max(state_diffs)
                max_action_diff = np.max(action_diffs)
                is_similar = (max_state_diff <= max_dimension_diff and 
                              max_action_diff <= max_dimension_diff)
            else:
                # Use the original p-norm distance approach
                if p_state == float('inf'):
                    state_dist = max(abs(reference_state - entry_state))
                else:
                    state_dist = np.sum(np.abs(reference_state - entry_state) ** p_state) ** (1/p_state)
                
                if p_action == float('inf'):
                    action_dist = max(abs(reference_action - entry_action))
                else:
                    action_dist = np.sum(np.abs(reference_action - entry_action) ** p_action) ** (1/p_action)
                
                # Entry is merged if both distances are below thresholds
                is_similar = (state_dist <= max_state_distance and 
                              action_dist <= max_action_distance)
                              
            if is_similar:
                similar_entries.append(entry)
                remaining_entries.pop(i)
            else:
                # Move to next entry
                i += 1
        
        # Update max cluster size
        max_cluster_size = max(max_cluster_size, len(similar_entries))
        
        # Compute average entry
        merged_entry = []
        for j in range(len(reference)):
            avg_value = round(sum(entry[j] for entry in similar_entries) / len(similar_entries), 2)
            merged_entry.append(avg_value)
        
        # Add merged entry to result
        merged_model.append(merged_entry)
    
    print(f"Merged {len(model)} entries into {len(merged_model)} entries (largest cluster: {max_cluster_size} entries)")
    return merged_model

def process_chunk(chunk_data, chunk_id, output_dir, max_state_distance, max_action_distance, 
                 p_state, p_action, global_abs_max, run_id, max_dimension_diff, state_dim, act_dim):
    """Process a single chunk of the data"""
    print(f"Processing chunk {chunk_id} with {len(chunk_data)} entries...")
    
    # Merge entries in this chunk
    merged_chunk = merge_similar_entries(
        chunk_data, max_state_distance, max_action_distance, p_state, p_action, max_dimension_diff,
        state_dim, act_dim
    )
    
    # Write the chunk result to a temporary file
    chunk_file = os.path.join(output_dir, f"chunk_{run_id}_{chunk_id}.txt")
    with open(chunk_file, "w") as f:
        for entry in merged_chunk:
            formatted_entry = [f"{value:.2f}" for value in entry]
            f.write(', '.join(formatted_entry) + '\n')
    
    print(f"Chunk {chunk_id} completed: {len(chunk_data)} entries merged to {len(merged_chunk)}")
    return chunk_file

def main():
    parser = argparse.ArgumentParser(description='Parallel merge similar entries in a rewards file')
    parser.add_argument('input', type=str,
                        help='Input rewards file')
    parser.add_argument('output', type=str,
                        help='Output merged rewards file')
    parser.add_argument('max_state_distance', type=float,
                        help='Maximum distance between state components (used if max_dimension_diff <= 0)')
    parser.add_argument('max_action_distance', type=float, 
                        help='Maximum distance between action components (used if max_dimension_diff <= 0)')
    parser.add_argument('max_dimension_diff', type=float, 
                        help='Maximum allowed difference per dimension. If > 0, use per-dimension comparison')
    parser.add_argument('p_state', type=float, 
                        help='Distance metric parameter for state (1=Manhattan, 2=Euclidean)')
    parser.add_argument('p_action', type=float, 
                        help='Distance metric parameter for action (1=Manhattan, 2=Euclidean)')
    parser.add_argument('chunks', type=int,
                        help='Number of chunks to split the data into')
    parser.add_argument('processes', type=int,
                        help='Number of parallel processes')
    parser.add_argument('state_dim', type=int,
                        help='Number of state dimensions')
    parser.add_argument('act_dim', type=int,
                        help='Number of action dimensions')
    
    args = parser.parse_args()
    
    # Calculate total dimension
    total_dim = args.state_dim + args.act_dim
    
    # Create a temporary directory for chunk outputs
    run_id = str(uuid.uuid4())[:8]  # Generate a short unique ID
    temp_dir = f"temp_merge_chunks_{run_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Load model data
    print(f"Loading data from {args.input}...")
    model = []
    
    with open(args.input, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:  # skip empty lines
                continue
            try:
                parts = line.split(',')
                expected_parts = total_dim + 1  # state + action + reward
                if len(parts) != expected_parts:
                    print(f"Warning: Line {line_num} contains {len(parts)} values instead of {expected_parts}. Skipping.")
                    continue
                
                values = [float(part.strip()) for part in parts]
                model.append(values)
            except ValueError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    print(f"Loaded {len(model)} entries in {time.time() - start_time:.2f} seconds")
    
    # Calculate global reward range for normalization
    all_rewards = [entry[total_dim] for entry in model]
    global_max = max(all_rewards)
    global_min = min(all_rewards)
    global_abs_max = max(global_max, abs(global_min))
    print(f"Reward range: {global_min:.2f} to {global_max:.2f}")
    
    # Split the data into chunks
    chunk_size = len(model) // args.chunks
    chunks = []
    for i in range(args.chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < args.chunks - 1 else len(model)
        chunks.append(model[start_idx:end_idx])
    
    print(f"Split data into {args.chunks} chunks of approximately {chunk_size} entries each")
    
    try:
        # Process chunks in parallel
        pool = multiprocessing.Pool(processes=args.processes)
        process_func = partial(
            process_chunk,
            output_dir=temp_dir,
            max_state_distance=args.max_state_distance,
            max_action_distance=args.max_action_distance,
            p_state=args.p_state,
            p_action=args.p_action,
            global_abs_max=global_abs_max,
            run_id=run_id,
            max_dimension_diff=args.max_dimension_diff,
            state_dim=args.state_dim,
            act_dim=args.act_dim
        )
        
        print(f"Starting parallel processing with {args.processes or multiprocessing.cpu_count()} processes...")
        chunk_files = pool.starmap(process_func, [(chunk, i) for i, chunk in enumerate(chunks)])
        pool.close()
        pool.join()
        
        # Combine results
        print("Combining results from all chunks...")
        with open(args.output, 'w') as outfile:
            for chunk_file in chunk_files:
                with open(chunk_file, 'r') as infile:
                    outfile.write(infile.read())
    
    finally:
        # Clean up temporary files
        print("Cleaning up temporary files...")
        try:
            for chunk_file in chunk_files:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Warning: Error during cleanup - {e}")
    
    end_time = time.time()
    print(f"Done! Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()