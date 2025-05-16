import gymnasium as gym
import numpy as np
import torch

#de is for decreasing epsilon (set to False for generating random data with epsilon = 1)
#et is for eligiblity traces, which don't seem to improve delayed reward performance
def train_q_agent(env, episodes, epsilon, gamma, learning_rate, sarsa=False, model=None, 
                  device=None, slippery=False, de=True, et=False, delayed=False):

    rewards_dict = {}
    #https://amreis.github.io/ml/reinf-learn/2017/11/02/reinforcement-learning-eligibility-traces.html
    #https://rl-book.com/learn/value_methods/eligibility_traces/#:~:text=Eligibility%20traces%20implement%20n%2DStep,past%20and%20update%20them%20accordingly.
    elig = [[0] * 4 for _ in range(48)]
    elig_lambda = 0.5
    alpha = 0.5

    if type(model) is str:
        # First pass: read all rewards
        all_rewards = []
        with open(model, 'r') as file:
            for line in file:
                state, action, reward = line.strip().split(', ')
                state, action, reward = int(float(state)), int(float(action)), float(reward)
                if state not in rewards_dict:
                    rewards_dict[state] = [0, 0, 0, 0]
                rewards_dict[state][action] = reward
                all_rewards.append(reward)
                
        # Compute the global maximum absolute value
        if all_rewards:  # Make sure we have rewards
            global_max = max(all_rewards)
            global_min = min(all_rewards)
            global_abs_max = max(abs(global_max), abs(global_min))
            
            # If global_abs_max is 0 or very small, avoid division by zero
            if global_abs_max > 0:
                # Normalize rewards
                for state in rewards_dict:
                    for action in range(4):
                        if (rewards_dict[state][action] == 0): #in case some state-action pair was not in the data
                            rewards_dict[state][action] = np.average(all_rewards)
                        rewards_dict[state][action] = rewards_dict[state][action] / global_abs_max
                
                print(f"Normalized rewards using global abs max: {global_abs_max}")
            else:
                print("Warning: Maximum reward magnitude is too small for normalization")

    episode_buffer = []
    q_values = {s: [0, 0, 0, 0] for s in range(48)}

    def egreedy_policy(q_values, state, epsilon):  
        if np.random.random() < epsilon:
            return np.random.choice(4), "random"
        else:
            return np.argmax(q_values[state]), "nonrandom"

    agent_batch_reward = 0
    # Iterate over 500 episodes
    #print("--Q-AGENT TRAIN LOOP INITIALIZED--")
    for i in range(episodes):
        if (de == True):
            epsilon = epsilon * 0.99
        state = env.reset()[0]
        #state = RLstate(state)
        if not state in q_values:
            q_values[state] = [0, 0, 0, 0]
        done = False

        if i >= 0:
            episode_buffer.append([])

        # delayed_reward = 0
        # episode_length = 0

        trueReward = 0
        episode_sum_reward = 0
        elig_counter = 0
        while not done:      
            action = egreedy_policy(q_values, state, epsilon)
            marker = action[1]
            action = action[0]
            if (et==True and marker == "nonrandom"):
                elig = [[0] * 4 for _ in range(48)]
            if (slippery == True):
                r = action
                if (np.random.random() < 0.2):
                    r = np.random.choice(4)
                    while(r == action):
                        r = np.random.choice(4)
                next_state, reward, done, _, _ = env.step(r)
            else:
                next_state, reward, done, _, _ = env.step(action)

            if not next_state in q_values:
                q_values[next_state] = [0, 0, 0, 0]

            if (next_state == 47): #note that block is undone in the next block for delayed rewards
                reward = 100
            trueReward = reward
            episode_sum_reward += reward

            if (delayed == True):
                reward = 0
                if (done == True):
                    reward = episode_sum_reward


            if not model is None:
                if type(model) is str:
                    trueReward = reward
                    reward = rewards_dict[state][action]

            if i >= 0: #TODO: Threshold
                episode_buffer[-1].append([state, action, reward, marker])
            # if (i > 1000 and i % 1000 == 0):
            #     print(episode_buffer[-1])
            td_target = reward + gamma * np.max(q_values[next_state])

            if (sarsa == True):
                best = np.max(q_values[next_state])
                summation  = gamma * 0.9 * best
                for q_value in q_values[next_state]:
                    if not (q_value == best):
                        summation += gamma * (epsilon / 3) * q_value
                td_target = reward + summation

            td_error = td_target - (q_values[state])[action]

            if (et == True and elig_counter < 500):
                elig_counter += 1
                elig[state][action] = 1
                for s in range(len(elig)):
                    for a in range(len(elig[s])):
                        (q_values[s])[a] += alpha * td_error * elig[s][a]
                        elig[s][a] = elig[s][a] * gamma * elig_lambda

            (q_values[state])[action] += learning_rate * td_error
            state = next_state
            agent_batch_reward += trueReward

            # if (agent_batch_reward < -100000):
            #     print("too long")
            #     return episode_buffer, q_values, False

        if (et==True):
            elig = [[0] * 4 for _ in range(48)]

        total = 0
        i += 1

        if (agent_batch_reward > 87.5):
            return episode_buffer, q_values, True

        if ( (not type(model) is str) and agent_batch_reward < -100000):
            #Note: this is intentionally only done for the baselines,
            #because if the actual algorithm fails to converge, we want to crash
            #(in this case loop forever) instead of merely returning failure
            #print("Baseline fails to converge, return failure on this trial")
            return episode_buffer, q_values, False

        batch_size = 1
        if (i >= batch_size):
            if (i % batch_size == 0):
                #print(i, agent_batch_reward / batch_size)
                total = agent_batch_reward / batch_size
                agent_batch_reward = 0
                #print(epsilon)
        i -= 1

    #print("--Q-AGENT TRAIN LOOP COMPLETE--")
    return episode_buffer, q_values, False
    #return False

if __name__ == "__main__":

    # env = gym.make("CliffWalking-v0", render_mode='ansi')
    # n_actions = 4
    # state, info = env.reset()
    # n_observations = 48
    # gamma = 0.9
    # learning_rate = 100

    # num_trials = 100

    # episode_lengths = [50, 100, 200, 300, 400, 500, 1000]

    # epsilon_starts = [0.9, 0.5, 0.1]

    # reward_files = [
    #     "cliffwalking_expert_rewards1.txt",
    #     "cliffwalking_expert_rewards2.txt",
    #     "cliffwalking_expert_rewards3.txt",
    #     "cliffwalking_expert_rewards4.txt",
    #     "cliffwalking_expert_rewards5.txt",
    #     "cliffwalking_expert_rewards6.txt",
    #     "cliffwalking_expert_rewards7.txt",
    #     "cliffwalking_expert_rewards8.txt",
    #     "cliffwalking_expert_rewards9.txt",
    #     "cliffwalking_expert_rewards10.txt",
    # ]

    # num_reward_files = len(reward_files)

    # threshold = 87.5

    # expert_inferred_experiments = []
    # for epsilon_start in epsilon_starts:
    #     for episode_length in episode_lengths:
    #         expert_optimals = 0
    #         for i in range(num_trials):
    #             for reward_file in reward_files:
    #                 expert_trial = train_q_agent(env, episode_length, 0.5, 0.99, 0.99, de=True)
    #                 if (expert_trial == True):
    #                     expert_optimals += 1
    #         expert_inferred_experiments.append([epsilon_start, episode_length, expert_optimals/(num_reward_files * num_trials)])

    # with open("cliffwalking_experiment_results_expert_inferred.txt", 'w') as file:
    #     file.write("Expert Inferred Experiments:\n")
    #     for experiment in expert_inferred_experiments:
    #         file.write(f"{experiment}\n")