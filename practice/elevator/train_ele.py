# --- train.py ---
import gymnasium as gym
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np
import random 
import time 
from elevator import ElevatorEnv 

register(
     id="Elevator-v0",         
     entry_point="elevator:ElevatorEnv", # filename, classname
     max_episode_steps=50
)

n_passengers = 10
reward_pickup = 10
reward_deliver = 20
max_reward = n_passengers * (reward_pickup + reward_deliver)

print("Registered!")
env = gym.make("Elevator-v0", n_floors = 5, passengers = n_passengers, reward_pickup = reward_pickup, reward_deliver = reward_deliver)
print("env created")

# ...  Q-learning  ...
state_space_size = env.observation_space.n   # 40960
action_space_size = env.action_space.n     # 3

# initialize Q table
q_table = np.zeros((state_space_size, action_space_size))

print(f"Q-table shape: {q_table.shape}")

# hyper parameter
total_episodes = 100000
sessions = total_episodes / 10

learning_rate = 0.1  

discount_factor = 0.7

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.1

epsilon_decay_rate = (max_epsilon - min_epsilon) / total_episodes

print("--- start training ---")
start_time = time.time()
total_reward = 0

reward_history = []

# iterate episodes
for episode in range(total_episodes):
    
    # reset
    state, info = env.reset()
    terminated = False
    truncated = False
    
    # run actions
    while not terminated and not truncated:
        
        # Epsilon-Greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # (0, 1, 2)
        else:
            action = np.argmax(q_table[state, :])

        # --- step ---
        new_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Q-Learning formula, Q(s, a) = Q(s, a) + alpha * (R + gamma * max_a'(Q(s', a')) - Q(s, a))

        # old Q 
        old_value = q_table[state, action]
        
        # future best Q
        future_best_value = np.max(q_table[new_state, :])
        
        # new Q
        new_value = reward + discount_factor * future_best_value
        
        # 4. Q-table update
        q_table[state, action] = old_value + learning_rate * (new_value - old_value)
        
        state = new_state
        
    # --- episode end, decay Epsilon ---
    epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)
    
    # print training progress
    if (episode + 1) % sessions == 0:
        avg_reward = total_reward / sessions / max_reward
        reward_history.append(avg_reward)
        print(f"Episode {episode + 1}/{total_episodes} - Epsilon: {epsilon:.4f} - Score:{avg_reward / max_reward}")
        total_reward = 0

# --- training ends ---
end_time = time.time()
print(f"--- Training complete ---")
print(f"used: {end_time - start_time:.2f} s")

# save Q-table
np.save("elevator_q_table.npy", q_table)
print("Q-table saved as 'elevator_q_table.npy'")

# === 新增：画 reward_history 图 ===
plt.figure(figsize=(8,4))
plt.plot(np.arange(1, len(reward_history)+1) * sessions, reward_history)
plt.xlabel("Episode")
plt.ylabel("Average Score (per 1000 episodes)")
plt.title("Training Reward Curve")
plt.grid(True)
plt.show()

# === 新增：打印 Q-table 稀疏度 ===
nonzero = np.count_nonzero(q_table)
total = q_table.size
print(f"\nQ-table non-zero entries: {nonzero}/{total}  ({nonzero/total*100:.2f}% non-zero)")

print("\n--- testing ---")
env.unwrapped.verbose = True #设置原本的verbose
test_episodes = 2
for episode in range(test_episodes):
    state, info = env.reset(seed=episode)
    terminated = False
    truncated = False
    total_reward = 0
    print(f"\n--- test episode {episode + 1} ---")
    
    while not terminated and not truncated:
        # --- 100% Utilization (Epsilon = 0) ---
        action = np.argmax(q_table[state, :])
        
        new_state, reward, terminated, truncated, info = env.step(action)
        # update reward
        state = new_state
        total_reward += reward

        # print info at every step
        print(f"  Action: {info['action_name']} -> floor: {info['current_floor']}, reward: {reward}")
    
    print(f"--- episode {episode + 1} end ---")
    print(f"Total reward: {total_reward}")

env.close()