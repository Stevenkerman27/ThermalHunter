import torch
import numpy as np
import gymnasium as gym
import glob
import os
import config
from train_dqn import QNetwork, normalize_state
from eval import plot_climb_results

def run_eval_dqn(env, model, n_episodes=config.N_EVAL_EPISODES, policy_type="trained"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_rewards = []
    all_climb_heights = []
    
    for ep in range(n_episodes):
        random_reset_time = np.random.randint(config.RESET_TIME_MIN, config.RESET_TIME_MAX)
        state, info = env.reset(options={"resettime": random_reset_time})
        h_start = info["height"]
        
        ep_reward = 0
        done = False
        
        while not done:
            if policy_type == "random":
                action = env.action_space.sample()
            else:
                s_norm = normalize_state(state)
                state_t = torch.FloatTensor(s_norm).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(state_t).argmax().item()
            
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            
        all_rewards.append(ep_reward)
        all_climb_heights.append(info["height"] - h_start)
        
    return all_rewards, all_climb_heights

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5_files = sorted(glob.glob(os.path.join(config.WIND_DIR, 'snapshots_s*.h5')), key=config.natural_key)
    env = gym.make("GliderContinuous-v0", h5_file_path=h5_files, polar_file_base=config.POLAR_BASE, continuous_obs=True)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = QNetwork(state_dim, action_dim).to(device)
    
    if os.path.exists(config.DQN_SAVE_PATH):
        # 使用 map_location 确保在没有 GPU 的机器上也能加载 CUDA 模型
        model.load_state_dict(torch.load(config.DQN_SAVE_PATH, map_location=device))
        print(f"Loaded model from {config.DQN_SAVE_PATH}")
    else:
        print(f"Warning: Model not found at {config.DQN_SAVE_PATH}. Evaluating random/untrained model.")

    print(f"Evaluating DQN Policy over {config.N_EVAL_EPISODES} episodes...")
    rnd_rewards, rnd_climbs = run_eval_dqn(env, model, policy_type="random")
    exp_rewards, exp_climbs = run_eval_dqn(env, model, policy_type="trained")
    
    print(f"--- DQN Evaluation Statistics ---")
    print(f"Random Policy: Mean Climb={np.mean(rnd_climbs):.1f}m")
    print(f"Trained DQN:  Mean Climb={np.mean(exp_climbs):.1f}m")
    
    eval_plot_path = os.path.join(config.TRAIN_RESULT_DIR, "dqn_climb_eval_result.png")
    plot_climb_results(rnd_climbs, exp_climbs, save_path=eval_plot_path, show=True)
    
    env.close()
