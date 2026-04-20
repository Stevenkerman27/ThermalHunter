import numpy as np
import torch
import os
from chaser import RelativePursuitEnv
from ppo import Agent

class MockEnvs:
    """模拟环境空间，用于初始化Agent网络"""
    def __init__(self, env):
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space

def evaluate_policy(env, model_path=None, num_episodes=100, device="cpu"):
    """
    评估策略性能。
    如果 model_path 为空，则使用 action=0 的基准纯追踪策略。
    """
    use_ppo = model_path is not None
    
    if use_ppo:
        agent = Agent(MockEnvs(env)).to(device)
        if os.path.exists(model_path):
            agent.load_state_dict(torch.load(model_path, map_location=device))
            agent.eval()
        else:
            raise FileNotFoundError(f"未找到模型文件: {model_path}")
            
    total_rewards = []
    success_counts = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            if use_ppo:
                with torch.no_grad():
                    # 评估模式：使用均值以获得确定的最优动作
                    obs_tensor = torch.Tensor(obs).to(device).unsqueeze(0)
                    action = agent.actor_mean(obs_tensor).cpu().numpy()[0]
            else:
                # 基准策略：动作始终为0（始终指向目标当前的视线角 LOS）
                action = np.array([0.0], dtype=np.float32)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        
        # 判断是否成功捕获 (依据环境设定，距离小于 d_min 即为捕获)
        final_dist = np.linalg.norm(env.target_pos - env.agent_pos)
        if final_dist < env.d_min:
            success_counts += 1

        total_rewards.append(ep_reward)

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    success_rate = (success_counts / num_episodes) * 100.0
    
    return avg_reward, std_reward, success_rate

if __name__ == "__main__":
    # 参数设置
    N = 500  # 测试回合数
    MODEL_PATH = "ppo.cleanrl_model"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = RelativePursuitEnv()
    
    print(f"开始评估，测试回合数 N={N}，设备: {device}\n")
    print("-" * 40)
    
    # 1. 评估基准策略 (纯追踪, Action=0)
    print("【策略 1】: 单纯指向目标 (Action = 0)")
    base_avg, base_std, base_sr = evaluate_policy(env, model_path=None, num_episodes=N, device=device)
    print(f"平均 Reward:   {base_avg:.2f} ± {base_std:.2f}")
    print(f"捕获成功率:    {base_sr:.1f}%")
    print("-" * 40)

    # 2. 评估 PPO 策略
    print(f"【策略 2】: PPO 模型 ({MODEL_PATH})")
    try:
        ppo_avg, ppo_std, ppo_sr = evaluate_policy(env, model_path=MODEL_PATH, num_episodes=N, device=device)
        print(f"平均 Reward:   {ppo_avg:.2f} ± {ppo_std:.2f}")
        print(f"捕获成功率:    {ppo_sr:.1f}%")
    except FileNotFoundError as e:
        print(e)
    print("-" * 40)