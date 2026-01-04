import gymnasium as gym
import torch
import numpy as np
from ppo_continuous_action import Agent, make_env 
import os

# --- 修改这里 ---
ENV_ID = "KellyBetting-v0"
# 1. 获取当前脚本(eval_model.py)所在的绝对路径目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 拼接出模型文件的完整路径
model_filename = "test.cleanrl_model" 
MODEL_PATH = os.path.join(script_dir, model_filename)

print(f"正在尝试加载模型: {MODEL_PATH}") # 打印出来检查一下

# 1. 创建环境 (如果不加 render_mode，就是纯后台计算)
env = gym.make(ENV_ID) 

# --- 【新增】手动补充 Agent 缺失的属性 ---
# 欺骗 Agent，告诉它“单个环境”的空间就是当前这个环境的空间
env.single_observation_space = env.observation_space
env.single_action_space = env.action_space
# -------------------------------------

# 2. 初始化网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(env).to(device)

# 3. 加载模型
print(f"Loading model from {MODEL_PATH}...")
agent.load_state_dict(torch.load(MODEL_PATH, map_location=device))
agent.eval()

# 4. 运行测试
obs, _ = env.reset()
done = False
print("\n--- 实战测试 (Kelly Strategy) ---")

steps = 0
bet_ratios = []
balances = []

while not done:
    obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 获取动作分布的均值 (对于高斯分布，mean 是最理性的动作)
        action, _, _, _ = agent.get_action_and_value(obs_tensor)
        
    actual_action = action.cpu().numpy()[0]
    
    # 执行
    obs, reward, terminated, truncated, info = env.step(actual_action)
    done = terminated or truncated
    
    # 记录数据
    fraction = info['fraction']
    bet_ratios.append(fraction)
    balances.append(obs[0])
    
    steps += 1
    print(f"Step {steps:4d} | 本金: {np.e**obs[0]:10.2f} | 下注比例: {fraction:.8f} ({'Last Win' if info['win'] else 'Last Loss'})")

print("\n" + "="*30)
print(f"最终结果 ({steps} 步):")
print(f"最终本金: {np.e**obs[0]:.2f}")
print(f"平均下注比例: {np.mean(bet_ratios):.4f}")
print(f"凯利最优值(参考): 0.2000 (如果胜率0.6赔率1:1)")
print("="*30)