import gymnasium as gym
import numpy as np
import imageio
import pygame
import os
import sys

# 将当前目录添加到路径以便导入 swimmer.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def save_multi_agent_gif(q_table_path, output_path, num_agents=10, grid_size=25, max_steps=100):
    # 加载 Q-Table
    try:
        q_table = np.load(q_table_path)
        print(f"成功加载 Q-Table: {q_table_path}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {q_table_path}")
        return

    # 初始化 pygame (用于渲染)
    pygame.init()
    window_size = 500
    pix_square_size = window_size / grid_size
    
    # 随机生成多个 Agent 和 Target
    agents = []
    for _ in range(num_agents):
        agent_pos = np.random.randint(0, grid_size, size=2)
        target_pos = np.random.randint(0, grid_size, size=2)
        while np.array_equal(agent_pos, target_pos):
            target_pos = np.random.randint(0, grid_size, size=2)
        agents.append({
            'pos': agent_pos.copy(), 
            'target': target_pos.copy(), 
            'done': False,
            'color': (0, 0, 255),        # 固定蓝色
            'target_color': (255, 0, 0)  # 固定红色
        })

    frames = []
    
    # 动作映射
    action_to_direction = {
        0: np.array([0, 1]),  # Up
        1: np.array([1, 0]),  # Right
        2: np.array([0, -1]), # Down
        3: np.array([-1, 0]), # Left
    }

    def get_toroidal_delta(pos1, pos2, size):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        if dx > size / 2: dx -= size
        elif dx < -size / 2: dx += size
        if dy > size / 2: dy -= size
        elif dy < -size / 2: dy += size
        return np.array([dx, dy])

    def get_obs(agent_pos, target_pos, size):
        delta = get_toroidal_delta(agent_pos, target_pos, size)
        dx, dy = delta[0], delta[1]
        is_up = 1 if dy > 0 else 0
        is_down = 1 if dy < 0 else 0
        is_left = 1 if dx < 0 else 0
        is_right = 1 if dx > 0 else 0
        return (is_up, is_down, is_left, is_right)

    print(f"正在按顺序模拟 {num_agents} 个 Agent...")
    
    for i, agent in enumerate(agents):
        print(f"Agent {i+1}/{num_agents} 正在移动...")
        step_count = 0
        while not agent['done'] and step_count < max_steps:
            # 1. 创建画布
            canvas = pygame.Surface((window_size, window_size))
            canvas.fill((255, 255, 255))
            
            # 画细网格线
            for x in range(grid_size + 1):
                pos = min(pix_square_size * x, window_size - 1)
                pygame.draw.line(canvas, (120, 120, 120), (0, pos), (window_size, pos), width=2)
                pygame.draw.line(canvas, (120, 120, 120), (pos, 0), (pos, window_size), width=2)

            # 绘制当前 Agent 的 Target
            pygame.draw.rect(canvas, agent['target_color'], pygame.Rect(
                agent['target'][0] * pix_square_size,
                (grid_size - 1 - agent['target'][1]) * pix_square_size,
                pix_square_size, pix_square_size
            ))
            
            # 绘制当前 Agent
            pygame.draw.rect(canvas, agent['color'], pygame.Rect(
                agent['pos'][0] * pix_square_size,
                (grid_size - 1 - agent['pos'][1]) * pix_square_size,
                pix_square_size, pix_square_size
            ))
            
            # 转换并保存帧
            frame = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
            frames.append(frame)

            # 获取观测并选择动作
            obs = get_obs(agent['pos'], agent['target'], grid_size)
            action = np.argmax(q_table[obs])
            
            # 移动
            agent['pos'] = (agent['pos'] + action_to_direction[action]) % grid_size
            step_count += 1
            
            # 检查是否到达
            if np.array_equal(agent['pos'], agent['target']):
                agent['done'] = True
                # 到达后多停两帧
                for _ in range(2):
                    frames.append(frame)

    print(f"保存 GIF 至 {output_path} (共 {len(frames)} 帧)...")
    imageio.mimsave(output_path, frames, fps=10, loop=0)
    pygame.quit()
    print("完成！")

if __name__ == "__main__":
    Q_TABLE_PATH = "practice/discrete swimmer/my_q_table.npy"
    OUTPUT_PATH = "practice/discrete swimmer/swimmer_10_agents.gif"
    save_multi_agent_gif(Q_TABLE_PATH, OUTPUT_PATH, num_agents=10)
