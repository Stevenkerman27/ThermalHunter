import pygame
import numpy as np
import torch
import os
from chaser import RelativePursuitEnv
from ppo import Agent

class ChaserVisualizer:
    def __init__(self, env, width=800, height=600, world_size=100.0):
        self.env = env
        self.width = width
        self.height = height
        self.world_size = world_size # 视觉循环的周期
        
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PPO Chaser - Periodic Visualization")
        
        self.colors = {
            'background': (20, 20, 30),
            'grid': (255, 40, 50),
            'agent': (0, 200, 255),
            'target': (255, 100, 100),
            'arrow': (255, 255, 0),
            'text': (220, 220, 220),
            'info_bg': (30, 30, 40, 220)
        }
        
        self.font = pygame.font.SysFont('consolas', 16)
        # 缩放因子
        self.scale = min(width, height) / (world_size * 1.1)
        self.center_x = width // 2
        self.center_y = height // 2

    def world_to_screen(self, pos):
        """
        核心逻辑：保留周期化映射
        使用取模运算确保坐标落在 [-world_size/2, world_size/2] 范围内
        """
        x, y = pos[0], pos[1]
        
        # 对坐标进行取模映射，使其在视觉上循环
        x_mod = ((x + self.world_size / 2) % self.world_size) - self.world_size / 2
        y_mod = ((y + self.world_size / 2) % self.world_size) - self.world_size / 2
            
        screen_x = self.center_x + x_mod * self.scale
        screen_y = self.center_y - y_mod * self.scale # Y轴反转
        return int(screen_x), int(screen_y)

    def render_frame(self, obs, reward, episode, step, total_reward):
        self.screen.fill(self.colors['background'])
        
        # 绘制背景边界参考框
        half_s = (self.world_size / 2) * self.scale
        pygame.draw.rect(self.screen, self.colors['grid'], 
                         (self.center_x - half_s, self.center_y - half_s, half_s*2, half_s*2), 1)
        
        # 转换坐标并绘制
        t_pos = self.world_to_screen(self.env.target_pos)
        a_pos = self.world_to_screen(self.env.agent_pos)
        
        # 绘制目标 (红色)
        pygame.draw.circle(self.screen, self.colors['target'], t_pos, 10)
        # 绘制 Agent (青色)
        pygame.draw.circle(self.screen, self.colors['agent'], a_pos, 8)
        
        # 绘制朝向箭头
        head_x = a_pos[0] + 20 * np.cos(self.env.agent_heading)
        head_y = a_pos[1] - 20 * np.sin(self.env.agent_heading)
        pygame.draw.line(self.screen, self.colors['arrow'], a_pos, (head_x, head_y), 3)

        # 信息面板
        info_lines = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Total Reward: {total_reward:.2f}",
            f"Reward: {reward:.2f}",
            f"Current Dist: {np.linalg.norm(self.env.target_pos - self.env.agent_pos):.2f}",
            f"Action(phi): {obs[0]:.2f}"
        ]
        for i, text in enumerate(info_lines):
            surf = self.font.render(text, True, self.colors['text'])
            self.screen.blit(surf, (20, 20 + i * 22))
            
        pygame.display.flip()

def run_demo(model_path):
    # 环境初始化
    env = RelativePursuitEnv()
    
    # 模拟环境空间以初始化 Agent
    class MockEnvs:
        def __init__(self, env):
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(MockEnvs(env)).to(device)
    
    # 加载权重
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()
        print(f"成功加载模型: {model_path}")
    else:
        print(f"未找到模型文件: {model_path}")
        return

    # 可视化器
    visualizer = ChaserVisualizer(env)
    clock = pygame.time.Clock()
    
    episode = 0
    running = True
    
    while running:
        obs, _ = env.reset()
        episode += 1
        total_reward = 0
        step = 0
        done = False
        
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            # PPO 推理：使用均值以获得确定性表现
            with torch.no_grad():
                action = agent.actor_mean(torch.Tensor(obs).to(device).unsqueeze(0))
                action = action.cpu().numpy()[0]

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            
            visualizer.render_frame(obs, reward, episode, step, total_reward)
            clock.tick(30) # 限制 30 帧

        if running: pygame.time.wait(1000)

    pygame.quit()

if __name__ == "__main__":
    # 填入你训练生成的模型实际路径
    MODEL_PATH = "ppo.cleanrl_model"
    run_demo(MODEL_PATH)