import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import config

def plot_dqn_training(csv_path=None, save_path=None):
    if csv_path is None:
        csv_path = os.path.join(config.TRAIN_RESULT_DIR, "dqn_train_stats.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Warning: CSV file is empty.")
        return

    all_returns = df['return'].values
    all_climbs = df['climb'].values
    
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16
    })
    
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot episodic returns
    ax1.plot(all_returns, alpha=0.3, color='blue', label='Episodic Return')
    if len(all_returns) >= 50:
        moving_avg = np.convolve(all_returns, np.ones(50)/50, mode='valid')
        ax1.plot(range(49, len(all_returns)), moving_avg, color='blue', linewidth=2, label='Moving Avg (50)')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episodic Return', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    # Use dynamic limits based on data but keep a minimum for visibility
    ax1.set_ylim(100, 1500)
    ax1.set_title('DQN Training Performance')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot climb height on secondary Y axis
    ax1_climb = ax1.twinx()
    ax1_climb.plot(all_climbs, alpha=0.3, color='forestgreen', label='Climb Height')
    if len(all_climbs) >= 50:
        moving_avg_climb = np.convolve(all_climbs, np.ones(50)/50, mode='valid')
        ax1_climb.plot(range(49, len(all_climbs)), moving_avg_climb, color='darkgreen', linewidth=2, label='Moving Avg Climb')

    ax1_climb.set_ylabel('Climb Height (m)', color='forestgreen')
    ax1_climb.tick_params(axis='y', labelcolor='forestgreen')
    
    # Dynamic limits for climb
    ax1_climb.set_ylim(-200, 250)
    ax1_climb.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Legends
    # To combine legends from two different axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_climb.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(config.TRAIN_RESULT_DIR, "dqn_train_result.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"Training plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_dqn_training()
