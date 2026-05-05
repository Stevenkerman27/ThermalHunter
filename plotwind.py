import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import os
import glob
from glider_discrete_simp import RBWindField
#plots gif of the wind
# =================配置区域=================
# 物理域大小 (用于坐标轴标签)
DOMAIN_SIZE_PHYSICAL = (100.0, 100.0, 100.0) # (X, Y, Z)
# 切片位置：Y轴中间
Y_SLICE_RATIO = 0.5 
# 动画跳帧设置 (每隔N帧渲染一帧，减小GIF体积)
FRAME_SKIP = 1
# 动画帧率 (FPS)
FPS = 5
# 输出文件名
OUTPUT_GIF = "convective_plumes_xz.gif"

plt.rcParams.update({'font.size': 16})


# =========================================

def find_raw_data_slice(wf, t_global, y_idx):
    """
    辅助函数：根据全局时间索引，定位并读取原始数据的XZ切片
    无需插值，直接读取 HDF5 dataset
    """
    # 1. 定位文件和局部索引
    file_idx = 0
    for i in range(len(wf.t_offsets) - 1):
        if wf.t_offsets[i] <= t_global < wf.t_offsets[i+1]:
            file_idx = i
            break
    local_t = t_global - wf.t_offsets[file_idx]
    
    # 2. 获取对应文件的数据集句柄
    # 注意：这里利用了之前修改 RBWindField 时暴露的 dsets_list
    dsets = wf.dsets_list[file_idx]
    
    # 3. 读取垂直速度 uz 的切片
    # 数据形状通常为 (T, X, Y, Z)，我们需要固定 T 和 Y
    # slice_data shape 将会是 (X, Z)
    slice_data = dsets['buoyancy'][local_t, :, y_idx, :]
    
    return slice_data

def render_gif():
    # --- 初始化 ---
    wind_dir = os.path.join(os.path.dirname(__file__), 'wind')
    h5_files = sorted(glob.glob(os.path.join(wind_dir, 'snapshots_s*.h5')))
    if not h5_files:
        print("错误：未找到风场文件。")
        return

    print("正在初始化风场数据...")
    try:
        # 我们只需要网格形状和时间信息，domain_size 在这里不影响原始数据读取
        wf = RBWindField(h5_files, domain_size=DOMAIN_SIZE_PHYSICAL, memory_mode=False)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    total_steps = wf.max_t_idx + 1
    grid_shape = wf.space_range # [Nx, Ny, Nz]
    
    # --- 修正后的绘图准备部分 ---

    # 1. 修正 grid_shape 的解包逻辑
    # grid_shape 是 [Nx, Ny, Nz]
    nx = int(grid_shape[0])
    ny = int(grid_shape[1])
    nz = int(grid_shape[2])

    # 2. 修正 Y 轴切片索引
    y_slice_idx = int(ny * Y_SLICE_RATIO)
    print(f"将在 Y索引 = {y_slice_idx} (物理位置约 Y={DOMAIN_SIZE_PHYSICAL[1]*Y_SLICE_RATIO:.1f}) 处进行 XZ 切片。")
    fig, ax = plt.subplots(figsize=(10, 6))
    # 3. 修正占位数据形状
    # imshow 的输入形状为 (行, 列)，在 XZ 平面图中：
    # 行对应 Z 轴 (高度)，即 nz
    # 列对应 X 轴 (水平)，即 nx
    placeholder_data = np.zeros((nz, nx)) 

    # 4. 绘图设置保持不变
    extent = [0, DOMAIN_SIZE_PHYSICAL[0], 0, DOMAIN_SIZE_PHYSICAL[2]]
    im = ax.imshow(placeholder_data, cmap='RdBu_r', vmin=0, vmax=1, 
                origin='lower', extent=extent, aspect='equal')
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Vertical Wind Velocity')

    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    title_text = ax.set_title('')

    start_frame = 0   
    frames_to_render = range(start_frame, total_steps, FRAME_SKIP)
    print(f"预计渲染帧数: {len(frames_to_render)}")

    # --- 动画更新逻辑 (手动循环以节省内存) ---
    def get_frame(frame_t_idx):
        # 1. 获取原始切片数据 (Shape: X, Z)
        slice_raw = find_raw_data_slice(wf, frame_t_idx, y_slice_idx)
        
        # 2. 转置数据以适配 imshow (变为 Z行, X列)
        slice_plotting = slice_raw.T
        
        # 3. 更新图像数据
        im.set_data(slice_plotting)
        
        # 4. 更新标题 (显示物理时间)
        sim_time = wf.all_sim_times[frame_t_idx]
        title_text.set_text(f'Frame: {frame_t_idx} | Time: {sim_time:.2f}')
        
        fig.canvas.draw()
        # 更加稳健的转换方式
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        return image[:, :, :3] # 去掉 Alpha 通道

    # --- 创建并保存动画 ---
    print(f"开始渲染 GIF 到 {OUTPUT_GIF} (可能需要几分钟)...")
    try:
        # 使用 imageio 串流写入 GIF
        with imageio.get_writer(OUTPUT_GIF, mode='I', duration=1000/FPS, loop=0) as writer:
            for i, frame_t_idx in enumerate(frames_to_render):
                frame_image = get_frame(frame_t_idx)
                writer.append_data(frame_image)
                if (i + 1) % 10 == 0:
                    print(f"已处理 {i + 1} / {len(frames_to_render)} 帧...")
        print("渲染完成！")
    except Exception as e:
        print(f"保存 GIF 失败: {e}")
    finally:
        wf.close()
        plt.close(fig)

if __name__ == "__main__":
    render_gif()