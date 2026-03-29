import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
from itertools import combinations

class ProbabilityFluxField:
    """
    基于给定的迷宫结构，生成并可视化概率通量场，并提供与原始物理空间的对齐参照。
    该最终版本修复了“单边组合爆炸”问题，并加入了视觉锚点以增强3D空间感。
    """
    def __init__(self, maze_matrix: np.ndarray, delta_x: int):
        """
        初始化概率通量场。

        Args:
            maze_matrix (np.ndarray): 0/1的二维numpy数组，代表迷宫。
            delta_x (int): 观测算符/分块大小。
        """
        if not isinstance(maze_matrix, np.ndarray):
            raise TypeError("maze_matrix必须是NumPy数组。")
        if not isinstance(delta_x, int) or delta_x <= 0:
            raise ValueError("delta_x必须是正整数。")
            
        self.maze = maze_matrix
        self.delta_x = delta_x
        self.height, self.width = self.maze.shape
        self.flux_matrix = np.zeros_like(self.maze, dtype=float)
        self.macro_field = None

    def _local_bfs(self, chunk_maze: np.ndarray, start_pos: tuple, end_pos: tuple):
        """
        在给定的区块内使用广度优先搜索（BFS）寻找两个点之间的最短路径。
        """
        h, w = chunk_maze.shape
        queue = deque([[start_pos]])
        visited = {start_pos}

        while queue:
            path = queue.popleft()
            y, x = path[-1]

            if (y, x) == end_pos:
                return path

            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and \
                   chunk_maze[ny, nx] == 1 and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    new_path = list(path)
                    new_path.append((ny, nx))
                    queue.append(new_path)
        return None

    def generate_field(self):
        """
        [流体力学终极版] 摒弃图论组合，引入“流体连续性方程”，彻底消灭能量奇点。
        """
        print("Generating flux field with Fluid Dynamics Continuity model...")
        
        self.flux_matrix = np.zeros_like(self.maze, dtype=float)

        # --- 步骤 1: 遍历区块，使用连通域寻找“宏观贯穿管” ---
        for y_start in range(0, self.height, self.delta_x):
            for x_start in range(0, self.width, self.delta_x):
                y_end = min(y_start + self.delta_x, self.height)
                x_end = min(x_start + self.delta_x, self.width)
                
                chunk_maze = self.maze[y_start:y_end, x_start:x_end]
                h_c, w_c = chunk_maze.shape
                chunk_flux = np.zeros_like(chunk_maze, dtype=float)

                # 核心魔法：直接标记区块内的所有连通水管
                # 0是墙，1,2,3...是互相独立的连通水池
                labeled_array, num_features = scipy.ndimage.label(chunk_maze)

                for i in range(1, num_features + 1):
                    mask = (labeled_array == i)

                    # 探测这滩水碰到了区块的哪几个面
                    touches_top = np.any(mask[0, :])
                    touches_bottom = np.any(mask[h_c-1, :])
                    touches_left = np.any(mask[:, 0])
                    touches_right = np.any(mask[:, w_c-1])

                    # 独立边界的数量
                    num_borders = sum([touches_top, touches_bottom, touches_left, touches_right])

                    # 【流体连续性物理法则】：
                    # 只有接触 >= 2 个面，水流才能贯穿这个区块（无论它在里面扭曲了多少次，通量最高就是 1.0）
                    # 如果只接触 1 个面，说明这是微观死胡同或原路返回的涡流（通量降至极低的基态 0.05）
                    if num_borders >= 2:
                        chunk_flux[mask] = 1.0
                    else:
                        chunk_flux[mask] = 0.05
                
                self.flux_matrix[y_start:y_end, x_start:x_end] += chunk_flux

        # --- 步骤 2: 宏观场信息压缩（高斯平滑） ---
        sigma = max(1.0, self.delta_x / 4.0)
        print(f"Step 2: Applying Gaussian filter (sigma={sigma:.2f})...")
        blurred_field = scipy.ndimage.gaussian_filter(self.flux_matrix, sigma=sigma)

        # --- 步骤 3: 归一化与非线性引力坍缩 ---
        min_val, max_val = blurred_field.min(), blurred_field.max()
        if max_val > min_val:
            normalized_field = (blurred_field - min_val) / (max_val - min_val)
        else:
            normalized_field = np.zeros_like(blurred_field)
        
        # 用 1.5 次方增强对比度，让大动脉形成锐利的山脊，毛细血管坍缩
        self.macro_field = np.power(normalized_field, 1.5)

        # 再次安全归一化
        min_val, max_val = self.macro_field.min(), self.macro_field.max()
        if max_val > min_val:
            self.macro_field = (self.macro_field - min_val) / (max_val - min_val)
        else:
            self.macro_field = np.zeros_like(self.macro_field)

        print("Flux field generation complete.")

    def render_field_with_reference(self, original_maze: np.ndarray, save_path: str = "flux_field_compare.png"):
        """
        [最终版] 可视化方法，加入视觉锚点以解决3D坐标错觉。
        """
        if self.macro_field is None:
            print("Error: Macro field has not been generated. Call generate_field() first.")
            return
            
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(f'Probability Flux Field Analysis (Observer Scale $\\Delta_x$={self.delta_x})', fontsize=16)

        rows, cols = original_maze.shape

        # --- 左侧子图: 2D 叠加 ---
        ax1 = fig.add_subplot(121)
        ax1.imshow(original_maze, cmap='gray', interpolation='none')
        im = ax1.imshow(self.macro_field, cmap='magma', alpha=0.7, interpolation='bilinear')
        ax1.set_title('2D: Physical Maze vs Macro Field')
        ax1.set_xlabel('X Axis (cols)')
        ax1.set_ylabel('Y Axis (rows)')
        
        # 【添加2D视觉锚点】
        # 注意：plot的坐标顺序是(x, y)，与数组索引(row, col)相反
        ax1.plot(0, 0, marker='*', color='cyan', markersize=15, label='Start (0,0)')
        ax1.plot(cols - 1, rows - 1, marker='*', color='red', markersize=15, label='End')
        # 将图例移至画框正上方，ncol=2 让图例横向排列，绝对不挡迷宫
        ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 1.02), borderaxespad=0., ncol=2)
        
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        # --- 右侧子图: 3D 表面图 ---
        ax2 = fig.add_subplot(122, projection='3d')
        
        X, Y = np.meshgrid(range(cols), range(rows))
        
        ax2.plot_surface(X, Y, self.macro_field, cmap='magma', edgecolor='none', alpha=0.9)
        
        ax2.set_title('3D: Probability Flux Ridge')
        ax2.set_xlabel('X Axis (cols)')
        ax2.set_ylabel('Y Axis (rows)')
        ax2.set_zlabel('Probability Amplitude')
        
        # 【添加3D视觉锚点】
        # Z值略微抬高以确保可见性
        ax2.scatter(0, 0, self.macro_field[0, 0] + 0.1, color='cyan', s=100, marker='*', depthshade=False, label='Start (0,0)')
        ax2.scatter(cols - 1, rows - 1, self.macro_field[rows - 1, cols - 1] + 0.1, color='red', s=100, marker='*', depthshade=False, label='End')
        
        ax2.invert_yaxis()
        
        # 【调整为更符合直觉的等距视角】
        ax2.view_init(elev=50, azim=-60)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=300)
        print(f"Field analysis comparison saved to: {save_path}")
        
        plt.show()

if __name__ == '__main__':
    print("--- Field Analysis: Loading 'Ground Truth' and generating final corrected flux field ---")
    try:
        maze_data = np.load('maze_data.npy')
        print(f"Successfully loaded 'maze_data.npy', shape: {maze_data.shape}")

        flux_field = ProbabilityFluxField(maze_matrix=maze_data, delta_x=16)

        flux_field.generate_field()

        flux_field.render_field_with_reference(original_maze=maze_data)

    except FileNotFoundError:
        print("Error: 'maze_data.npy' not found. Please run maze_env.py first to generate it.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")