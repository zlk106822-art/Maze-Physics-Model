import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple

class MazeEnvironment:
    """
    一个使用深度优先搜索（DFS）算法生成并可视化完美迷宫的类。

    属性:
        width (int): 迷宫的宽度（以像素或单元格为单位）。
        height (int): 迷宫的高度。
        maze (np.ndarray): 存储迷宫布局的2D NumPy数组。
                           1 代表通路, 0 代表墙壁。
        start (Tuple[int, int]): 迷宫的起点坐标 (y, x)。
        end (Tuple[int, int]): 迷宫的终点坐标 (y, x)。
    """
    def __init__(self, width: int, height: int):
        """
        初始化迷宫环境。

        Args:
            width (int): 迷宫的期望宽度。必须为奇数。
            height (int): 迷宫的期望高度。必须为奇数。
        
        Raises:
            ValueError: 如果宽度或高度不是奇数。
        """
        if width % 2 == 0 or height % 2 == 0:
            raise ValueError("迷宫的宽度和高度必须是奇数，以确保四周有墙壁。")

        self.width = width
        self.height = height
        # 核心数据结构：0代表墙壁，迷宫在生成时被“雕刻”出来
        self.maze = np.zeros((height, width), dtype=np.uint8)
        
        # 定义起点和终点，通常在迷宫的对角
        # (1, 1) 是左上角的第一个通路单元格
        self.start = (1, 1)
        # (height-2, width-2) 是右下角的最后一个通路单元格
        self.end = (height - 2, width - 2)

        # 调用私有方法生成迷宫结构
        self._generate_maze()

    def _generate_maze(self):
        """
        使用深度优先搜索（DFS）的递归回溯算法生成迷宫。
        这是一种经典的完美迷宫生成算法，确保迷宫完全连通。
        """
        # 计算迷宫的“单元格”尺寸，我们的算法在这些单元格上操作
        # 迷宫坐标 (mx, my) 对应于单元格坐标 (cx, cy) 的关系是:
        # mx = 2*cx + 1, my = 2*cy + 1
        cell_width = (self.width - 1) // 2
        cell_height = (self.height - 1) // 2

        # 递归函数，用于“雕刻”迷宫路径
        def carve_path(cx, cy):
            # 将当前单元格标记为通路 (1)
            self.maze[2 * cy + 1, 2 * cx + 1] = 1

            # 定义四个方向：上、下、左、右
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            random.shuffle(directions)  # 随机打乱方向，确保迷宫的随机性

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy

                # 检查邻居单元格是否在边界内且未被访问
                if 0 <= nx < cell_width and 0 <= ny < cell_height:
                    # 如果邻居单元格是墙壁（即未访问）
                    if self.maze[2 * ny + 1, 2 * nx + 1] == 0:
                        # 移除当前单元格与邻居单元格之间的墙壁
                        self.maze[2 * cy + 1 + dy, 2 * cx + 1 + dx] = 1
                        # 递归地访问邻居
                        carve_path(nx, ny)

        # 从左上角的单元格 (0,0) 开始生成迷宫
        carve_path(0, 0)

    def render(self, save_path: str = "maze.png"):
        """
        使用 matplotlib 可视化迷宫，标记起点和终点，并保存为图像文件。

        Args:
            save_path (str): 保存迷宫图像的文件路径。默认为 'maze.png'。
        """
        # 创建一个RGB图像副本以标记起点和终点
        # np.stack 将灰度图扩展为3个通道
        # 乘以255是为了让matplotlib正确显示颜色
        maze_rgb = np.stack([self.maze * 255] * 3, axis=-1).astype(np.uint8)

        # 将起点和终点像素标记为红色 [R, G, B] -> [255, 0, 0]
        start_y, start_x = self.start
        end_y, end_x = self.end
        maze_rgb[start_y, start_x] = [255, 0, 0]  # 红色起点
        maze_rgb[end_y, end_x] = [255, 0, 0]      # 红色终点

        plt.figure(figsize=(10, 10))
        # imshow可以直接显示我们创建的RGB图像
        plt.imshow(maze_rgb)
        
        # 隐藏坐标轴，获得纯净的视觉效果
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        # 保存图像文件
        # bbox_inches='tight' 和 pad_inches=0 确保没有多余的白边
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"迷宫图像已保存至: {save_path}")

        # 显示图像
        plt.show()


if __name__ == "__main__":
    # --- 测试代码 ---
    # 实例化一个 65x65 的迷宫。
    # 注意：尺寸必须为奇数，以确保迷宫四周能被墙壁完美包裹。
    try:
        maze_env = MazeEnvironment(width=65, height=65)
        
        # 调用 render 方法来显示迷宫并将其保存为 "maze.png"
        maze_env.render()
        
                # 打印迷宫尺寸和起点/终点信息
        print(f"成功生成 {maze_env.width}x{maze_env.height} 的迷宫。")
        print(f"起点 (y, x): {maze_env.start}")
        print(f"终点 (y, x): {maze_env.end}")

        # --- 保存迷宫矩阵 ---
        # 将生成的迷宫矩阵保存为 .npy 文件，作为后续步骤的“客体原图”
        np.save('maze_data.npy', maze_env.maze)
        print("迷宫矩阵已成功保存到 'maze_data.npy'。")

    except ValueError as e:
        print(f"错误: {e}")
