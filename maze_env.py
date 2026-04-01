import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple

class MazeEnvironment:
    """
    一个使用深度优先搜索(DFS)算法生成“完美迷宫”后，
    再通过随机“编织”创造回路，最终生成一个多连通复杂迷宫的类。
    """
    def __init__(self, width: int, height: int, braid_p: float = 0.08):
        """
        初始化迷宫环境。

        Args:
            width (int): 迷宫的期望宽度。必须为奇数。
            height (int): 迷宫的期望高度。必须为奇数。
            braid_p (float): 编织概率，即拆除一堵有效墙壁以制造回路的概率。
        """
        if width % 2 == 0 or height % 2 == 0:
            raise ValueError("迷宫的宽度和高度必须是奇数，以确保四周有墙壁。")

        self.width = width
        self.height = height
        self.maze = np.zeros((height, width), dtype=np.uint8)
        
        self.start = (1, 1)
        self.end = (height - 2, width - 2)

        # 1. 生成一个完美的DFS迷宫
        self._generate_maze()
        
        # 2. 对完美迷宫进行编织，创造回路
        self.braid_maze(p=braid_p)

    def _generate_maze(self):
        """
        使用深度优先搜索(DFS)的递归回溯算法生成一个标准的完美迷宫。
        """
        cell_width = (self.width - 1) // 2
        cell_height = (self.height - 1) // 2

        def carve_path(cx, cy):
            self.maze[2 * cy + 1, 2 * cx + 1] = 1
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < cell_width and 0 <= ny < cell_height and self.maze[2 * ny + 1, 2 * nx + 1] == 0:
                    self.maze[2 * cy + 1 + dy, 2 * cx + 1 + dx] = 1
                    carve_path(nx, ny)
        
        carve_path(0, 0)

    def braid_maze(self, p: float = 0.05):
        """
        遍历迷宫，以概率 p 拆除部分墙壁，创造回路。
        这会将一个“完美迷宫”变成一个“多连通编织迷宫”。
        
        Args:
            p (float): 拆除一堵有效墙壁的概率。
        """
        print(f"Braining maze with p={p}...")
        # 遍历所有内部墙壁（跳过边界）
        for r in range(1, self.height - 1):
            for c in range(1, self.width - 1):
                # 如果当前位置是墙壁
                if self.maze[r, c] == 0:
                    # 检查是否是分隔两个通路的墙
                    # 情况1: 水平分隔 (左边和右边都是通路)
                    is_horizontal_divider = (self.maze[r, c-1] == 1 and self.maze[r, c+1] == 1)
                    # 情况2: 垂直分隔 (上边和下边都是通路)
                    is_vertical_divider = (self.maze[r-1, c] == 1 and self.maze[r+1, c] == 1)

                    if (is_horizontal_divider or is_vertical_divider) and random.random() < p:
                        self.maze[r, c] = 1 # 拆掉这堵墙

    def render(self, save_path: str = "maze.png"):
        """
        使用 matplotlib 可视化迷宫。
        """
        maze_rgb = np.stack([self.maze * 255] * 3, axis=-1).astype(np.uint8)
        
        start_y, start_x = self.start
        end_y, end_x = self.end
        maze_rgb[start_y, start_x] = [255, 0, 0]
        maze_rgb[end_y, end_x] = [255, 0, 0]

        plt.figure(figsize=(10, 10))
        plt.imshow(maze_rgb)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Maze image saved to: {save_path}")
        plt.show()

if __name__ == "__main__":
    # --- 测试代码 ---
    try:
        # 现在生成的将直接是多连通迷宫
        maze_env = MazeEnvironment(width=65, height=65, braid_p=0.08)
        
        maze_env.render()
        
        print(f"Successfully generated a {maze_env.width}x{maze_env.height} braided maze.")
        print(f"Start (y, x): {maze_env.start}")
        print(f"End (y, x): {maze_env.end}")

        np.save('maze_data.npy', maze_env.maze)
        print("Braided maze matrix has been saved to 'maze_data.npy'.")

    except ValueError as e:
        print(f"Error: {e}")