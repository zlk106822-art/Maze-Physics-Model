import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class Agent:
    def __init__(self, maze_matrix, macro_field):
        self.maze = maze_matrix
        self.macro_field = macro_field
        self.rows, self.cols = maze_matrix.shape
        
        path_cells = np.argwhere(self.maze == 1)
        path_cells = sorted(path_cells, key=lambda p: p[0] + p[1])
        self.start_pos = tuple(path_cells[0])
        self.end_pos = tuple(path_cells[-1])
        
        print(f"小白鼠空投成功！真实起点: {self.start_pos}, 终点: {self.end_pos}")
        
        self.current_pos = self.start_pos
        self.last_pos = None 
        
        self.path =[self.start_pos]
        self.delta_p_list = [1] 
        self.mcts_triggers = []
        
        self.forced_march_queue =[] 
        self.visited_freq = {} 
        
        # 找回肉体记忆：极其微弱的足迹惩罚，仅用于打破局部死循环
        self.trauma_memory = np.zeros_like(self.macro_field, dtype=float)

    def get_physical_neighbors(self, pos):
        y, x = pos
        neighbors =[]
        for dy, dx in[(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.rows and 0 <= nx < self.cols and self.maze[ny, nx] == 1:
                neighbors.append((ny, nx))
        return neighbors

    def _apply_local_mark(self, center_pos):
        """
        【吸收您的建议】：在死锁终点投下局域标记！
        以死锁点为中心，降低周围一圈的宏观高度，彻底砸平这片骗人的虚假高地。
        """
        cy, cx = center_pos
        radius = 5 # 影响半径
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.rows and 0 <= nx < self.cols and self.maze[ny, nx] == 1:
                    dist = abs(dy) + abs(dx)
                    if dist <= radius:
                        # 距离越近砸得越深，中心扣除 0.3 的绝对高度
                        penalty = 0.3 * (1.0 - dist / (radius + 1))
                        self.macro_field[ny, nx] -= penalty

    def step(self):
        if self.current_pos == self.end_pos:
            return True

        # 微弱的物理踩踏记忆
        self.trauma_memory[self.current_pos] += 0.02

        self.visited_freq[self.current_pos] = self.visited_freq.get(self.current_pos, 0) + 1
        if self.visited_freq[self.current_pos] > 15: 
            print(f"【死锁保护】在 {self.current_pos} 陷入循环，退出。")
            return True

        if self.forced_march_queue:
            next_pos = self.forced_march_queue.pop(0)
            self.last_pos = self.current_pos
            self.current_pos = next_pos
            self.path.append(self.current_pos)
            self.delta_p_list.append(1) 
            return self.current_pos == self.end_pos

        neighbors = self.get_physical_neighbors(self.current_pos)
        forward_neighbors =[n for n in neighbors if n != self.last_pos]

        if len(neighbors) == 1:
            if not forward_neighbors:
                forward_neighbors = neighbors
        elif len(forward_neighbors) == 1:
            next_pos = forward_neighbors[0]
            self.last_pos = self.current_pos
            self.current_pos = next_pos
            self.path.append(self.current_pos)
            self.delta_p_list.append(1)
            return self.current_pos == self.end_pos

        best_n = None
        best_score = -float('inf')
        
        for n in forward_neighbors:
            p = self.macro_field[n]
            dist = abs(n[0] - self.end_pos[0]) + abs(n[1] - self.end_pos[1])
            dist_penalty = (dist / (self.rows + self.cols)) * 0.1 
            # 得分 = 宏观概率 - 终点引力 - 肉体踩踏记忆
            score = p - dist_penalty - self.trauma_memory[n]
            if score > best_score:
                best_score = score
                best_n = n

        curr_dist = abs(self.current_pos[0] - self.end_pos[0]) + abs(self.current_pos[1] - self.end_pos[1])
        curr_score = self.macro_field[self.current_pos] - (curr_dist / (self.rows + self.cols)) * 0.1 - self.trauma_memory[self.current_pos]

        # 发现前面的路还不如脚下好（遇到死坑或长隔断）
        if len(forward_neighbors) == 0 or best_score <= curr_score:
            print(f"坐标 {self.current_pos} 遇阻！投下局域标记并展开波前分析...")
            self.mcts_triggers.append(self.current_pos)
            
            # 【重要】：记录砸坑前的原始海拔和距离，作为波前溢出的参考系
            original_p = self.macro_field[self.current_pos]
            original_dist = curr_dist
            
            # 1. 应用您的建议：投下局域炸弹，砸平死角
            self._apply_local_mark(self.current_pos)
            
            # 2. 展开结合包络分析的水波纹
            escape_path, delta_p = self._wavefront_search(original_p, original_dist)
            
            self.delta_p_list[-1] += delta_p
            
            if not escape_path:
                print("极度异常：完全无路可走，智能体停机。")
                return True
                
            self.forced_march_queue = escape_path
            return self.step() 

        self.last_pos = self.current_pos
        self.current_pos = best_n
        self.path.append(self.current_pos)
        self.delta_p_list.append(1)
        
        return self.current_pos == self.end_pos

    def _wavefront_search(self, start_p, start_dist):
        """
        【吸收您的建议】：在水波扩散时，实时分析包络面！
        永远记录那条“离终点最近”的水流。
        """
        queue = deque([(self.current_pos, [])])
        flooded = {self.current_pos} 
        
        water_volume = 0 
        
        best_envelope_path =[]
        min_dist_to_goal = start_dist

        while queue:
            curr, path = queue.popleft()
            water_volume += 1
            
            # 绝对优先：水流碰到终点，直接胜利！
            if curr == self.end_pos:
                return path, water_volume
                
            dist = abs(curr[0] - self.end_pos[0]) + abs(curr[1] - self.end_pos[1])
            p = self.macro_field[curr]
            
            # 【包络面分析】：实时记录漫延过的地方中，离终点物理距离最近的路径！
            if dist < min_dist_to_goal:
                min_dist_to_goal = dist
                best_envelope_path = path
            
            # 破局条件：漫过了原始假山的高度，或者成功绕过长墙（距离大幅拉近）
            if p > start_p + 0.05 or dist < start_dist - 15:
                return path, water_volume
            
            # 算力限制
            if water_volume > 4000: 
                break
                
            for n in self.get_physical_neighbors(curr):
                if n not in flooded:
                    flooded.add(n)
                    queue.append((n, path + [n]))

        # 【修复临门一脚的遗憾】：如果水用光了还没溢出，
        # 绝不返回空！把包络面中离终点最近的那条路拿来走！
        return best_envelope_path, water_volume

def run_and_render():
    print("Loading physics data...")
    try:
        maze_matrix = np.load('maze_data.npy')
        macro_field = np.load('macro_field_data.npy')
    except FileNotFoundError:
        print("Data file not found. Please run flux_field.py first.")
        return

    agent = Agent(maze_matrix, macro_field)

    max_steps = agent.rows * agent.cols * 2
    for step in range(max_steps):
        if agent.step():
            break
            
    print(f"Simulation completed. Total real steps: {len(agent.path)}")

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Physical Execution: Wavefront Envelope & Local Collapse', fontsize=16)

    ax1 = fig.add_subplot(121)
    ax1.set_title('2D Trajectory & Quantum Walls')
    ax1.imshow(maze_matrix, cmap='gray', interpolation='none')
    ax1.imshow(macro_field, cmap='magma', alpha=0.5, interpolation='none')

    path_y, path_x = zip(*agent.path)
    ax1.plot(path_x, path_y, 'g-', linewidth=2.5, label='Agent Path')
    
    ax1.plot(agent.start_pos[1], agent.start_pos[0], marker='*', color='cyan', markersize=15, label='Start')
    ax1.plot(agent.end_pos[1], agent.end_pos[0], marker='*', color='red', markersize=15, label='End')

    if agent.mcts_triggers:
        mcts_y, mcts_x = zip(*agent.mcts_triggers)
        ax1.scatter(mcts_x, mcts_y, c='yellow', marker='X', s=80, edgecolors='black', label='Wavefront Burst (Delta P)', zorder=3)

    ax1.legend(loc='upper right')
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')

    ax2 = fig.add_subplot(122)
    ax2.set_title(r'Computational Cost ($\Delta p$) vs Physical Steps')
    ax2.plot(agent.delta_p_list, color='r', linewidth=1.5)
    ax2.set_xlabel('Real Physical Steps Taken')
    ax2.set_ylabel(r'Volume of Water Poured ($\Delta p$)')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig("final_physics_result.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    run_and_render()