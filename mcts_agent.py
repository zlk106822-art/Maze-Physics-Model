import numpy as np
import matplotlib.pyplot as plt
import random

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
        # 足迹创伤记录
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
        """局部场坍缩：砸平虚假高地"""
        cy, cx = center_pos
        radius = 5 
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.rows and 0 <= nx < self.cols and self.maze[ny, nx] == 1:
                    dist = abs(dy) + abs(dx)
                    if dist <= radius:
                        # 惩罚值
                        self.macro_field[ny, nx] -= 0.2 * (1.0 - dist / (radius + 1))

    def step(self):
        if self.current_pos == self.end_pos:
            return True

        # 实时足迹惩罚
        self.trauma_memory[self.current_pos] += 0.05

        self.visited_freq[self.current_pos] = self.visited_freq.get(self.current_pos, 0) + 1
        if self.visited_freq[self.current_pos] > 20: 
            print(f"【紧急保护】死锁检测：{self.current_pos}")
            return True

        # 执行强行军队列
        if self.forced_march_queue:
            next_pos = self.forced_march_queue.pop(0)
            self.last_pos = self.current_pos
            self.current_pos = next_pos
            self.path.append(self.current_pos)
            self.delta_p_list.append(1) 
            return self.current_pos == self.end_pos

        neighbors = self.get_physical_neighbors(self.current_pos)
        forward_neighbors =[n for n in neighbors if n != self.last_pos]

        # 基础寻路逻辑（单行道直冲）
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

        # 路口决策
        best_n = None
        best_score = -float('inf')
        
        for n in forward_neighbors:
            p = self.macro_field[n]
            dist = abs(n[0] - self.end_pos[0]) + abs(n[1] - self.end_pos[1])
            dist_penalty = (dist / (self.rows + self.cols)) * 0.1 
            score = p - dist_penalty - self.trauma_memory[n]
            if score > best_score:
                best_score = score
                best_n = n

        curr_dist = abs(self.current_pos[0] - self.end_pos[0]) + abs(self.current_pos[1] - self.end_pos[1])
        curr_score = self.macro_field[self.current_pos] - (curr_dist / (self.rows + self.cols)) * 0.1 - self.trauma_memory[self.current_pos]

        # 触发波前搜索
        if len(forward_neighbors) == 0 or best_score <= curr_score:
            print(f"坐标 {self.current_pos} 遇阻！展开【黄金分割自适应搜索】...")
            self.mcts_triggers.append(self.current_pos)
            
            original_p = self.macro_field[self.current_pos]
            original_dist = curr_dist
            
            self._apply_local_mark(self.current_pos)
            
            escape_path, delta_p = self._wavefront_search(original_p, original_dist)
            
            self.delta_p_list[-1] += delta_p
            
            if not escape_path:
                print("停止：所有火种熄灭。")
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
        【吸收您的全部思想】：
        1. 黄金分割逃生门槛 (0.382)
        2. 包络面修剪 + 火种保留
        3. 自适应水量
        """
        frontier = [(self.current_pos, [])]
        flooded = {self.current_pos} 
        
        water_volume = 0 
        best_envelope_path =[]
        min_dist_to_goal = start_dist

        max_water = 15000     
        max_envelope = 50     

        # 【核心：黄金分割缩进】
        # 门槛不再固定，而是随着离终点越近而越小
        jump_threshold = max(1, int(start_dist * 0.382))
        if jump_threshold > 15: jump_threshold = 15 # 封顶，防止远距离过度溢出

        while frontier:
            next_frontier =[]
            
            for curr, path in frontier:
                water_volume += 1
                
                # 终点绝对检查（临门一脚）
                if curr == self.end_pos:
                    return path, water_volume
                    
                dist = abs(curr[0] - self.end_pos[0]) + abs(curr[1] - self.end_pos[1])
                p = self.macro_field[curr]
                
                if dist < min_dist_to_goal:
                    min_dist_to_goal = dist
                    best_envelope_path = path
                
                # 【自适应破局判定】
                # 离得越近，要求的“跃迁”步数越小
                if p > start_p + 0.05 or dist <= start_dist - jump_threshold:
                    return path, water_volume
                        
                for n in self.get_physical_neighbors(curr):
                    if n not in flooded:
                        flooded.add(n)
                        next_frontier.append((n, path + [n]))
                        
            if water_volume > max_water: 
                break
                
            # 包络面修剪与火种保留
            if len(next_frontier) > max_envelope:
                scored_frontier =[]
                for n, pth in next_frontier:
                    n_dist = abs(n[0] - self.end_pos[0]) + abs(n[1] - self.end_pos[1])
                    n_p = self.macro_field[n]
                    score = n_dist - (n_p * 15.0)
                    scored_frontier.append((score, n, pth))
                    
                scored_frontier.sort(key=lambda x: x[0])
                keep_elite = max_envelope // 2
                elites = [(item[1], item[2]) for item in scored_frontier[:keep_elite]]
                remaining = [(item[1], item[2]) for item in scored_frontier[keep_elite:]]
                num_sparks = min(len(remaining), max_envelope - keep_elite)
                sparks = random.sample(remaining, num_sparks)
                frontier = elites + sparks
            else:
                frontier = next_frontier

        return best_envelope_path, water_volume

def run_and_render():
    print("Loading physics data...")
    try:
        maze_matrix = np.load('maze_data.npy')
        macro_field = np.load('macro_field_data.npy')
    except FileNotFoundError:
        print("Data file not found. Run flux_field.py first.")
        return

    agent = Agent(maze_matrix, macro_field)

    max_steps = agent.rows * agent.cols * 2
    for step in range(max_steps):
        if agent.step():
            break
            
    print(f"Simulation completed. Total real steps: {len(agent.path)}")

    # 【修复核心：死亡确认机制】
    is_success = (agent.current_pos == agent.end_pos)

    shortest_path =[]
    if is_success:
        # --- 核心逻辑：最优路径提纯 (Path Pruning) ---
        path_dict = {pos: i for i, pos in enumerate(agent.path)}
        curr = agent.start_pos
        while curr != agent.end_pos:
            shortest_path.append(curr)
            neighbors =[n for n in agent.get_physical_neighbors(curr) if n in path_dict]
            if not neighbors: # 极端情况防错
                break
            curr = max(neighbors, key=lambda n: path_dict[n])
        shortest_path.append(agent.end_pos)
    else:
        print("【警告】智能体未能到达终点（死锁或步数耗尽），正在绘制挣扎遗像...")

    # --- 视觉渲染 ---
    fig = plt.figure(figsize=(16, 8))
    title_text = 'Physical Execution: Success (Distilled Path)' if is_success else 'Physical Execution: FAILED (Deadlock)'
    fig.suptitle(title_text, fontsize=16, color='green' if is_success else 'red')

    ax1 = fig.add_subplot(121)
    ax1.set_title('2D: Exploration Noise (Magenta) vs distilled Path (Green)')
    ax1.imshow(maze_matrix, cmap='gray', interpolation='none')
    ax1.imshow(macro_field, cmap='magma', alpha=0.5, interpolation='none')

    # 1. 绘制【探索噪声/死胡同】：淡红紫色 (Magenta)
    path_y, path_x = zip(*agent.path)
    ax1.plot(path_x, path_y, color='magenta', alpha=0.6, linewidth=1.5, label='Exploration Noise')
    
    # 2. 如果成功，绘制【成功路径】：鲜绿色 (Green)
    if is_success and shortest_path:
        opt_y, opt_x = zip(*shortest_path)
        ax1.plot(opt_x, opt_y, color='#00FF00', alpha=1.0, linewidth=3.0, label='Distilled Success Path', zorder=4)
    
    # 标注起点和终点
    ax1.plot(agent.start_pos[1], agent.start_pos[0], marker='*', color='cyan', markersize=15, label='Start', zorder=5)
    ax1.plot(agent.end_pos[1], agent.end_pos[0], marker='*', color='red', markersize=15, label='End', zorder=5)

    # 如果死在半路，用一个醒目的黑色骷髅头/标记画出它的死亡地点！
    if not is_success:
        ax1.plot(agent.current_pos[1], agent.current_pos[0], marker='s', color='black', markersize=12, label='Death Point', zorder=6)

    if agent.mcts_triggers:
        mcts_y, mcts_x = zip(*agent.mcts_triggers)
        ax1.scatter(mcts_x, mcts_y, c='yellow', marker='X', s=80, edgecolors='black', label='MCTS/Wavefront Burst', zorder=6)

    ax1.legend(loc='upper right', fontsize='small')
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')

    # 右图: 算力脉冲
    ax2 = fig.add_subplot(122)
    ax2.set_title(r'Computational Cost ($\Delta p$) - Intelligence Pulse')
    ax2.plot(agent.delta_p_list, color='r', linewidth=1.5)
    ax2.set_xlabel('Real Physical Steps Taken')
    ax2.set_ylabel(r'Compute Volume ($\Delta p$)')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig("distilled_path_result.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    run_and_render()