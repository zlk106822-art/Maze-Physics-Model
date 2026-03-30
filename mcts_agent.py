import numpy as np
import matplotlib.pyplot as plt
import random
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
        
        print(f"小白鼠空投成功！起点: {self.start_pos}, 终点: {self.end_pos}")
        
        self.current_pos = self.start_pos
        self.last_pos = None 
        self.path =[self.start_pos]
        self.delta_p_list = [1] 
        self.mcts_triggers = []
        self.forced_march_queue =[] 
        self.visited_freq = {} 
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
        cy, cx = center_pos
        radius = 5 
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.rows and 0 <= nx < self.cols and self.maze[ny, nx] == 1:
                    dist = abs(dy) + abs(dx)
                    if dist <= radius:
                        self.macro_field[ny, nx] -= 0.3 * (1.0 - dist / (radius + 1))

    def step(self):
        if self.current_pos == self.end_pos:
            return True

        self.trauma_memory[self.current_pos] += 0.05
        self.visited_freq[self.current_pos] = self.visited_freq.get(self.current_pos, 0) + 1
        
        if self.visited_freq[self.current_pos] > 20: 
            print(f"【死锁保护】触发：{self.current_pos}")
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
            if not forward_neighbors: forward_neighbors = neighbors
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
            dist = abs(n[0] - self.end_pos[0]) + abs(n[1] - self.end_pos[1])
            score = self.macro_field[n] - (dist / (self.rows + self.cols)) * 0.1 - self.trauma_memory[n]
            if score > best_score:
                best_score = score
                best_n = n

        curr_dist = abs(self.current_pos[0] - self.end_pos[0]) + abs(self.current_pos[1] - self.end_pos[1])
        curr_score = self.macro_field[self.current_pos] - (curr_dist / (self.rows + self.cols)) * 0.1 - self.trauma_memory[self.current_pos]

        if len(forward_neighbors) == 0 or best_score <= curr_score:
            print(f"坐标 {self.current_pos} 遇阻！展开【双源终极波前搜索】...")
            self.mcts_triggers.append(self.current_pos)
            self._apply_local_mark(self.current_pos)
            
            escape_path, delta_p = self._wavefront_search(curr_dist)
            self.delta_p_list[-1] += delta_p
            
            if not escape_path: return True
            self.forced_march_queue = escape_path
            return self.step() 

        self.last_pos = self.current_pos
        self.current_pos = best_n
        self.path.append(self.current_pos)
        self.delta_p_list.append(1)
        return self.current_pos == self.end_pos

    def _wavefront_search(self, start_dist):
        """
        【V3.3 终极改进】：
        1. 双源注水：从当前点和上一步点同时开始。
        2. 终点绝对判定：离终点近时，不达红星绝不收网。
        3. 动态关闭修剪：终点前水量全开。
        """
        # 【双源注水】：允许水流从当前点和前一步同时出发，寻找错过的岔路
        frontier = [(self.current_pos, [])]
        if self.last_pos:
            frontier.append((self.last_pos, [self.last_pos]))
            
        flooded = {self.current_pos}
        if self.last_pos: flooded.add(self.last_pos)
        
        water_volume = 0 
        best_envelope_path =[]
        min_dist_to_goal = start_dist
        start_p = self.macro_field[self.current_pos]

        max_water = 30000     # 再次上调水量
        max_envelope = 60     

        while frontier:
            next_frontier =[]
            for curr, path in frontier:
                water_volume += 1
                if curr == self.end_pos: return path, water_volume
                    
                dist = abs(curr[0] - self.end_pos[0]) + abs(curr[1] - self.end_pos[1])
                p = self.macro_field[curr]
                
                if dist < min_dist_to_goal:
                    min_dist_to_goal = dist
                    best_envelope_path = path
                
                # 【关键逻辑修改】：
                # 如果离终点很近（<25步），取消一切中间判定，必须看到终点才算成功！
                if start_dist > 25:
                    jump_threshold = max(5, int(start_dist * 0.382))
                    if p > start_p + 0.05 or dist <= start_dist - jump_threshold:
                        return path, water_volume
                
                for n in self.get_physical_neighbors(curr):
                    if n not in flooded:
                        flooded.add(n)
                        next_frontier.append((n, path + [n]))
                        
            if water_volume > max_water: break
                
            # 【修剪优化】：如果离终点较近，彻底关闭修剪，保留所有火种
            if len(next_frontier) > max_envelope and start_dist > 30:
                scored_frontier = []
                for n, pth in next_frontier:
                    n_dist = abs(n[0] - self.end_pos[0]) + abs(n[1] - self.end_pos[1])
                    score = n_dist - (self.macro_field[n] * 15.0)
                    scored_frontier.append((score, n, pth))
                scored_frontier.sort(key=lambda x: x[0])
                keep = max_envelope // 2
                frontier = [(it[1], it[2]) for it in scored_frontier[:keep]] + \
                           random.sample([(it[1], it[2]) for it in scored_frontier[keep:]], min(len(scored_frontier)-keep, max_envelope-keep))
            else:
                frontier = next_frontier

        return best_envelope_path, water_volume

def run_and_render():
    print("Loading physics data...")
    try:
        maze_matrix = np.load('maze_data.npy')
        macro_field = np.load('macro_field_data.npy')
    except: return

    agent = Agent(maze_matrix, macro_field)
    for _ in range(agent.rows * agent.cols * 3):
        if agent.step(): break
            
    is_success = (agent.current_pos == agent.end_pos)
    shortest_path = []
    if is_success:
        path_dict = {pos: i for i, pos in enumerate(agent.path)}
        curr = agent.start_pos
        while curr != agent.end_pos:
            shortest_path.append(curr)
            neighbors = [n for n in agent.get_physical_neighbors(curr) if n in path_dict]
            curr = max(neighbors, key=lambda n: path_dict[n])
        shortest_path.append(agent.end_pos)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Wavefront V3.3: Dual-Source & Ultimate Goal Focus', fontsize=16, color='g' if is_success else 'r')
    ax1 = fig.add_subplot(121)
    ax1.imshow(maze_matrix, cmap='gray')
    ax1.imshow(macro_field, cmap='magma', alpha=0.5)
    path_y, path_x = zip(*agent.path)
    ax1.plot(path_x, path_y, color='magenta', alpha=0.6, linewidth=1.5, label='Exploration')
    if is_success:
        opt_y, opt_x = zip(*shortest_path)
        ax1.plot(opt_x, opt_y, color='#00FF00', linewidth=3.0, label='Success Path')
    ax1.plot(agent.start_pos[1], agent.start_pos[0], marker='*', color='cyan', markersize=15)
    ax1.plot(agent.end_pos[1], agent.end_pos[0], marker='*', color='red', markersize=15)
    if not is_success: ax1.plot(agent.current_pos[1], agent.current_pos[0], 'ks', markersize=12)
    if agent.mcts_triggers:
        ty, tx = zip(*agent.mcts_triggers)
        ax1.scatter(tx, ty, c='yellow', marker='X', s=80, edgecolors='black')
    ax1.legend()
    ax2 = fig.add_subplot(122)
    ax2.plot(agent.delta_p_list, color='r')
    ax2.set_yscale('log')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_and_render()