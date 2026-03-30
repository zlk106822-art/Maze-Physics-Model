import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import random
from collections import deque
from itertools import combinations
import time

# --- Module 1: Maze Environment Code ---
class MazeEnvironment:
    """Generates a perfect maze using DFS."""
    def __init__(self, width: int, height: int):
        if width % 2 == 0 or height % 2 == 0:
            raise ValueError("Maze dimensions must be odd.")
        self.width = width
        self.height = height
        self.maze = np.zeros((height, width), dtype=np.uint8)
        self._generate_maze()

    def _generate_maze(self):
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

# --- Module 2: Flux Field Code ---
class ProbabilityFluxField:
    """Calculates the macro-field using the Fluid Dynamics model."""
    def __init__(self, maze_matrix: np.ndarray, delta_x: int):
        self.maze = maze_matrix
        self.delta_x = delta_x
        self.height, self.width = self.maze.shape
        self.flux_matrix = np.zeros_like(self.maze, dtype=float)
        self.macro_field = None

    def generate_field(self):
        for y_start in range(0, self.height, self.delta_x):
            for x_start in range(0, self.width, self.delta_x):
                y_end = min(y_start + self.delta_x, self.height)
                x_end = min(x_start + self.delta_x, self.width)
                
                chunk_maze = self.maze[y_start:y_end, x_start:x_end]
                h_c, w_c = chunk_maze.shape
                chunk_flux = np.zeros_like(chunk_maze, dtype=float)

                labeled_array, num_features = scipy.ndimage.label(chunk_maze)
                for i in range(1, num_features + 1):
                    mask = (labeled_array == i)
                    touches_top = np.any(mask[0, :])
                    touches_bottom = np.any(mask[h_c-1, :]) if h_c > 1 else False
                    touches_left = np.any(mask[:, 0])
                    touches_right = np.any(mask[:, w_c-1]) if w_c > 1 else False
                    num_borders = sum([touches_top, touches_bottom, touches_left, touches_right])
                    
                    chunk_flux[mask] = 1.0 if num_borders >= 2 else 0.05
                
                self.flux_matrix[y_start:y_end, x_start:x_end] += chunk_flux

        sigma = max(1.0, self.delta_x / 4.0)
        blurred_field = scipy.ndimage.gaussian_filter(self.flux_matrix, sigma=sigma)

        min_val, max_val = blurred_field.min(), blurred_field.max()
        normalized_field = (blurred_field - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(blurred_field)
        
        self.macro_field = np.power(normalized_field, 1.5)
        min_val, max_val = self.macro_field.min(), self.macro_field.max()
        self.macro_field = (self.macro_field - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(self.macro_field)

# --- Module 3: Agent Code (V3.3) ---
class Agent:
    """The final intelligent agent with the Wavefront Envelope search."""
    def __init__(self, maze_matrix, macro_field):
        self.maze = maze_matrix
        self.macro_field = np.copy(macro_field) # Use a copy to allow local modifications
        self.rows, self.cols = maze_matrix.shape
        
        path_cells = np.argwhere(self.maze == 1)
        # Robustly find start and end
        self.start_pos = (1, 1)
        self.end_pos = (self.rows - 2, self.cols - 2)
        
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
                if 0 <= ny < self.rows and 0 <= nx < self.cols:
                    dist = abs(dy) + abs(dx)
                    if dist <= radius:
                        penalty = 0.3 * (1.0 - dist / (radius + 1))
                        self.macro_field[ny, nx] = max(0, self.macro_field[ny, nx] - penalty)

    def step(self):
        if self.current_pos == self.end_pos:
            return True

        self.trauma_memory[self.current_pos] += 0.05
        self.visited_freq[self.current_pos] = self.visited_freq.get(self.current_pos, 0) + 1
        
        if self.visited_freq[self.current_pos] > 20: 
            return True # Deadlock, mark as failure

        if self.forced_march_queue:
            next_pos = self.forced_march_queue.pop(0)
            self.last_pos = self.current_pos
            self.current_pos = next_pos
            self.path.append(self.current_pos)
            self.delta_p_list.append(1) 
            return False

        neighbors = self.get_physical_neighbors(self.current_pos)
        forward_neighbors =[n for n in neighbors if n != self.last_pos]
        if not forward_neighbors and neighbors:
            forward_neighbors = neighbors

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

        if not forward_neighbors or best_score <= curr_score:
            self.mcts_triggers.append(self.current_pos)
            self._apply_local_mark(self.current_pos)
            escape_path, delta_p = self._wavefront_search(curr_dist)
            self.delta_p_list[-1] += delta_p
            if not escape_path: return True # Trapped, mark as failure
            self.forced_march_queue = escape_path
            return self.step() 

        self.last_pos = self.current_pos
        self.current_pos = best_n
        self.path.append(self.current_pos)
        self.delta_p_list.append(1)
        return False

    def _wavefront_search(self, start_dist):
        frontier = deque([(self.current_pos, [])])
        if self.last_pos:
            frontier.append((self.last_pos, [self.last_pos]))
        
        flooded = {self.current_pos, self.last_pos}
        water_volume = 0 
        best_envelope_path = []
        min_dist_to_goal = start_dist
        start_p = self.macro_field[self.current_pos]
        
        max_water = 4000 # Safety break for wavefront

        while frontier:
            water_volume += 1
            if water_volume > max_water: break
            
            curr, path = frontier.popleft()
            if curr == self.end_pos: return path, water_volume
                
            dist = abs(curr[0] - self.end_pos[0]) + abs(curr[1] - self.end_pos[1])
            if dist < min_dist_to_goal:
                min_dist_to_goal = dist
                best_envelope_path = path

            if start_dist > 25:
                jump_threshold = max(5, int(start_dist * 0.382))
                if self.macro_field[curr] > start_p + 0.05 or dist <= start_dist - jump_threshold:
                    return path, water_volume
            
            for n in self.get_physical_neighbors(curr):
                if n not in flooded:
                    flooded.add(n)
                    frontier.append((n, path + [n]))

        return best_envelope_path, water_volume

# --- Main Benchmarking Logic ---
def run_benchmark():
    """Runs the full pipeline in-memory for N iterations and reports statistics."""
    N_RUNS = 100
    MAZE_SIZE = 65
    DELTA_X = 16

    # --- Statistics ---
    success_count = 0
    total_steps = 0
    total_bursts = 0
    total_time = 0.0

    print("="*50)
    print(f"Starting Robustness Test: {N_RUNS} iterations...")
    print("="*50)

    for i in range(N_RUNS):
        start_time = time.time()
        try:
            # 1. Generate Maze
            maze_env = MazeEnvironment(width=MAZE_SIZE, height=MAZE_SIZE)
            maze = maze_env.maze

            # 2. Generate Field
            flux_field = ProbabilityFluxField(maze_matrix=maze, delta_x=DELTA_X)
            flux_field.generate_field()
            field = flux_field.macro_field

            # 3. Run Agent
            agent = Agent(maze, field)
            max_agent_steps = maze.size  # Generous step limit
            for _ in range(max_agent_steps):
                if agent.step():
                    break
            
            # 4. Record results
            if agent.current_pos == agent.end_pos:
                success_count += 1
            
            total_steps += len(agent.path)
            total_bursts += len(agent.mcts_triggers)

        except Exception as e:
            print(f"!!! Run {i+1}/{N_RUNS} failed with an unexpected error: {e}")
            # This run is counted as a failure, but we continue the benchmark
        
        end_time = time.time()
        run_time = end_time - start_time
        total_time += run_time
        print(f"Run {i+1}/{N_RUNS} completed in {run_time:.2f}s. Success: {agent.current_pos == agent.end_pos}")

        # IMPORTANT: Close all plot figures to prevent memory leaks and slowdowns
        plt.close('all')

    # --- Final Report ---
    print("\n" + "="*50)
    print("      ALGORITHM ROBUSTNESS REPORT")
    print("="*50)
    print(f"Total Mazes Tested: {N_RUNS}")
    print(f"Success Rate:       {success_count / N_RUNS * 100:.1f}%")
    print(f"Average Steps:      {total_steps / N_RUNS:.1f} (includes failed runs)")
    print(f"Average Wavefronts: {total_bursts / N_RUNS:.2f} bursts/run")
    print(f"Average Time/Run:   {total_time / N_RUNS:.2f}s")
    print("="*50)

if __name__ == '__main__':
    run_benchmark()