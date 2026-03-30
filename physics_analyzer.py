import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import random
from collections import deque
from itertools import combinations
import time
from scipy.optimize import curve_fit

# --- All necessary classes are bundled here for a self-contained script ---

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
        if self.current_pos == self.end_pos: return True
        self.trauma_memory[self.current_pos] += 0.05
        self.visited_freq[self.current_pos] = self.visited_freq.get(self.current_pos, 0) + 1
        if self.visited_freq[self.current_pos] > 20: return True
        if self.forced_march_queue:
            self.last_pos, self.current_pos = self.current_pos, self.forced_march_queue.pop(0)
            self.path.append(self.current_pos); self.delta_p_list.append(1); return False
        neighbors = self.get_physical_neighbors(self.current_pos)
        forward_neighbors =[n for n in neighbors if n != self.last_pos]
        if not forward_neighbors and neighbors: forward_neighbors = neighbors
        best_n, best_score = None, -float('inf')
        for n in forward_neighbors:
            dist = abs(n[0] - self.end_pos[0]) + abs(n[1] - self.end_pos[1])
            score = self.macro_field[n] - (dist / (self.rows + self.cols)) * 0.1 - self.trauma_memory[n]
            if score > best_score: best_score, best_n = score, n
        curr_dist = abs(self.current_pos[0] - self.end_pos[0]) + abs(self.current_pos[1] - self.end_pos[1])
        curr_score = self.macro_field[self.current_pos] - (curr_dist / (self.rows + self.cols)) * 0.1 - self.trauma_memory[self.current_pos]
        if not forward_neighbors or best_score <= curr_score:
            self.mcts_triggers.append(self.current_pos); self._apply_local_mark(self.current_pos)
            escape_path, delta_p = self._wavefront_search(curr_dist)
            self.delta_p_list[-1] += delta_p
            if not escape_path: return True
            self.forced_march_queue = escape_path; return self.step() 
        self.last_pos, self.current_pos = self.current_pos, best_n
        self.path.append(self.current_pos); self.delta_p_list.append(1); return False

    def _wavefront_search(self, start_dist):
        frontier = deque([(self.current_pos, [])])
        if self.last_pos: frontier.append((self.last_pos, [self.last_pos]))
        flooded = {self.current_pos, self.last_pos}
        water_volume, best_envelope_path, min_dist_to_goal = 0, [], start_dist
        start_p = self.macro_field[self.current_pos]
        while frontier:
            water_volume += 1
            if water_volume > 4000: break
            curr, path = frontier.popleft()
            if curr == self.end_pos: return path, water_volume
            dist = abs(curr[0] - self.end_pos[0]) + abs(curr[1] - self.end_pos[1])
            if dist < min_dist_to_goal: min_dist_to_goal, best_envelope_path = dist, path
            if start_dist > 25:
                jump_threshold = max(5, int(start_dist * 0.382))
                if self.macro_field[curr] > start_p + 0.05 or dist <= start_dist - jump_threshold:
                    return path, water_volume
            for n in self.get_physical_neighbors(curr):
                if n not in flooded: flooded.add(n); frontier.append((n, path + [n]))
        return best_envelope_path, water_volume

# --- Physics Analysis Logic ---

def power_law_func(x, a, c):
    """Power law function for fitting: f(x) = c * x^a"""
    return c * np.power(x, a)

def run_physics_analysis():
    """Runs a large number of simulations to collect data for power-law analysis."""
    N_RUNS = 200
    all_bursts = []

    print("="*60)
    print(f"Starting Deep Physics Analysis: {N_RUNS} iterations...")
    print("This will take a few minutes. No GUI will be shown.")
    print("="*60)

    start_time_total = time.time()
    for i in range(N_RUNS):
        maze = MazeEnvironment(width=65, height=65).maze
        flux_field = ProbabilityFluxField(maze_matrix=maze, delta_x=16)
        flux_field.generate_field()
        field = flux_field.macro_field
        agent = Agent(maze, field)
        for _ in range(maze.size):
            if agent.step(): break
        
        bursts = [p for p in agent.delta_p_list if p > 1]
        if bursts:
            all_bursts.extend(bursts)
        
        if (i + 1) % 10 == 0:
            print(f"Run {i+1}/{N_RUNS} completed...")
        plt.close('all')

    print(f"\nData collection finished in {time.time() - start_time_total:.2f}s.")
    print(f"Found {len(all_bursts)} non-trivial computational bursts.")

    if not all_bursts:
        print("No wavefront bursts were triggered. Cannot perform analysis.")
        return

    bursts_array = np.array(all_bursts)
    bins = np.logspace(np.log10(min(bursts_array)), np.log10(max(bursts_array)), 20)
    hist, bin_edges = np.histogram(bursts_array, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    non_zero_mask = hist > 0
    bin_centers_fit = bin_centers[non_zero_mask]
    hist_fit = hist[non_zero_mask]

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Physics Analysis of Computational Cost (Δp)', fontsize=16)

    # 1. Right Plot (as it was originally): Log-Log Power Law Analysis
    ax1 = fig.add_subplot(122)
    ax1.loglog(bin_centers_fit, hist_fit, 'o', color='cyan', markersize=8, label='Data (Log-Binned PDF)')
    alpha = -1.5 
    try:
        popt, _ = curve_fit(power_law_func, bin_centers_fit, hist_fit, p0=[alpha, np.max(hist_fit)])
        alpha = popt[0]
        ax1.loglog(bin_centers_fit, power_law_func(bin_centers_fit, *popt), 'r-', lw=2,
                   label=f'Power Law Fit (α = {alpha:.2f})')
    except RuntimeError:
        print("Curve fit failed. The data may not follow a power law.")
        alpha = None
    ax1.set_title('Power Law Distribution of Δp')
    ax1.set_xlabel('Computational Burst Size (Log Δp)')
    ax1.set_ylabel('Probability Density (Log PDF)')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # 2. Left Plot (as it was originally): Lorenz Curve (Pareto Analysis)
    ax2 = fig.add_subplot(121)
    sorted_bursts = np.sort(bursts_array)
    cum_bursts = np.cumsum(sorted_bursts)
    x_lorenz = np.linspace(0., 1., len(cum_bursts))
    y_lorenz = cum_bursts / cum_bursts[-1]
    ax2.plot(x_lorenz, y_lorenz, color='magenta', lw=2, label='Lorenz Curve')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, label='Line of Equality')
    ax2.fill_between(x_lorenz, y_lorenz, x_lorenz, color='magenta', alpha=0.2)

    # --- CORRECTED PARETO ANALYSIS ---
    pareto_x_consumes_80_percent = None
    cost_consumed_by_top_20_percent = None
    try:
        # What % of top bursts consume 80% of cost?
        idx_for_80_percent_cost = np.searchsorted(y_lorenz, 0.2)
        events_at_20_cost = x_lorenz[idx_for_80_percent_cost]
        pareto_x_consumes_80_percent = (1 - events_at_20_cost) * 100
        
        # What % of cost is consumed by top 20% of bursts?
        idx_for_20_percent_bursts = np.searchsorted(x_lorenz, 0.8)
        cost_at_80_bursts = y_lorenz[idx_for_20_percent_bursts]
        cost_consumed_by_top_20_percent = (1 - cost_at_80_bursts) * 100

        # Add annotation to the plot
        pareto_text = (f'Top 20% of most complex bursts\n'
                       f'consume {cost_consumed_by_top_20_percent:.1f}% of total Δp')
        ax2.annotate(pareto_text, xy=(0.8, cost_at_80_bursts), xytext=(0.35, 0.55),
                     arrowprops=dict(facecolor='white', shrink=0.05),
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="k", lw=1, alpha=0.8))
    except (ValueError, IndexError):
        print("Could not perform Pareto analysis.")

    ax2.set_title('Pareto Analysis of Computational Cost (Lorenz Curve)')
    ax2.set_xlabel('Cumulative % of Bursts (Sorted from least to most costly)')
    ax2.set_ylabel('Cumulative % of Total Δp Cost')
    ax2.legend(loc='upper left')
    ax2.grid(True, ls="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("physics_analysis_report.png", dpi=300)
    plt.show()

    # --- Final Conclusion Output ---
    print("\n" + "="*60)
    print("      COMPUTATIONAL PHYSICS REPORT")
    print("="*60)
    if alpha is not None:
        print(f"Power Law Exponent (α): {alpha:.3f}")
        print("  Interpretation: A value between -1 and -2 is typical for many\n"
              "  self-organized critical systems, indicating frequent small events\n"
              "  and increasingly rare, catastrophically large ones.")
    if pareto_x_consumes_80_percent is not None and cost_consumed_by_top_20_percent is not None:
        print(f"\nPareto Principle Analysis:")
        print(f"  - The top {pareto_x_consumes_80_percent:.1f}% of complex bursts "
              f"consumed 80% of the total computational energy (Δp).")
        print(f"  - Conversely, the top 20% of complex bursts "
              f"consumed {cost_consumed_by_top_20_percent:.1f}% of the total energy.")
    print("="*60)

if __name__ == '__main__':
    run_physics_analysis()