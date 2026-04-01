import numpy as np
import time
import matplotlib.pyplot as plt
import sys

# 1. Import all required classes from our project modules
try:
    from maze_env import MazeEnvironment
    from flux_field import ProbabilityFluxField
    from mcts_agent import Agent
except ImportError as e:
    print(f"Error: Could not import necessary modules. {e}")
    print("Please ensure maze_env.py, flux_field.py, and mcts_agent.py are in the same directory.")
    sys.exit(1)

def run_benchmark():
    """
    Runs the full, in-memory pipeline with the V4.0 agent on braided mazes,
    collects performance statistics, and saves computational cost data.
    """
    N_RUNS = 100
    MAZE_SIZE = 65
    BRAID_P = 0.08
    DELTA_X = 16

    # --- Statistics ---
    success_count = 0
    total_steps = 0
    total_bursts = 0
    total_time = 0.0
    all_non_trivial_delta_p = []

    print("="*60)
    print(f"Starting V4.0 Agent Robustness Test ({N_RUNS} runs on Braided Mazes)...")
    print("="*60)

    start_time_total = time.time()
    for i in range(N_RUNS):
        try:
            # 1. Generate a new braided maze in memory
            maze_env = MazeEnvironment(width=MAZE_SIZE, height=MAZE_SIZE, braid_p=BRAID_P)
            maze = maze_env.maze

            # 2. Generate the corresponding macro-field in memory
            flux_field = ProbabilityFluxField(maze_matrix=maze, delta_x=DELTA_X)
            flux_field.generate_field()
            field = flux_field.macro_field

            # 3. Initialize and run the V4.0 Agent
            agent = Agent(maze, field)
            # Use a generous step limit to detect true deadlocks
            max_agent_steps = maze.size * 2
            for _ in range(max_agent_steps):
                if agent.step():
                    break
            
            # --- 4. Data Collection ---
            is_success = (agent.current_pos == agent.end_pos)
            if is_success:
                success_count += 1
            
            # Collect all bursts (delta_p > 1) for later analysis
            bursts = [p for p in agent.delta_p_list if p > 1]
            if bursts:
                all_non_trivial_delta_p.extend(bursts)
            
            # Update aggregate stats for final report
            total_steps += len(agent.path)
            total_bursts += len(agent.mcts_triggers)

        except Exception as e:
            print(f"\n!!! Run {i+1}/{N_RUNS} failed with an unexpected error: {e}")
            # This run is counted as a failure, but we continue the benchmark
        
        # --- Progress Bar and Live Stats ---
        progress = (i + 1) / N_RUNS
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        current_rate = success_count / (i + 1) * 100
        sys.stdout.write(f'\rProgress: |{bar}| {i+1}/{N_RUNS} | Current Success Rate: {current_rate:.1f}%')
        sys.stdout.flush()

        # IMPORTANT: Ensure no plot windows are held in memory
        plt.close('all')

    total_time = time.time() - start_time_total
    
    # --- Final Report ---
    print("\n\n" + "="*60)
    print("      V4.0 AGENT ROBUSTNESS REPORT (BRAIDED MAZES)")
    print("="*60)
    print(f"Total Mazes Tested: {N_RUNS}")
    print(f"Success Rate:       {success_count / N_RUNS * 100:.1f}%")
    print(f"Average Steps:      {total_steps / N_RUNS:.1f} (includes failed runs)")
    print(f"Average Wavefronts: {total_bursts / N_RUNS:.2f} bursts/run")
    print(f"Average Time/Run:   {total_time / N_RUNS:.2f}s")
    print("="*60)

    # --- Save collected data for physics analysis ---
    if all_non_trivial_delta_p:
        delta_p_array = np.array(all_non_trivial_delta_p)
        np.save('all_delta_p.npy', delta_p_array)
        print(f"\nSuccessfully saved {len(delta_p_array)} non-trivial Δp bursts to 'all_delta_p.npy'.")
        print("Ready for 'physics_analyzer.py'.")
    else:
        print("\nNo non-trivial Δp bursts were recorded.")

if __name__ == '__main__':
    run_benchmark()