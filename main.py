import argparse
import subprocess
import os
import sys

def run_pipeline(generate_new=False):
    # 1. 如果需要新迷宫，运行生成器
    if generate_new or not os.path.exists('maze_data.npy'):
        print(">>> 正在生成全新物理迷宫...")
        subprocess.run([sys.executable, 'maze_env.py'], check=True)
    else:
        print(">>> 使用现有迷宫数据。")

    # 2. 运行宏观场计算 (Flux Field)
    print(">>> 正在计算宏观概率引力场...")
    subprocess.run([sys.executable, 'flux_field.py'], check=True)

    # 3. 运行寻路智能体 (Agent)
    print(">>> 智能体入场寻路...")
    subprocess.run([sys.executable, 'mcts_agent.py'], check=True)
    
    print(">>> 全流程运行完毕！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="迷宫物理模型主控程序")
    parser.add_argument('--new', action='store_true', help="强制生成新迷宫进行鲁棒性测试")
    args = parser.parse_args()
    
    run_pipeline(generate_new=args.new)