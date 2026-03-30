import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_interactive_3d():
    print("Loading pre-calculated physics data...")
    try:
        maze = np.load('maze_data.npy')
        macro_field = np.load('macro_field_data.npy')
    except FileNotFoundError:
        print("Error: 找不到数据文件，请先运行 flux_field.py")
        return

    rows, cols = maze.shape

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Interactive Probability Flux Ridge (Viewer)', fontsize=16)

    # 左侧 2D
    ax1 = fig.add_subplot(121)
    ax1.imshow(maze, cmap='gray', interpolation='none')
    im = ax1.imshow(macro_field, cmap='magma', alpha=0.7, interpolation='bilinear')
    ax1.plot(0, 0, marker='*', color='cyan', markersize=15, label='Start (0,0)')
    ax1.plot(cols - 1, rows - 1, marker='*', color='red', markersize=15, label='End')
    ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 1.02), borderaxespad=0., ncol=2)
    ax1.set_title('2D: Physical vs Macro')

    # 右侧 3D
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(range(cols), range(rows))
    ax2.plot_surface(X, Y, macro_field, cmap='magma', edgecolor='none', alpha=0.9)
    ax2.scatter(0, 0, macro_field[0, 0] + 0.1, color='cyan', s=100, marker='*')
    ax2.scatter(cols - 1, rows - 1, macro_field[rows - 1, cols - 1] + 0.1, color='red', s=100, marker='*')
    ax2.invert_yaxis()
    ax2.view_init(elev=50, azim=-60)
    ax2.set_title('3D: Interactive Ridge')

    print("Opening 3D viewer... You can now drag and rotate!")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    show_interactive_3d()