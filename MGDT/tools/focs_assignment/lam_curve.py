import numpy as np
import matplotlib.pyplot as plt

def superellipse(a, b, n, num_points=1000):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = a * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** (2/n)
    y = b * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** (2/n)
    return x, y

# 参数设置
a, b = 2, 2  # 半轴长度
n_values = [2, 3, 4, 10]  # 不同n值
colors = ['blue', 'green', 'red', 'purple']  # 每条曲线的颜色
labels = ['Ellipse (n=2)', 'Rounded Rectangle (n=5)', 
          'Near Rectangle (n=10)', 'Rectangle (n=1000)']  # 图例标签

# 绘制图形
plt.figure(figsize=(10, 6))
for n, color, label in zip(n_values, colors, labels):
    x, y = superellipse(a, b, n)
    plt.plot(x, y, color=color, label=label, linewidth=2)

# 添加标题、标签和图例
plt.title("Superellipse Transformation (a=2, b=1)", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10, loc='upper right')

# 保存图像到文件
plt.savefig('/workspace/MCL/tools/focs_assignment/superellipse.png', dpi=300, bbox_inches='tight')
plt.show()