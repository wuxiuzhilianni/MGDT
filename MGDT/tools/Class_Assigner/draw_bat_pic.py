import cv2
import numpy as np

# 类别及对应数据
categories = [
    "plane", "baseball-diamond", "bridge", "ground-track-field",
    "small-vehicle", "large-vehicle", "ship", "tennis-court",
    "basketball-court", "storage-tank", "soccer-ball-field",
    "roundabout", "harbor", "swimming-pool", "helicopter"
]
counts = [
    112192402, 44783214, 619844425, 12976181, 87773316, 27592813, 
    176801826, 20409588, 6897853, 83466614, 114294767, 10036747, 
    36483650, 15731532, 528780
]
mean_probs = [
    0.005566, 0.003911, 0.002444, 0.008423, 0.004959, 0.010681, 
    0.004444, 0.004184, 0.006222, 0.002672, 0.002640, 0.008272, 
    0.008245, 0.005577, 0.024506
]

# 数据归一化
max_count = max(counts)
max_mean_prob = max(mean_probs)
normalized_counts = [count / max_count for count in counts]
normalized_probs = [prob / max_mean_prob for prob in mean_probs]

# 创建画布
canvas_width = 1200
canvas_height = 800
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# 柱状图参数
bar_width = 20
space_between_bars = 40
bar_color_counts = (135, 206, 250)  # Sky blue
bar_color_probs = (255, 165, 0)  # Orange
origin_x = 100
origin_y = canvas_height - 100

# 绘制 count 百分比柱状图
for i, value in enumerate(normalized_counts):
    x1 = origin_x + i * (bar_width + space_between_bars)
    y1 = origin_y
    x2 = x1 + bar_width
    y2 = origin_y - int(value * 600)  # 映射到高度 600
    cv2.rectangle(canvas, (x1, y1), (x2, y2), bar_color_counts, -1)
    cv2.putText(canvas, f"{categories[i]}", (x1 - 10, origin_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

# 绘制 mean_prob 柱状图
for i, value in enumerate(normalized_probs):
    x1 = origin_x + i * (bar_width + space_between_bars) + bar_width + 10
    y1 = origin_y
    x2 = x1 + bar_width
    y2 = origin_y - int(value * 600)  # 映射到高度 600
    cv2.rectangle(canvas, (x1, y1), (x2, y2), bar_color_probs, -1)

# 添加标题和图例
cv2.putText(canvas, "Count Percentage (Orange)", (800, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color_counts, 1, cv2.LINE_AA)
cv2.putText(canvas, "Mean Probability (Sky Blue)", (800, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color_probs, 1, cv2.LINE_AA)
cv2.putText(canvas, "Category Distribution", (canvas_width // 3, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

# 保存图像
output_path = "/workspace/animax/MCL/tools/Class_Assigner/bar_chart.png"
cv2.imwrite(output_path, canvas)
print(f"Bar chart saved as {output_path}")
