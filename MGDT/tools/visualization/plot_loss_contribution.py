import numpy as np
import matplotlib.pyplot as plt

# 读取文件并提取数据
file_path = "/workspace/animax/MCL/tools/visualization/px_loss_contribution_iter_25601.txt"  # 请替换为实际的文件路径

ranges = []
negative_losses = []
positive_losses = []

# 读取txt文件中的内容
with open(file_path, 'r') as f:
    lines = f.readlines()

# 处理每行数据
for line in lines[2:]:  # 跳过前两行标题
    if line.strip():
        parts = line.split()
        if len(parts) == 3:  # 确保有3列数据
            range_str, negative_loss, positive_loss = parts[0], parts[1], parts[2]
            if "Total" in range_str:  # 排除包含总损失的行
                continue
            try:
                # 解析数据
                # 只保留范围的起始部分，并转换为浮动数值
                range_value = float(range_str.split('~')[0])  # 只使用范围的开始部分
                ranges.append(range_value)
                negative_losses.append(float(negative_loss))
                positive_losses.append(float(positive_loss))
            except ValueError:
                # 如果无法转换为浮动数值，则跳过当前行
                print(f"Skipping line due to ValueError: {line}")

# 检查数据的长度是否一致
if len(ranges) == len(negative_losses) == len(positive_losses):
    # 绘制条形图
    plt.figure(figsize=(12, 6))

    bar_width = 0.35
    index = np.arange(len(ranges))

    # 绘制负损失和正损失的条形图
    plt.bar(index, negative_losses, bar_width, label='Negative Loss', color='blue')
    plt.bar(index + bar_width, positive_losses, bar_width, label='Positive Loss', color='red')

    # 添加标题和标签
    plt.title("Loss Contribution in Each Range", fontsize=14)
    plt.xlabel("PX", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)

    # 设置横坐标刻度，每隔10个index显示一个刻度
    tick_interval = 10  # 每10个数据点显示一个刻度
    tick_positions = np.arange(0, len(ranges), tick_interval)  # 计算刻度的位置
    tick_labels = [f"{ranges[i]:.1f}" for i in tick_positions]  # 生成刻度标签
    plt.xticks(tick_positions, tick_labels, rotation=90)  # 设置横坐标刻度和标签

    # 显示图例
    plt.legend()

    # 保存图像到文件
    plt.tight_layout()  # 防止标签重叠
    plt.savefig("/workspace/animax/MCL/tools/visualization/loss_range_contribution_iter_25601.png", dpi=300)
    plt.show()
else:
    print("Data length mismatch. Please check the input file.")
