import numpy as np
import cv2

# 文件路径
stat_file = '/workspace/animax/MCL/tools/visualization/px_distribution_iter_16001.txt'

# 读取 px 值
def read_px_values(file_path):
    px_values = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                px_values.append(float(line.strip()))  # 转换为浮点数
            except ValueError:
                continue  # 跳过无法解析的行
    return np.array(px_values)

# 绘制直方图 (仅显示 0~0.1 范围)
def draw_histogram(px_values, bins=50, range_min=0, range_max=0.1, output_file='/workspace/animax/MCL/tools/visualization/px_distribution_16001.png'):
    # 过滤 px 值，仅保留范围内的值
    px_values = px_values[(px_values >= range_min) & (px_values <= range_max)]
    
    # 计算直方图
    hist, bin_edges = np.histogram(px_values, bins=bins, range=(range_min, range_max))
    
    # 创建一个空白图像
    width, height = 800, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白底

    # 绘制坐标轴
    margin = 50
    cv2.line(img, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 2)  # x轴
    cv2.line(img, (margin, margin), (margin, height - margin), (0, 0, 0), 2)  # y轴

    # 绘制直方图
    max_count = np.max(hist)
    bin_width = (width - 2 * margin) / bins
    for i, count in enumerate(hist):
        x1 = int(margin + i * bin_width)
        x2 = int(margin + (i + 1) * bin_width)
        y1 = height - margin
        y2 = int(height - margin - (count / max_count) * (height - 2 * margin))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)  # 红色柱子

    # 添加刻度和标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    for i in range(0, bins + 1, 10):
        x = int(margin + i * bin_width)
        y = height - margin + 15
        cv2.putText(img, f'{range_min + i * (range_max - range_min) / bins:.3f}', (x - 15, y), font, font_scale, (0, 0, 0), 1)

    for i in range(0, int(max_count) + 1, max(1, int(max_count / 10))):
        x = margin - 40
        y = int(height - margin - (i / max_count) * (height - 2 * margin))
        cv2.putText(img, str(i), (x, y + 5), font, font_scale, (0, 0, 0), 1)

    # 保存图像
    cv2.imwrite(output_file, img)
    print(f"Histogram saved as {output_file}")

# 主程序
px_values = read_px_values(stat_file)
draw_histogram(px_values, range_min=0, range_max=0.5)
