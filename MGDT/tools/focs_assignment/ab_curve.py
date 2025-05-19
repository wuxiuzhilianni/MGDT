import torch
import matplotlib.pyplot as plt
import numpy as np

# === 1. 模拟图像宽度 & 超参数 ===
img_w = 1024
k = 0.05  # 控制 sigmoid 斜率（越大越陡）

# === 2. 定义尺度感知采样分段 ===
piecewise_weights = [1/2, 1/3, 1/5, 1/7, 1/16, 1/32]
ratio_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4]
thresholds = [r * img_w for r in ratio_thresholds]

# === 3. 宽度范围（模拟目标宽度）===
w_vals = torch.linspace(0, img_w * 0.5, steps=1000)

# === 4. 平滑分段函数（与主代码一致） ===
def smooth_piecewise(tensor, thresholds, weights, k):
    assert len(weights) == len(thresholds) + 1
    sigmoids = [1 / (1 + torch.exp(-k * (tensor - t))) for t in thresholds]

    result = weights[0] * (1 - sigmoids[0])
    for i in range(1, len(thresholds)):
        result += weights[i] * (sigmoids[i-1] - sigmoids[i])
    result += weights[-1] * sigmoids[-1]
    return tensor * result

# === 5. 计算 a(w) 曲线 ===
a_vals = smooth_piecewise(w_vals, thresholds, piecewise_weights, k)

# === 6. 绘图 ===
w_vals_np = w_vals.numpy()
a_vals_np = a_vals.numpy()

plt.figure(figsize=(8, 5))
plt.plot(w_vals_np, a_vals_np, label=r'$a(w)$', color='blue', linewidth=2)

# 标注阈值位置
for i, t in enumerate(thresholds):
    plt.axvline(t, color='gray', linestyle='--', linewidth=1, label=f'thresh{i+1}' if i == 0 else None)

plt.title('Scale-aware Adaptive Sampling Function', fontsize=14)
plt.xlabel('Object Width (w)', fontsize=12)
plt.ylabel('Adaptive Radius a', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('/workspace/MCL/tools/focs_assignment/adaptive_sampling_radius.png', dpi=300)
