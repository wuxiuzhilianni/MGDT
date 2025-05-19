# 类别和对应的 count 和 mean_prob 数据
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
gt_counts = [
    711, 898, 4239, 1275, 2281, 1111, 502, 3113, 610, 468, 375, 282,
    777, 468, 123
]

# 总注释数量
total_gt_count = 17233

# 计算每个类别的 count 百分比
total_count = sum(counts)
count_percentages = [count / total_count * 100 for count in counts]

# 计算每个类别的 mean_prob 百分比
max_mean_prob = max(mean_probs)
mean_prob_percentages = [mean_prob / max_mean_prob * 100 for mean_prob in mean_probs]

# 计算每个类别的 gt 数量百分比
gt_percentages = [gt_count / total_gt_count * 100 for gt_count in gt_counts]

# 输出结果
print(f"{'Category':<20} {'Count':<15} {'Count %':<10} {'Mean Prob':<15} {'Mean Prob %':<15} {'GT Count':<15} {'GT Count %':<10}")
print("-" * 90)

for i, category in enumerate(categories):
    print(f"{category:<20} {counts[i]:<15} {count_percentages[i]:<10.2f} {mean_probs[i]:<15.6f} {mean_prob_percentages[i]:<15.2f} {gt_counts[i]:<15} {gt_percentages[i]:<10.2f}")

"""
(mcl) root@7fb8c9563f98:/workspace# python /workspace/animax/MCL/tools/Class_Assigner/calculate_percent.py
Category             Count           Count %    Mean Prob       Mean Prob %     GT Count        GT Count %
------------------------------------------------------------------------------------------
plane                112192402       8.19       0.005566        22.71           711             4.13      
baseball-diamond     44783214        3.27       0.003911        15.96           898             5.21      
bridge               619844425       45.25      0.002444        9.97            4239            24.60     
ground-track-field   12976181        0.95       0.008423        34.37           1275            7.40      
small-vehicle        87773316        6.41       0.004959        20.24           2281            13.24     
large-vehicle        27592813        2.01       0.010681        43.59           1111            6.45      
ship                 176801826       12.91      0.004444        18.13           502             2.91      
tennis-court         20409588        1.49       0.004184        17.07           3113            18.06     
basketball-court     6897853         0.50       0.006222        25.39           610             3.54      
storage-tank         83466614        6.09       0.002672        10.90           468             2.72      
soccer-ball-field    114294767       8.34       0.002640        10.77           375             2.18      
roundabout           10036747        0.73       0.008272        33.75           282             1.64      
harbor               36483650        2.66       0.008245        33.64           777             4.51      
swimming-pool        15731532        1.15       0.005577        22.76           468             2.72      
helicopter           528780          0.04       0.024506        100.00          123             0.71  
"""