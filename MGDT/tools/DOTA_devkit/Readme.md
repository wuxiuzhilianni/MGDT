Split_Sparse_Image_and_Label:  对于每张图像，统计各个类别总数，随机选择其中的 X% 且至少为1，并且对数据集进行拆分
Split_Sparse_Image_Only:    仅拆分图像
change_file_name:   将 label 文件夹下所有 P0000__1.0__0___0.txt 换成 P0000__1024__0___0.txt

split_labeled_and_unlabeled:    将 image 和 label 分为 labeled 和 unlabeled
split_sparse_dota_method1:  在 DOTA 数据集分割之前，对于每个类别，统计总数，随机选取其中的 X%
split_sparse_dota_method2:  在 DOTA 数据集分割之前，对于每张图像，统计各个类别总数，随机选择其中的 X% 且至少为1

