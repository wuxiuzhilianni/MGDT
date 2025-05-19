```shell

python MCL/tools/analyze_logs/analyze_logs_stmodel.py plot_curve \
    xxx.log.json \
    --keys xxx \
    --title xxx \
    --legend "xxx" \
    --out xxx.png

# ST Model mAP
python /workspace/animax/MCL/tools/analyze_logs/analyze_logs_stmodel.py plot_curve \
    /workspace/animax/MCL/tools/analyze_logs/dense_teacher_analyze/dense_teacher_fcos_percent5_with_sparse_focal_loss.json\
    --keys teacher.mAP student.mAP \
    --title "Training Curve mAP" \
    --legend "teacher.mAP" "student.mAP" \
    --out /workspace/animax/MCL/tools/analyze_logs/dense_teacher_analyze/dense_teacher_fcos_percent5_with_sparse_focal_loss_mAP.png

# ST Model Loss
python /workspace/animax/MCL/tools/analyze_logs/analyze_logs_stmodel.py plot_curve \
    /workspace/animax/MCL/tools/analyze_logs/unbiased_teacher_analyze/20241030_163803.log.json \
    --keys loss_cls_sup loss_bbox_sup loss_centerness_sup loss_cls_unsup loss_bbox_unsup loss_centerness_unsup loss \
    --title "Training Curve" \
    --legend "Loss Cls Sup" "Loss BBox Sup" "Loss Centerness Sup" "Loss Cls Unsup" "Loss BBox Unsup" "Loss Centerness Unsup" "Total Loss" \
    --out /workspace/animax/MCL/tools/analyze_logs/loss.png

# Base Model mAP
python /workspace/animax/MCL/tools/analyze_logs/analyze_logs_basemodel.py \
    /workspace/animax/MCL/tools/analyze_logs/redet_base_analyze/redet_percent5_trainval_sfl_3x.json \
    --metric mAP \
    --title "mAP over Epochs" \
    --xlabel "Epoch" \
    --ylabel "Mean Average Precision" \
    --legend "mAP" \
    --out /workspace/animax/MCL/tools/analyze_logs/mAP.png

```
