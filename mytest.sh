## how to change the evaluate_object_3d_offline.cpp the file

rm smoke/data/datasets/evaluation/kitti/kitti_eval/evaluate_object_3d_offline
mv evaluate_object_3d_offline.cpp smoke/data/datasets/evaluation/kitti/kitti_eval/
python tools/plain_train_net.py --eval-only --config-file "configs/smoke_gn_vector.yaml"

# Most config files are under /smoke/config/defaults.py


# to change the backbone to be Mobile Net the fellowing should be down:
#1. /smoke/configure/defaults.py line 76 _C.MODEL.BACKBONE.CONV_BODY = "MobileNetV2" #"DLA-34-DCN"
#2. make sure the tools/logs file belongs to the right BACKBONE