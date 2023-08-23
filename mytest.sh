## how to change the evaluate_object_3d_offline.cpp the file

rm smoke/data/datasets/evaluation/kitti/kitti_eval/evaluate_object_3d_offline
mv evaluate_object_3d_offline.cpp smoke/data/datasets/evaluation/kitti/kitti_eval/
python tools/plain_train_net.py --eval-only --config-file "configs/smoke_gn_vector.yaml"
