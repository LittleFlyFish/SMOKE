## how to change the evaluate_object_3d_offline.cpp the file

rm smoke/data/datasets/evaluation/kitti/kitti_eval/evaluate_object_3d_offline
mv evaluate_object_3d_offline.cpp smoke/data/datasets/evaluation/kitti/kitti_eval/

# How to test the model.pth and plot the video demos
python tools/plain_train_net.py --eval-only --config-file "configs/smoke_gn_vector.yaml"
cp -r tools/logs/inference/kitti_test/data datasets/kitti/pred
python visualization.py
python ShortZeros.py
python Images2Video.py

###########################################################################################
## Important changes:
## check_point.py line 49, define if the training need save in pth
## plain_train_net.py line 86, define which ckpt is loaded
## config/defaults.py line 157, define the threshold
## config/defaults.py line 76, define the Model is DLA or Mobile


#### The problem could be the Image.Open, most image has problem in annotations or image files. need to check this.

# Most config files are under /smoke/config/defaults.py


# to change the backbone to be Mobile Net the fellowing should be down:
#1. /smoke/configure/defaults.py line 76 _C.MODEL.BACKBONE.CONV_BODY = "MobileNetV2" #"DLA-34-DCN"
#2. make sure the tools/logs file belongs to the right BACKBONE

# How to solve the problem of libjsoncpp.so.19 not found?
#  ln -s /usr/lib64/libjsoncpp.so.19 /home/soe/anaconda3/envs/kitti_vis/lib/libjsoncpp.so.19
#  ln -s /usr/lib64/libjsoncpp.so.19 /home/soe/anaconda3/envs/pointcloud/lib/libjsoncpp.so.19


# export PATH=/home/soe/Downloads/QTInstall/Tools/QtCreator/lib/Qt/bin:$PATH

#cp /home/soe/anaconda3/envs/pointcloud/lib/libtbb.so.2  /home/soe/anaconda3/envs/kitti_vis/lib
#export LD_LIBRARY_PATH=/home/soe/anaconda3/envs/kitti_vis/lib:$LD_LIBRARY_PATH