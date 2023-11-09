# This file is using to draw the 3D Bounding Box on the image

from __future__ import print_function
import numpy as np
import cv2
import os, math
from scipy.optimize import leastsq
from PIL import Image
import sys
sys.path.append('/media/data1/yanran/SMOKE')
import smoke.utils.kitti_util as utils
from visualization import show_predAndGT_with_boxes
from rtree import index

class kitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split="training", args=None):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split = split
        print('The dataset root direction is:', root_dir)
        print('The split is:', split)
        self.split_dir = os.path.join(root_dir, split)

        if split == "training":
            self.num_samples = 4000
        elif split == "testing":
            self.num_samples = 4000
        else:
            print("Unknown split: %s" % (split))
            exit(-1)

        lidar_dir = "velodyne"
        depth_dir = "depth"
        pred_dir = "pred"
        if args is not None:
            lidar_dir = args.lidar
            depth_dir = args.depthdir
            pred_dir = args.preddir

        self.image_dir = os.path.join(self.split_dir, "image_2")
        self.label_dir = os.path.join(self.split_dir, "label_2")
        self.calib_dir = os.path.join(self.split_dir, "calib")

        self.depthpc_dir = os.path.join(self.split_dir, "depth_pc")
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.depth_dir = os.path.join(self.split_dir, depth_dir)
        self.pred_dir = os.path.join(self.split_dir, pred_dir)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.image_dir, "%06d.png" % (idx))
        return utils.load_image(img_filename)


    def get_calibration(self, idx):
        assert idx < self.num_samples
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert idx < self.num_samples and self.split == "training"
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        return utils.read_label(label_filename)

    def get_pred_objects(self, idx):
        assert idx < self.num_samples
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            return None

    def get_depth(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_image(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_pc(self, idx):
        assert idx < self.num_samples
        lidar_filename = os.path.join(self.depthpc_dir, "%06d.bin" % (idx))
        is_exist = os.path.exists(lidar_filename)
        if is_exist:
            return utils.load_velo_scan(lidar_filename), is_exist
        else:
            return None, is_exist
        # print(lidar_filename, is_exist)
        # return utils.load_velo_scan(lidar_filename), is_exist

    def get_top_down(self, idx):
        pass

    def isexist_pred_objects(self, idx):
        assert idx < self.num_samples and self.split == "training"
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        return os.path.exists(pred_filename)

    def isexist_depth(self, idx):
        assert idx < self.num_samples and self.split == "training"
        depth_filename = os.path.join(self.depth_dir, "%06d.txt" % (idx))
        return os.path.exists(depth_filename)

def resize_coordinates(corners_2d, original_size, new_size):
    """
    Adjusts the coordinates of a 2D bounding box after an image is resized.

    Parameters:
    corners_2d (numpy.ndarray): The original coordinates, as an array with shape (8, 2).
    original_size (tuple): The original image size (width, height).
    new_size (tuple): The new image size (width, height).

    Returns:
    numpy.ndarray: The adjusted coordinates, as an array with shape (8, 2).
    """

    # Calculate the ratios for width and height
    width_ratio = new_size[0] / original_size[0]
    height_ratio = new_size[1] / original_size[1]

    # Apply the ratios to the coordinates
    corners_2d[:, 0] *= width_ratio
    corners_2d[:, 1] *= height_ratio

    # Return the adjusted coordinates
    return corners_2d


def compute_bounding_rect(corners_2d):
    """计算2D边界框的最小封闭矩形"""
    xmin = np.min(corners_2d[0, :])
    xmax = np.max(corners_2d[0, :])
    ymin = np.min(corners_2d[1, :])
    ymax = np.max(corners_2d[1, :])
    return [xmin, ymin, xmax, ymax]

def compute_iou(box1, box2):
    """计算两个2D矩形的IOU"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersection = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)
    iou = intersection / float(area_box1 + area_box2 - intersection)
    return iou

def select_non_occluded_objects(objects, P):
    """选择不会互相遮挡的物体"""
    idx = index.Index()
    obj_corners = [0 for _ in objects]
    for i, obj in enumerate(objects):
        obj_corners_2d, _ = utils.compute_box_3d(obj, P)
        if obj_corners_2d is None:
            continue
        idx.insert(i, compute_bounding_rect(obj_corners_2d))
        obj_corners[i] = obj_corners_2d

    new_objects = []
    for i, obj in enumerate(objects):
        obj_corners_2d, _ = utils.compute_box_3d(obj, P)
        if obj_corners_2d is None:
            continue
        box1 = compute_bounding_rect(obj_corners_2d)
        overlapped_indices = list(idx.intersection(box1))
        overlapped_indices.remove(i)  # 移除自己
        for j in overlapped_indices:
            if obj_corners[j].all == 0:
                continue
            box2 = compute_bounding_rect(obj_corners[j])
            iou = compute_iou(box1, box2)
            if iou > 0.5:  # 如果IOU大于0.5，那么认为存在遮挡
                break
        else:  # 如果没有找到遮挡的物体，那么添加到新的列表中
            new_objects.append(obj)
    return new_objects

def SaveObjects(new_objects, save_path, data_index):
    # This function will save the generated new_obejcts into label_2.txt files
    filename = str(data_index).zfill(6) + '.txt'
    with open(os.path.join(save_path, filename), 'w') as f:
        for obj in new_objects:
            line = ' '.join([
                obj.type,
                str(obj.truncation),
                str(obj.occlusion),
                str(obj.alpha),
                str(obj.xmin),
                str(obj.ymin),
                str(obj.xmax),
                str(obj.ymax),
                str(obj.h),
                str(obj.w),
                str(obj.l),
                str(obj.t[0]),
                str(obj.t[1]),
                str(obj.t[2]),
                str(obj.ry),
            ])
            f.write(line + '\n')



if __name__ == "__main__":
    save_path =  '/home/soe/Documents/MyProjects/Polysurance/SMOKE/datasets/kitti/Results' # '/media/data1/yanran/SMOKE/datasets/kitti/Results' #
    root_dir =  '/home/soe/Documents/MyProjects/Polysurance/SMOKE/datasets/kitti' # '/media/data1/yanran/SMOKE/datasets/kitti' # training and testing are under it

    set_label = ['training', 'testing']
    split_set = set_label[0]
    print(split_set)
    total_objects = 0
    total_new_objects = 0

    if split_set == 'training':
        dataset = kitti_object(root_dir, split='training', args=None)
        save_label = root_dir + '/' + split_set + '/' + 'label_2_new'
        print(len(dataset))
        for data_idx in range(len(dataset)):
            # load the information of 3D box from txt files
            objects = dataset.get_label_objects(
                data_idx)  # it will get from label_2 file for the ground truth 3D bounding box
            calib = dataset.get_calibration(data_idx)
            img = dataset.get_image(data_idx)
            # if img is None:
            #     continue
            # if img.dtype != np.uint8:
            #     # 如果数据类型不是np.uint8，则将其转换为np.uint8
            #     img = img.astype(np.uint8)
            print(data_idx)
            new_objects = select_non_occluded_objects(objects, calib.P)
            SaveObjects(new_objects, save_path, data_idx)
            total_objects = total_objects + len(objects)
            total_new_objects = total_new_objects + len(new_objects)
            # print('The dataset before fliter and after filter:', (len(objects), len(new_objects)))
            save_folder = save_path + "/KITTI_3D_" + split_set + "_PredAndGT"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            # print(save_folder)
            objects_pred = objects # this is not useful
            # show_predAndGT_with_boxes(save_folder, img, objects_pred, new_objects, calib, data_idx)


    print('The total changes are:',(total_objects, total_new_objects))




