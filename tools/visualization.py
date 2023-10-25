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
            self.num_samples = 7481
        elif split == "testing":
            self.num_samples = 7518
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


def show_predictions_with_boxes(save_folder, img, objects_pred, calib, show2d=True, show3d=True, count=0):
    # draw 2D Pred
    if show2d:
        img1 = np.copy(img)
        type_list = ["Pedestrian", "Car", "Cyclist"]
        if objects_pred is not None:
            color = (255, 0, 0)
            for obj in objects_pred:
                if obj.type not in type_list:
                    continue
                cv2.rectangle(
                    img1,
                    (int(obj.xmin), int(obj.ymin)),
                    (int(obj.xmax), int(obj.ymax)),
                    color,
                    1,
                )
            startx = 165
            font = cv2.FONT_HERSHEY_SIMPLEX

            text_lables = [obj.type for obj in objects_pred if obj.type in type_list]
            text_lables.insert(0, "3D Pred:")
            for n in range(len(text_lables)):
                text_pos = (startx, 25 * (n + 1))
                cv2.putText(
                    img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA
                )

        # Specify the desired file path and extension
        save_path = save_folder + '2D/' + str(count).zfill(6) + ".jpg"
        cv2.imwrite(save_path, img1)
    if show3d:
        img2 = np.copy(img)
        for obj in objects_pred:
            box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
            if box3d_pts_2d is None:
                print("something wrong in the 3D box.")
                continue
            if obj.type == "Car":
                img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
            elif obj.type == "Pedestrian":
                img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(255, 255, 0))
            elif obj.type == "Cyclist":
                img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 255))
                # Specify the desired file path and extension
        save_path = save_folder + '3D/' + str(count).zfill(6) + ".jpg"
        cv2.imwrite(save_path, img2)
    return img1, img2

def show_predAndGT_with_boxes(save_folder,img, objects_pred, objects, calib, count=0):
    # This function will show the predicted 3D box in dash and the GroundTruth
    img1 = np.copy(img)
    for obj in objects_pred:
        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            print("something wrong in the 3D box.")
            continue
        if obj.type == "Car":
            img1 = utils.draw_projected_box3d_Dash(img1, box3d_pts_2d, color=(255, 0, 255))
        elif obj.type == "Pedestrian":
            img1 = utils.draw_projected_box3d_Dash(img1, box3d_pts_2d, color=(0, 0, 255))
        elif obj.type == "Cyclist":
            img1 = utils.draw_projected_box3d_Dash(img1, box3d_pts_2d, color=(255, 0, 0))
            # Specify the desired file path and extension
    for obj in objects:
        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            print("something wrong in the 3D box.")
            continue
        if obj.type == "Car":
            img1 = utils.draw_projected_box3d(img1, box3d_pts_2d, color=(0, 255, 0))
        elif obj.type == "Pedestrian":
            img1 = utils.draw_projected_box3d(img1, box3d_pts_2d, color=(255, 255, 0))
        elif obj.type == "Cyclist":
            img1 = utils.draw_projected_box3d(img1, box3d_pts_2d, color=(0, 255, 255))
            # Specify the desired file path and extension
    save_path = save_folder + '/' + str(count).zfill(6) + ".jpg"
    cv2.imwrite(save_path, img1)
    print(save_path)
    print(img1)
    return img1
def show_image_with_boxes(save_folder, img, objects, calib, count=0):
    img1 = np.copy(img)
    img2 = np.copy(img)  # for 3d bbox
    # TODO: change the color of boxes
    for obj in objects:
        if obj.type == "DontCare":
            continue
        if obj.type == "Car":
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (255, 255, 0),
                2,
            )
        if obj.type == "Pedestrian":
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (0, 255, 0),
                2,
            )
        if obj.type == "Cyclist":
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (0, 255, 255),
                2,
            )
        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            print("something wrong in the 3D box.")
            continue
        if obj.type == "Car":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
        elif obj.type == "Pedestrian":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(255, 255, 0))
        elif obj.type == "Cyclist":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 255))

    save_path = os.path.join(save_folder, str(count).zfill(6) + ".jpg")
    cv2.imwrite(save_path, img2)
    return img1, img2


if __name__ == "__main__":
    save_path =  '/media/data1/yanran/SMOKE/datasets/kitti/Results' #
    root_dir =  '/media/data1/yanran/SMOKE/datasets/kitti' # training and testing are under it

    set_label = ['training', 'testing']
    split_set = set_label[0]
    print(split_set)

    if split_set == 'testing':
        print('testing')
        dataset = kitti_object(root_dir, split='testing', args=None)
        for data_idx in range(len(dataset)):
            # load the information of 3D box from txt files
            # it will get from label_2 file for the ground truth 3D bounding box
            objects_pred = dataset.get_pred_objects(data_idx)
            if objects_pred == None:
                continue
            calib = dataset.get_calibration(data_idx)
            img = dataset.get_image(data_idx)
            print(data_idx)

            # extract each objects from the data
            n_obj = 0
            # for obj in objects_pred:
            #     if obj.type != "DontCare":
            #         print("=== {} object ===".format(n_obj + 1))
            #         obj.print_object()
            #         n_obj += 1
            # draw out the predicted text img data
            save_folder = save_path + "/KITTI_3D_" + split_set + "_Pred"
            if img is None:
                continue
            show_predictions_with_boxes(save_folder, img, objects_pred, calib, True, True, data_idx)
        if split_set == 'training':
            dataset = kitti_object(root_dir, split='training', args=None)
            print(len(dataset))
            for data_idx in range(len(dataset)):
                # load the information of 3D box from txt files
                objects = dataset.get_label_objects(
                    data_idx)  # it will get from label_2 file for the ground truth 3D bounding box
                objects_pred = dataset.get_pred_objects(data_idx)
                calib = dataset.get_calibration(data_idx)
                img = dataset.get_image(data_idx)

                # extract each objects from the data
                n_obj = 0
                # for obj in objects:
                #     if obj.type != "DontCare":
                #         print("=== {} object ===".format(n_obj + 1))
                #         obj.print_object()
                #         n_obj += 1
                # draw out the predicted text img data
                save_folder = save_path + "/KITTI_3D_" + split_set + "_PredAndGT"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                print(save_folder)
                show_predAndGT_with_boxes(save_folder, img, objects_pred, objects, calib, data_idx)

