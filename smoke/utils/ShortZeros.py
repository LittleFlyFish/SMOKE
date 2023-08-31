# use this file to change the 000000000.png into 000000.png

import os
folder_path = '/home/soe/Documents/kitti/testing/image_2'

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        file_number = int(filename.split(".")[0])
        new_filename = str(file_number).zfill(6) + ".png"
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))