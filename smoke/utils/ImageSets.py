import os
files = os.listdir('./datasets/kitti/training/image_2')
files.sort()

save_txt = open('./datasets/kitti/training/ImageSets/trainval.txt', 'w')

for file in files:
    print('file:', file)
    file_name = file.split('.')[0]
    save_txt.write(file_name)
    save_txt.write('\n')
