import os
files = os.listdir('/home/soe/Documents/MyProjects/Polysurance/SMOKE/datasets/kitti/testing/image_2')
files.sort()

save_txt = open('/home/soe/Documents/MyProjects/Polysurance/SMOKE/datasets/kitti/testing/ImageSets/test.txt', 'w')

for file in files:
    print('file:', file)
    file_name = file.split('.')[0]
    save_txt.write(file_name)
    save_txt.write('\n')
