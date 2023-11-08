import shutil

# Path to the file to be copied
source_file =  '/home/soe/Documents/MyProjects/Polysurance/kitti/testing/calib1382X512.txt' # '/smoke/utils/2011_9_26.txt'

# Path to the destination folder
destination_folder = '/home/soe/Documents/MyProjects/Polysurance/kitti/testing/calib/'

# Number of files to be generated
num_files = 209 # The number of images in image_2 folder

# Loop to generate and copy files
for i in range(num_files):
    # Generate the new file name
    new_file_name = str(i).zfill(6) + ".txt"

    # Path to the new file
    new_file_path = f"{destination_folder}/{new_file_name}"

    # Copy the source file to the new file path
    shutil.copy2(source_file, new_file_path)