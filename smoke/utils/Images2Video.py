import cv2
import os

# Path to the folder containing the images
folder_path = '/home/soe/Documents/MyProjects/Polysurance/kitti/Results/KITTI_3D_testing_Pred3D/'

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Sort the files based on their names
files.sort()

# Get the first image file to extract its dimensions
first_image = cv2.imread(os.path.join(folder_path, files[0]))
height, width, _ = first_image.shape

# Define the output video path and settings
output_path = '/home/soe/Documents/MyProjects/Polysurance/kitti/Results/output_video.mp4'
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through each image file and write to the video
for file in files:
    image_path = os.path.join(folder_path, file)
    image = cv2.imread(image_path)
    video_writer.write(image)

# Release the video writer and display a success message
video_writer.release()
print(f"Video created: {output_path}")
