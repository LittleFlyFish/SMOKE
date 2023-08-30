import cv2
import os

# Specify the path to the main folder containing the video folders
main_folder = '/home/soe/Documents/kitti/TestVideos'

video_files = [f for f in os.listdir(main_folder) if f.endswith('.mp4')]

# Iterate through each video file
count = 0
for video_file in video_files:
    count = count + 1
    # Extract the video file name without the extension
    video_name = os.path.splitext(video_file)[0]
    video_path = os.path.join(main_folder, video_file)

    # Create a folder to save the extracted frames
    frames_folder = os.path.join(main_folder, str(count))
    os.makedirs(frames_folder, exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Read the frames and save them as PNG files
    frame_count = 0
    while True:
         # Read the next frame
         success, frame = video_capture.read()
         if not success:
             break

         # Save the frame as a PNG file
         frame_file = os.path.join(frames_folder, f"{str(frame_count).zfill(6)}.png")
         cv2.imwrite(frame_file, frame)
         frame_count += 1

    # Release the video capture object
    video_capture.release()