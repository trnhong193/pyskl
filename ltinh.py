import pickle


with open('/media/ivsr/data2/pyskl/demo/results/ntu_small_true.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)
'''
doc full file pkl
'''
# import numpy as np
# import pickle
# # Open your pickle file (replace "your_file.txt" with your actual filename)
# with open("/media/ivsr/data2/pyskl/demo/results/ntu_small.pkl", "rb") as f:
#     data = pickle.load(f)
# # Increase the threshold to display all elements
# np.set_printoptions(threshold=np.inf) #important
# # Print the keypoint array
# print("FRAME")
# print(data["annotations"][0]["total_frames"])
# print("KEYPOINT")
# print(data["annotations"][0]["keypoint"])
# print("KEYPOINT_SCORE")
# print(data["annotations"][0]["keypoint_score"])

"""
05/03/24 - train
ntu_rgb - lay ra 6 hd => ntu_small
008: sit down - 0
009: stand up - 1
024: kick sth - 2
026: jumping up(1 foot) - 3
027: jump up - 4
043: falling - 5
"""
# import os
# import shutil
# # /media/ivsr/newHDD1/nturgbd_rgb_s001/nturgbd_videos

# # Define the source directory where all the folders are located
# # source_dir = '/media/ivsr/newHDD1/nturgbd_rgb_s011/nturgb+d_rgb'
# # source_dir = '/media/ivsr/newHDD1/nturgbd_rgb_s001/nturgbd_videos'
# source_dir= '/media/ivsr/newHDD1/nturgb+d_rgb_s017'

# # Define the target directories where videos will be moved based on labels
# target_dirs = {
#     '008': '/media/ivsr/data2/pyskl/video_data/0',
#     '009': '/media/ivsr/data2/pyskl/video_data/1',
#     '024': '/media/ivsr/data2/pyskl/video_data/2',
#     '026': '/media/ivsr/data2/pyskl/video_data/3',
#     '027': '/media/ivsr/data2/pyskl/video_data/4',
#     '043': '/media/ivsr/data2/pyskl/video_data/5'
# }
# # for i in target_dirs:
# #     print(i)
# # Ensure all target directories exist
# for target_dir in target_dirs.values():
#     os.makedirs(target_dir, exist_ok=True)
# # Loop through each video file in the source directory
# for video_file in os.listdir(source_dir):
#     if video_file.endswith('.avi'):
#         # Extract label from video file name
#         label = video_file.split('.')[0].split('_')[0][-3:]
#         # print(label)
#         # Check if the label matches any of the target labels
#         if label in target_dirs:
#             target_dir = target_dirs[label]
#             # print(target_dir)
#             # Move the video file to the appropriate target directory
#             shutil.copy(os.path.join(source_dir, video_file), target_dir)

'''
convert vid .avi --> vid .mp4
'''
# import moviepy.editor as moviepy
# clip = moviepy.VideoFileClip("/media/ivsr/data2/pyskl/IMG_4206.MOV")
# clip.write_videofile("/media/ivsr/data2/pyskl/IMG_4206.mp4")

# import subprocess

# input_file = "/media/ivsr/data2/pyskl/IMG_4206.MOV"
# output_file = "/media/ivsr/data2/pyskl/IMG_4206.mp4"

# # Run ffmpeg command to convert the video
# subprocess.run(['ffmpeg', '-i', input_file, '-c:v', 'copy', '-c:a', 'copy', output_file])

# import cv2

# # Replace 'video.mp4' with the path to your video file
# cap = cv2.VideoCapture('/media/ivsr/data2/pyskl/IMG_4579.MOV')

# # Check if video capture was successful
# if not cap.isOpened():
#     print("Error opening video stream or file")
#     exit()

# # Get video width and height using get() method with specific properties
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Print the width and height
# print(f"Video width: {width}")
# print(f"Video height: {height}")

# # Loop through each folder in the source directory
# for folder in os.listdir(source_dir):
#     folder_path = os.path.join(source_dir, folder)
    
#     # Ensure it's a directory
#     if os.path.isdir(folder_path):
#         # Loop through each video file in the folder
#         for video_file in os.listdir(folder_path):
#             if video_file.endswith('.avi'):
#                 # Extract label from video file name
#                 label = video_file.split('.')[0][-3:]
#                 # Check if the label matches any of the target labels
#                 if label in target_dirs:
#                     target_dir = target_dirs[label]
#                     # Create the target directory if it doesn't exist
#                     if not os.path.exists(target_dir):
#                         os.makedirs(target_dir)
#                     # Move the video file to the appropriate target directory
#                     shutil.copy(os.path.join(folder_path, video_file), target_dir)




# line = 'frame: 173'
# frame_num = int(line.split(':')[1])
# print(frame_num)
# import numpy as np
# frames = [1,2,2,3,4,4,5,6]
# unique_frames, frame_counts = np.unique(frames, return_counts=True)
# print(unique_frames)
# print(frame_counts)


# import pickle
# with open("/media/ivsr/data2/pyskl/test_hrnet66.pkl", "rb") as file: 
#     obj = pickle.load(file)

# obj =dict (obj)
# print(obj['split'])

""" thu vid de test"""

# import numpy as np 
# import cv2 

# # This will return video from the first webcam on your computer. 
# cap = cv2.VideoCapture(0) 

# # Define the codec and create VideoWriter object 
# fourcc = cv2.VideoWriter_fourcc(*'XVID') 
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) 

# # loop runs if capturing has been initialized. 
# while(True): 
# 	# reads frames from a camera 
# 	# ret checks return at each frame 
# 	ret, frame = cap.read() 

# 	# Converts to HSV color space, OCV reads colors as BGR 
# 	# frame is converted to hsv 
# 	# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
	
# 	# output the frame 
# 	out.write(frame) 
	
# 	# The original input frame is shown in the window 
# 	cv2.imshow('Original', frame) 

# 	# The window showing the operated video stream 
# 	# cv2.imshow('frame', hsv) 

	
# 	# Wait for 'a' key to stop the program 
# 	if cv2.waitKey(1) & 0xFF == ord('a'): 
# 		break

# # Close the window / Release webcam 
# cap.release() 

# # After we release our webcam, we also release the output 
# out.release() 

# # De-allocate any associated memory usage 
# cv2.destroyAllWindows() 

## cat vid ne
# from moviepy.video.io.VideoFileClip import VideoFileClip

# input_video_path = 'output.avi'
# output_video_path = 'test2.mp4'

# with VideoFileClip(input_video_path) as video:
#     new = video.subclip(6,11)
#     new.write_videofile(output_video_path, audio_codec='aac')
