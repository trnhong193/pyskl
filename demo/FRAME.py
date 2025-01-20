# def read_file(file_path):
#   """
#   Reads a file containing frame information and returns them as a list.

#   Args:
#       file_path: The path to the text file.

#   Returns:
#       A list of frames, where each frame is a list containing:
#           - a list of 2D points, where each point is a list of [x, y] coordinates.
#           - a list of confidences corresponding to the points in the previous list.
#   """
#   frames = []
#   current_frame_points = []
#   current_frame_confidences = []

#   with open(file_path, 'r') as f:
#     for line in f:
#       line = line.strip()
#       if line.startswith("frame:"):
#         # Handle frame information
#         frame_number = int(line.split(":")[1])
#         if frames:
#           frames.append([current_frame_points, current_frame_confidences])
#         current_frame_points = []
#         current_frame_confidences = []
#       else:
#         # Handle points and confidences
#         data = line.split()
#         point = [float(x) for x in data[:2]]
#         confidence = float(data[2])
#         current_frame_points.append(point)
#         current_frame_confidences.append(confidence)

#   # Add the last frame if it exists
#   if current_frame_points:
#     frames.append([current_frame_points, current_frame_confidences])

#   return frames

# # Example usage
# file_path = "/home/ivsr/Downloads/testsfuck.txt"  # Replace with your actual file path
# frames = read_file(file_path)

# # Access frame data
# for frame_data in frames:
#   points = frame_data[0]
#   confidences = frame_data[1]
#   # Process points and confidences as needed
#   print(f"Frame: {points}")
#   print(f"Confidences: {confidences}")

# import re

# def read_data_from_file(file_path):
#     frames = []
#     with open(file_path, 'r') as file:
#         frame_data = {}
#         for line in file:
#             if line.startswith('frame'):
#                 if frame_data:
#                     frames.append(frame_data)
#                     frame_data = {}
#                 frame_number = int(re.search(r'\d+', line).group())
#                 frame_data['frame_number'] = frame_number
#             elif line.strip():
#                 if 'points' not in frame_data:
#                     frame_data['points'] = eval(line.strip())
#                 else:
#                     frame_data['confidences'] = eval(line.strip())
#         # Append the last frame after loop ends
#         if frame_data:
#             frames.append(frame_data)
#     return frames

# file_path = '/home/ivsr/Downloads/testsfuck.txt'
# frames = read_data_from_file(file_path)
# kp = []
# kps = []
# # Now you can access the data for each frame
# for frame in frames:
#     frame_number = frame['frame_number']
#     points = frame['points']
#     confidences = frame['confidences']
#     kp.append(points)
#     kps.append(confidences)
#     print(f"Frame {frame_number}:")
#     print("Points:", kp)
#     print("Confidences:", kps)

import numpy as np

def read_frame_info(file_path):
    frames = []
    confidences = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        frame_points = []
        frame_confidences = []
        for line in lines:
            if line.startswith('[[['):
                # Parse the array of 2D points
                points = np.array(eval(line.strip()))
                frame_points.append(points)
            elif line.startswith('[['):
                # Parse the array of confidences
                confidence = np.array(eval(line.strip()))
                frame_confidences.append(confidence)
            elif line.strip():
                # Start of a new frame, store the points and reset frame_points and frame_confidences
                if frame_points:
                    frames.append(frame_points)
                    confidences.append(frame_confidences)
                    frame_points = []
                    frame_confidences = []
        if frame_points:
            frames.append(frame_points)
            confidences.append(frame_confidences)
    return frames, confidences

file_path = '/home/ivsr/Downloads/testsfuck.txt'  # Change this to your file path
frames, confidences = read_frame_info(file_path)
print("Frames:")
for frame in frames:
    print(frame)
print("\nConfidences:")
for confidence in confidences:
    print(confidence)

