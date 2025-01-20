import numpy as np
import ast

# def parse_hongga(file_path):

#     kps, scores, frames = [], [], []

#     with open(file_path, 'r') as file:

#         lines = file.read().splitlines()

#         i = 0

#         while i < len(lines):

#             line = lines[i]

#             if 'frame' in line:

#                 frame_num = int(line.split(':')[1])

#                 kp = []

#                 kp_score = []

#                 while i < len(lines) and 'frame' not in lines[i + 1]:

#                     # kp.append(ast.literal_eval(lines[i + 1]))
#                     try:
#                         this_kp = ast.literal_eval(lines[i+1])
#                     except Exception as e:
#                         sys.exit(-1)
                        
#                     kp.append(this_kp)

#                     # kp_score.append(ast.literal_eval(lines[i + 2]))
#                     try:
#                         this_kp_s = ast.literal_eval(lines[i+2])
#                     except Exception as e:
#                         sys.exit(-2)
                
#                     kp_score.append(this_kp_s)
#                     i += 3

#                 frames.append(frame_num)

#                 kps.append(kp)

#                 scores.append(kp_score)

#             else:

#                 i += 1



#   # Get unique frames and count people per frame

#     unique_frames, frame_counts = np.unique(frames, return_counts=True)



#   # Create arrays with the desired shape

#     num_persons = max(frame_counts)

#     num_frames = len(unique_frames)

#     num_keypoints = len(kps[0][0])



#     kp = np.zeros((num_persons, num_frames, num_keypoints, 2), dtype=np.float16)

#     kp_sc = np.zeros((num_persons, num_frames, num_keypoints), dtype=np.float16)



#     for i, (frame_num, count) in enumerate(zip(unique_frames, frame_counts)):
#         start_index = np.where(frames == frame_num)[0][0]
#         end_index = start_index + count

#         for j in range(count):
#             kp[j, i] = np.array(kps[start_index + j])
#             kp_sc[j, i] = np.array(scores[start_index + j])

#     return kp, kp_sc

# # Example usage
# file_path = "/media/ivsr/data/pyskl/testsfuck.txt"
# kp, kp_s = parse_hongga(file_path)
# print("aaaa")
# print(kp)
# print(kp_s)
# print(f'{type(kp)}, {type(kp_s)}')

# import numpy as np
# import ast
# import sys

# def parse_hongga(file_path):
#     kps, scores, frames = [], [], []

#     with open(file_path, 'r') as file:
#         lines = file.read().splitlines()
#         i = 0
#         while i < len(lines):
#             line = lines[i]
#             if 'frame' in line:
#                 frame_num = int(line.split(':')[1])
#                 kp = []
#                 kp_score = []
#                 while i < len(lines) and 'frame' not in lines[i + 1]:
#                     try:
#                         this_kp = ast.literal_eval(lines[i+1])
#                         this_kp_s = ast.literal_eval(lines[i+2])
#                     except Exception as e:
#                         sys.exit(-1)
#                     kp.append(this_kp)
#                     kp_score.append(this_kp_s)
#                     i += 3
#                 frames.append(frame_num)
#                 kps.append(kp)
#                 scores.append(kp_score)
#             else:
#                 i += 1

#     # Get unique frames and count people per frame
#     unique_frames, frame_counts = np.unique(frames, return_counts=True)

#     # Create arrays with the desired shape
#     num_persons = max(frame_counts)
#     num_frames = len(unique_frames)
#     num_keypoints = len(kps[0][0])

#     kp = np.zeros((num_persons, num_frames, num_keypoints, 2), dtype=np.float16)
#     kp_sc = np.zeros((num_persons, num_frames, num_keypoints), dtype=np.float16)

#     # Populate arrays
#     # for i, (frame_num, count) in enumerate(zip(unique_frames, frame_counts)):
#     #     start_index = np.where(frames == frame_num)[0][0]
#     #     end_index = start_index + count
#     #     for j in range(count):
#     #         this_kp = np.array(kps[start_index + j])
#     #         # Reshape the keypoints array based on its size
#     #         num_keypoints_detected = len(this_kp) // 2
#     #         this_kp = this_kp.reshape((num_keypoints_detected, 2))
#     #         kp[j, i, :num_keypoints_detected] = this_kp
#     #         kp_sc[j, i, :num_keypoints_detected] = np.array(scores[start_index + j])
#     print(unique_frames)
#     return num_persons, num_frames, num_keypoints
#     # return kp, kp_sc

# # Example usage
# file_path = "/media/ivsr/data/pyskl/testsfuck.txt"
# # kp, kp_s = parse_hongga(file_path)
# a,b,c = parse_hongga(file_path)
# print("aaaa")
# print(a)
# print(b)
# print(c)
# # print(f'{type(kp)}, {type(kp_s)}')

# import numpy as np
# import ast
# import sys


# def parse_hongga(file_path):
#     kps, scores, frames = [], [], []

#     with open(file_path, 'r') as file:
#         lines = file.read().splitlines()
#         i = 0
#         while i < len(lines):
#             line = lines[i]
#             if 'frame' in line:
#                 frame_num = int(line.split(':')[1])
#                 kp = []
#                 kp_score = []
#                 while i < len(lines) - 1 and 'frame' not in lines[i + 1]:
#                     try:
#                         this_kp = ast.literal_eval(lines[i+1])
#                         this_kp_s = ast.literal_eval(lines[i+2])
#                     except Exception as e:
#                         sys.exit(-1)
#                     kp.append(this_kp)
#                     kp_score.append(this_kp_s)
#                     i += 3
#                 frames.append(frame_num)
#                 kps.append(kp)
#                 scores.append(kp_score)
#             else:
#                 i += 1

#     # Get unique frames and count people per frame
#     unique_frames, frame_counts = np.unique(frames, return_counts=True)

#     # Create arrays with the desired shape
#     num_persons = max(frame_counts)
#     num_frames = len(unique_frames)
#     num_keypoints = len(kps[0][0])

#     kp = np.zeros((num_persons, num_frames, num_keypoints, 2), dtype=np.float16)
#     kp_sc = np.zeros((num_persons, num_frames, num_keypoints), dtype=np.float16)

#     # Populate arrays
#     person_index = 0
#     for i, (frame_num, count) in enumerate(zip(unique_frames, frame_counts)):
#         start_index = np.where(frames == frame_num)[0][0]
#         end_index = start_index + count
#         for j in range(count):
#             this_kp = np.array(kps[start_index + j])
#             for k in range(num_keypoints):
#                 kp[person_index, i, k] = this_kp[k]
#             kp_sc[person_index, i, :num_keypoints] = np.array(scores[start_index + j])
#             person_index += 1
#         person_index = 0

#     return kp, kp_sc

# # Example usage
# file_path = "/media/ivsr/data/pyskl/testsfuck.txt"
# kp, kp_s = parse_hongga(file_path)
# print("Keypoints array shape:", kp.shape)
# print("Keypoints scores array shape:", kp_s.shape)

import numpy as np
import ast

def parse_hongga_to_anno(file_path):
  """
  Parses a Hong Ga (Red Rooster) data file and returns data for the `fake_anno` dictionary.

  Args:
      file_path: Path to the file containing keypoint data.

  Returns:
      A dictionary containing keypoint and keypoint score information:
          - keypoint: NumPy array with shape (num_person, num_frame, num_keypoint, 2).
          - keypoint_score: NumPy array with shape (num_person, num_frame, num_keypoint).
  """

  with open(file_path, 'r') as file:
    content = file.read()

  lines = content.splitlines()
  frame_dict = {}  # Track unique frames and number of people

  # Initialize empty lists to store keypoints and scores per frame
  all_keypoints = []
  all_keypoint_scores = []
  current_frame = None
  current_keypoints = []
  current_keypoint_scores = []

  for line in lines:
    if 'frame' in line:
      # New frame
      if current_frame is not None:
        all_keypoints.append(current_keypoints)
        all_keypoint_scores.append(current_keypoint_scores)
      current_frame = int(line.split()[1])
      current_keypoints = []
      current_keypoint_scores = []
      frame_dict[current_frame] = frame_dict.get(current_frame, 0) + 1
    else:
      try:
        keypoints = ast.literal_eval(line)
        keypoint_scores = ast.literal_eval(lines[lines.index(line) + 1])
        current_keypoints.append(keypoints)
        current_keypoint_scores.append(keypoint_scores)
      except Exception:
        pass  # Skip lines with errors

  # Add the last frame's data
  if current_frame is not None:
    all_keypoints.append(current_keypoints)
    all_keypoint_scores.append(current_keypoint_scores)

  # Get maximum number of people and keypoints across all frames
  # max_person = max(len(x) for x in all_keypoints)
  max_person = max(frame_dict.values())
  num_keypoint = len(all_keypoints[0][0]) if all_keypoints else 0

  # Convert keypoints and scores to NumPy arrays with appropriate shapes
  keypoint = np.zeros((max_person, len(frame_dict), num_keypoint, 2), dtype=np.float16)
  keypoint_score = np.zeros((max_person, len(frame_dict), num_keypoint), dtype=np.float16)

  frame_count = 0
  for frame, num_people in frame_dict.items():
    for person in range(num_people):
      if person < len(all_keypoints[frame_count]):
        keypoint[person, frame_count] = all_keypoints[frame_count][person]
        keypoint_score[person, frame_count] = all_keypoint_scores[frame_count][person]
    frame_count += 1

  return {
      'keypoint': keypoint,
      'keypoint_score': keypoint_score,
  }
file_path = "/media/ivsr/data2/pyskl/testsfuck.txt"
data = parse_hongga_to_anno(file_path)
keypoint = data['keypoint']
keypoint_score = data['keypoint_score']
num_person, num_frame, num_keypoint,_ = keypoint.shape

for person_idx in range(num_person):
  kp = keypoint[person_idx]
  kps = keypoint_score[person_idx]
  print(keypoint)
  print(keypoint_score)
# print(keypoint)
# print(type(keypoint_score))
# Iterate over the keypoint array and print its contents
# for person_index in range(keypoint.shape[0]):
#     for frame_index in range(keypoint.shape[1]):
#         print(f"Person {person_index + 1}, Frame {frame_index + 1}:")
#         for keypoint_index in range(keypoint.shape[2]):
#             x = keypoint[person_index, frame_index, keypoint_index, 0]
#             y = keypoint[person_index, frame_index, keypoint_index, 1]
#             score = keypoint_score[person_index, frame_index, keypoint_index]
#             print(f"Keypoint {keypoint_index + 1}: (x={x}, y={y}), Score={score}")


# num_person, num_frame, num_keypoint, _ = keypoint.shape
# print(keypoint.shape)
# print("Keypoint data:")
# for person in range(num_person):
#   for frame in range(num_frame):
#     for keypoint_id in range(num_keypoint):
#       x, y = keypoint[person, frame, keypoint_id]
#       print(f"Person {person+1}, Frame {frame+1}, Keypoint {keypoint_id+1}: ({x:.2f}, {y:.2f})")
# print("Keypoints array shape:", kp.shape)
# print("Keypoints scores array shape:", kp_s.shape)



