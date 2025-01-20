
# import numpy as np

# import re

# def parse_file(file_path):
#     with open(file_path, 'r') as file:
#         content = file.read()

#     frames = re.findall(r'frame: (\d+)\n(.*?)\n(.*?)\n', content, re.DOTALL)
    
#     keypoints_list = []
#     scores_list = []

#     for frame_number, keypoints_str, scores_str in frames:
#         keypoints = eval(keypoints_str)
#         scores = eval(scores_str)

#         keypoints_list.append(keypoints)
#         scores_list.append(scores)

#     keypoints_output = "[[" + "]\n\n[".join(map(str, keypoints_list)) + "]]"
#     scores_output = "[[" + "]\n\n[".join(map(str, scores_list)) + "]]"
#     return len(frames), keypoints_output, scores_output

#     # return f"FRAME\n{len(frames)}\nKEYPOINT\n{keypoints_output}\nKEYPOINT_SCORE\n{scores_output}\n"

# file_path = "/media/ivsr/data/pyskl/testsfuck.txt"  # Update with the path to your file
# formatted_data = parse_file(file_path)[1]
# print(formatted_data)

# keypoints_np = np.array([np.array(frame_keypoints) for frame_keypoints in formatted_data[1]])
# scores_np = np.array([np.array(frame_scores) for frame_scores in formatted_data[2]])

# print(keypoints_np)
# print(type(keypoints_np))  # Output: <class 'numpy.ndarray'>
# print(type(scores_np))    # Output: <class 'numpy.ndarray'>


# import numpy as np
# import ast
# import sys

# def parse_hongga(file_path):
#     '''
#     HONG RAT GA TRONG PYTHON
#     '''
    
#     with open(file_path, 'r') as file:
#         content = file.read()
    
#     kp_dict = {}
#     kp_s_dict = {}

#     lines = content.splitlines()
    
#     i = 0
#     while i < len(lines):
#         line = lines[i]
        
#         if 'frame' in line:
            
#             try:
#                 this_kp = ast.literal_eval(lines[i+1])
#             except Exception as e:
#                 sys.exit(-1)
                
#             try:
#                 this_kp_s = ast.literal_eval(lines[i+2])
#             except Exception as e:
#                 sys.exit(-2)
                
#             if str(this_kp) not in kp_dict:
#                 kp_dict[str(this_kp)] = []
#                 kp_s_dict[str(this_kp)] = []

#             kp_dict[str(this_kp)].append(this_kp)
#             kp_s_dict[str(this_kp)].append(this_kp_s)
                
#             i+=3
#         else:
#             i+=1
            
#     # Convert dictionaries to numpy arrays
#     num_person = len(kp_dict)
#     num_frame = len(kp_dict[list(kp_dict.keys())[0]])
#     num_keypoint = len(kp_dict[list(kp_dict.keys())[0]][0])

#     kp_np = np.zeros((num_person, num_frame, num_keypoint, 2), dtype=np.float16)
#     kp_s_np = np.zeros((num_person, num_frame, num_keypoint), dtype=np.float16)

#     for person_idx, (kp_key, kp_list) in enumerate(kp_dict.items()):
#         for frame_idx, kp_frame in enumerate(kp_list):
#             kp_np[person_idx, frame_idx] = np.array(kp_frame)

#     for person_idx, (kp_key, kp_s_list) in enumerate(kp_s_dict.items()):
#         for frame_idx, kp_s_frame in enumerate(kp_s_list):
#             kp_s_np[person_idx, frame_idx] = np.array(kp_s_frame)
    
#     return kp_np, kp_s_np

# # Example usage
# file_path = "/media/ivsr/data/pyskl/testsfuck.txt"
# kp, kp_s = parse_hongga(file_path)

# print(kp.shape)
# print(kp_s.shape)


# import numpy as np
# import ast
# import sys

# def parse_hongga_enhanced(file_path):
#     """
#     Parses a HONGGA file, handling multiple persons per frame and calculating true num_frame.

#     Args:
#         file_path (str): Path to the HONGGA file.

#     Returns:
#         tuple: (kp, kp_sc, num_person, num_frame, num_keypoint)
#     """

#     kp_list = []
#     kp_sc_list = []
#     frame_nums = []

#     with open(file_path, 'r') as file:
#         content = file.read()

#         lines = content.splitlines()
#         i = 0
#         while i < len(lines):
#             line = lines[i]

#             if 'frame' in line:
#                 frame_num = int(line.split(':')[1])
#                 # frame_num = int(line[7:])
#                 # frame_num_str = line.split(':')[1].strip()  # Remove leading/trailing whitespaces
#                 # frame_num = int(frame_num_str)
#                 # try:
#                 #     frame_num = int(line.split(':')[1])
#                 # except ValueError:
#                 # # Handle the error gracefully (e.g., log the issue or skip the line)
#                 #     pass
#                 try:
#                     kp_str = ast.literal_eval(lines[i + 1])
#                     kp_sc_str = ast.literal_eval(lines[i + 2])
#                 except Exception as e:
#                     print(f'Error parsing data at line {i + 1}: {e}')
#                     sys.exit(-1)

#                 kp = np.array(kp_str)
#                 kp_sc = np.array(kp_sc_str)

#                 kp_list.append(kp)
#                 kp_sc_list.append(kp_sc)
#                 frame_nums.append(frame_num)

#                 i += 3
#             else:
#                 i += 1

#     num_person = len(frame_nums)  # Number of persons based on frame repetitions
#     frame_nums = np.unique(frame_nums)  # Get unique frames
#     num_frame = len(frame_nums)  # True num_frame

#     # Ensure kp and kp_sc are lists of NumPy arrays
#     kp = [np.array(kp) for kp in kp_list]
#     kp_sc = [np.array(kp_sc) for kp_sc in kp_sc_list]

#     # Reorder data based on frame numbers
#     kp = [kp[i] for i in np.argsort(frame_nums)]
#     kp_sc = [kp_sc[i] for i in np.argsort(frame_nums)]

#     # Get number of keypoints from first frame
#     num_keypoint = kp[0].shape[1]

#     # Create final arrays with desired dimensions
#     kp = np.zeros((num_person, num_frame, num_keypoint, 2), dtype=np.float16)
#     kp_sc = np.zeros((num_person, num_frame, num_keypoint), dtype=np.float16)

#     for i in range(num_person):
#         for j in range(num_frame):
#             kp[i, j] = kp_list[i * num_frame + j]
#             kp_sc[i, j] = kp_sc_list[i * num_frame + j]

#     return kp, kp_sc, num_person, num_frame, num_keypoint
# file_path = "/media/ivsr/data/pyskl/testsfuck.txt"
# kp = parse_hongga_enhanced(file_path)[0]
# print(kp)

import numpy as np
import ast
import sys

def parse_hongga(file_path):
    kps, scores, frames = [], [], []

    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'frame' in line:
                frame_num = int(line.split(':')[1])
                kp = []
                kp_score = []
                while i < len(lines) - 1 and 'frame' not in lines[i + 1]:
                    try:
                        this_kp = ast.literal_eval(lines[i+1])
                        this_kp_s = ast.literal_eval(lines[i+2])
                    except Exception as e:
                        sys.exit(-1)
                    kp.append(this_kp)
                    kp_score.append(this_kp_s)
                    i += 3
                frames.append(frame_num)
                kps.append(kp)
                scores.append(kp_score)
            else:
                i += 1

    # Get unique frames and count people per frame
    unique_frames, frame_counts = np.unique(frames, return_counts=True)

    # Create arrays with the desired shape
    num_persons = max(frame_counts)
    num_frames = len(unique_frames)
    num_keypoints = len(kps[0][0])

    keypoint = np.zeros((num_persons, num_frames, num_keypoints, 2), dtype=np.float16)
    keypoint_score = np.zeros((num_persons, num_frames, num_keypoints), dtype=np.float16)

    # Populate arrays
    person_index = 0
    for i, (frame_num, count) in enumerate(zip(unique_frames, frame_counts)):
        start_index = np.where(frames == frame_num)[0][0]
        end_index = start_index + count
        for j in range(count):
            this_kp = np.array(kps[start_index + j])
            keypoint[person_index, i, :num_keypoints] = this_kp[:, :2]  # Extract x, y coordinates
            keypoint_score[person_index, i, :num_keypoints] = np.array(scores[start_index + j])
            person_index += 1
        person_index = 0

    return keypoint, keypoint_score

# Example usage
file_path = "/media/ivsr/data2/pyskl/testsfuck.txt"
keypoint, keypoint_score = parse_hongga(file_path)
print("Keypoints array shape:", keypoint.shape)
print("Keypoints scores array shape:", keypoint_score.shape)
