'''
test >1 person
'''
# import numpy as np
# import ast
# import sys
# def read_data(file_path):
#   """
#   Hàm này lấy dữ liệu keypoint và keypoint_score từ file .txt và trả về
#   dictionary chứa keypoint, keypoint_score theo từng frame.

#   Args:
#     file_path: Đường dẫn đến file .txt.

#   Returns:
#     Dictionary chứa keypoint, keypoint_score theo từng frame.
#   """

#   with open(file_path, 'r') as file:
#     lines = file.readlines()

#   # Khởi tạo dictionary để lưu trữ dữ liệu
#   data = {}
#   data['keypoint'] = []
#   data['keypoint_score'] = []
#   data['frame_index'] = []

#   # Biến để theo dõi frame hiện tại
#   current_frame = None
#   # Biến để lưu trữ keypoint và keypoint_score tạm thời cho mỗi frame
#   frame_keypoints = []
#   frame_keypoint_scores = []

#   for line in lines:
#     # Xử lý thông tin frame
#     if 'frame:' in line:
#       # Lưu trữ dữ liệu frame trước đó
#       if current_frame is not None:
#         data['keypoint'].append(frame_keypoints)
#         data['keypoint_score'].append(frame_keypoint_scores)
#         data['frame_index'].append(current_frame)

#       # Khởi tạo dữ liệu cho frame mới
#       current_frame = int(line.split()[1])
#       frame_keypoints = []
#       frame_keypoint_scores = []

#     # Xử lý thông tin keypoint
#     elif 'kpoint:' in line:
#       keypoints = ast.literal_eval(line.split(': ')[1])
#       frame_keypoints.append(keypoints)

#     # Xử lý thông tin keypoint_score
#     elif 'score:' in line:
#       keypoint_scores = ast.literal_eval(line.split(': ')[1])
#       frame_keypoint_scores.append(keypoint_scores)

#   # Lưu trữ dữ liệu frame cuối cùng
#   if current_frame is not None:
#     data['keypoint'].append(frame_keypoints)
#     data['keypoint_score'].append(frame_keypoint_scores)
#     data['frame_index'].append(current_frame)

#   # Chuyển đổi dữ liệu sang NumPy array với kích thước mong muốn
#   max_person = max(len(x) for x in data['keypoint'])
#   num_keypoint = len(data['keypoint'][0][0])

#   keypoint = np.zeros((max_person, len(data['frame_index']), num_keypoint, 2), dtype=np.float16)
#   keypoint_score = np.zeros((max_person, len(data['frame_index']), num_keypoint), dtype=np.float16)

#   for i, frame_data in enumerate(data['keypoint']):
#     for person, keypoints in enumerate(frame_data):
#       keypoint[person, i] = keypoints
#       keypoint_score[person, i] = data['keypoint_score'][i][person]

#   return keypoint, keypoint_score

# # Ví dụ sử dụng
# file_path = '/media/ivsr/data2/pyskl/demo/results/example2pp_0.txt'
# keypoint, keypoint_score = read_data(file_path)

# # In ra số lượng người, số lượng frame và số lượng keypoint
# print(f"Số lượng người: {keypoint.shape[0]}")
# print(f"Số lượng frame: {keypoint.shape[1]}")
# print(f"Số lượng keypoint: {keypoint.shape[2]}")

# # Truy cập keypoint và keypoint_score cho người thứ 2, frame thứ 10
# person_id = 1
# frame_id = 9

# person_keypoint = keypoint[person_id, :]
# person_keypoint_score = keypoint_score[person_id, :]

# print(f"Keypoint cho người {person_id} frame {frame_id}: {person_keypoint}")
# print(f"Keypoint score cho người {person_id} frame {frame_id}: {person_keypoint_score}")


import numpy as np

def parse_txt_to_anno(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    frame_dict = {}
    all_keypoints = []
    all_keypoint_scores = []

    current_frame_keypoints = []
    current_frame_keypoint_scores = []

    for line in content:
        line = line.strip()
        if line.startswith('frame'):
            if current_frame_keypoints:
                all_keypoints.append(current_frame_keypoints)
                all_keypoint_scores.append(current_frame_keypoint_scores)
            current_frame_keypoints = []
            current_frame_keypoint_scores = []
            frame_num = int(line.split()[1])
            frame_dict[frame_num] = frame_dict.get(frame_num, 0) + 1
        elif line.startswith('kpoint'):
            kpoint_str = line.split(':')[1].strip()
            keypoints = np.array(eval(kpoint_str))
            current_frame_keypoints.append(keypoints)
        elif line.startswith('score'):
            score_str = line.split(':')[1].strip()
            scores = np.array(eval(score_str))
            current_frame_keypoint_scores.append(scores)

    if current_frame_keypoints:
        all_keypoints.append(current_frame_keypoints)
        all_keypoint_scores.append(current_frame_keypoint_scores)

    max_person = max(frame_dict.values())
    num_frame = len(frame_dict)
    num_keypoint = len(all_keypoints[0][0])

    keypoint = np.zeros((max_person, num_frame, num_keypoint, 2), dtype=np.float16)
    keypoint_score = np.zeros((max_person, num_frame, num_keypoint), dtype=np.float16)

    frame_count = 0
    # print(f'framedict.item: {frame_dict.items()}')
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

file_path = "/media/ivsr/data2/pyskl/demo/results/example2pp_0.txt"
data = parse_txt_to_anno(file_path)
keypoint = data['keypoint']
keypoint_score = data['keypoint_score']

print("Shape of keypoint:", keypoint.shape)
print("Shape of keypoint_score:", keypoint_score)
