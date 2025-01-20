import numpy as np
import ast
import re

def read_data(file_path):
    """
    Hàm này lấy dữ liệu keypoint và keypoint_score từ file .txt và trả về
    dictionary chứa keypoint, keypoint_score theo từng người cho từng frame.

    Args:
        file_path: Đường dẫn đến file .txt.

    Returns:
        Dictionary chứa keypoint, keypoint_score theo từng người cho từng frame.
    """

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Khởi tạo dictionary để lưu trữ dữ liệu
    data = {}
    data['keypoint'] = {}
    data['keypoint_score'] = {}
    data['frame_index'] = {}
    data['person_id'] = {}

    # Biến để theo dõi frame hiện tại và danh sách bbox
    current_frame = None
    bboxes = []

    # Biến để lưu trữ keypoint và keypoint_score tạm thời cho mỗi người trong mỗi frame
    frame_keypoints = {}
    frame_keypoint_scores = {}

    # Biến để theo dõi id của từng người
    person_id_counter = 0

    for line in lines:
        # Xử lý thông tin frame
        if 'frame:' in line:
            # Lưu trữ dữ liệu frame trước đó
            if current_frame is not None:
                # Gán dữ liệu cho từng người
                for person_id, bbox in enumerate(bboxes):
                    data['keypoint'][current_frame][person_id] = frame_keypoints[person_id]
                    data['keypoint_score'][current_frame][person_id] = frame_keypoint_scores[person_id]
                    data['person_id'][current_frame][person_id] = person_id_counter
                    person_id_counter += 1
                data['frame_index'][current_frame] = bboxes

            # Khởi tạo dữ liệu cho frame mới
            current_frame = int(line.split()[1])
            bboxes = []
            frame_keypoints = {}
            frame_keypoint_scores = {}

        # Xử lý thông tin bbox
        elif 'bbox:' in line:
            bbox_str = line.split(":")[1].strip()  # Extract bbox string
            match = re.findall(r'\d+', bbox_str)  # Find all integer values in the bbox string
            bbox = list(map(int, match))  # Convert the integer strings to integers
            bboxes.append(bbox)

        # Xử lý thông tin keypoint
        elif 'kpoint:' in line:
            keypoints = ast.literal_eval(line.split(': ')[1])
            person_id = get_person_id_from_bbox(keypoints, bboxes, current_frame, data)  # Xác định người dựa vào bbox

            # Lưu trữ keypoint và keypoint_score cho người tương ứng
            frame_keypoints[person_id] = keypoints
            keypoint_scores = ast.literal_eval(line.split(': ')[1])
            frame_keypoint_scores[person_id] = keypoint_scores

    # Lưu trữ dữ liệu frame cuối cùng
    if current_frame is not None:
        # Gán dữ liệu cho từng người
        for person_id in range(len(bboxes)):
            data['keypoint'][current_frame][person_id] = frame_keypoints[person_id]
            data['keypoint_score'][current_frame][person_id] = frame_keypoint_scores[person_id]
            data['person_id'][current_frame][person_id] = person_id_counter

        data['frame_index'][current_frame] = bboxes

    return data

def get_bbox_from_keypoints(keypoints):
    """
    Calculate the bounding box surrounding the keypoints.

    Args:
        keypoints: List of keypoints.

    Returns:
        List representing the bounding box [x_min, y_min, width, height].
    """
    x_coords = [point[0] for point in keypoints]
    y_coords = [point[1] for point in keypoints]
    x_min = min(x_coords)
    y_min = min(y_coords)
    width = max(x_coords) - x_min
    height = max(y_coords) - y_min
    return [x_min, y_min, width, height]


# Hàm để xác định người dựa vào bbox và keypoint
def get_person_id_from_bbox(keypoints, bboxes, current_frame, data):
    """
    Hàm này xác định người dựa vào bbox và keypoint.

    Args:
        keypoints: Danh sách keypoint của một người.
        bboxes: Danh sách bbox của tất cả mọi người trong frame.
        current_frame: Frame hiện tại.
        data: Dictionary lưu trữ dữ liệu tracking.

    Returns:
        ID của người tương ứng với keypoint.
    """
    global person_id_counter  # Declare person_id_counter as a global variable
    global frame_keypoints  # Declare frame_keypoints as a global variable
    global frame_keypoint_scores  # Declare frame_keypoint_scores as a global variable
    
    person_id_counter = 0  # Initialize person_id_counter
    frame_keypoints = {}  # Initialize frame_keypoints
    frame_keypoint_scores = {}  # Initialize frame_keypoint_scores
    
    # Tính toán diện tích bbox
    bbox_areas = [bbox[2] * bbox[3] for bbox in bboxes]

    # Tìm bbox có diện tích gần nhất với diện tích bao quanh keypoint
    min_area_diff = float('inf')
    person_id = None
    for i, bbox in enumerate(bboxes):
        bbox_area = bbox[2] * bbox[3]
        keypoint_bbox = get_bbox_from_keypoints(keypoints)
        keypoint_area = keypoint_bbox[2] * keypoint_bbox[3]
        area_diff = abs(bbox_area - keypoint_area)

        # Kiểm tra nếu frame trước đó có person thì ưu tiên gán cho person đó
        if current_frame > 0 and data['person_id'].get(current_frame - 1, None) is not None:
            prev_person_id = data['person_id'][current_frame - 1][i]
            prev_bbox = data['frame_index'][current_frame - 1][i]
            prev_bbox_area = prev_bbox[2] * prev_bbox[3]
            prev_area_diff = abs(prev_bbox_area - keypoint_area)
            if prev_area_diff < area_diff:
                person_id = prev_person_id
                continue

        # Cập nhật person_id_counter nếu không tìm thấy person phù hợp
        if area_diff < min_area_diff:
            min_area_diff = area_diff
            person_id = person_id_counter
            person_id_counter += 1

    return person_id


def get_person_keypoints_and_scores(data, person_id, frame_id):
    """
    Hàm này lấy keypoint và keypoint_score của người có id cụ thể trong frame cụ thể.

    Args:
        data: Dictionary lưu trữ dữ liệu tracking.
        person_id: ID của người.
        frame_id: ID của frame.

    Returns:
        Tuple chứa keypoint và keypoint_score của người.
    """

    keypoints = data['keypoint'][frame_id][person_id]
    keypoint_scores = data['keypoint_score'][frame_id][person_id]

    return keypoints, keypoint_scores 

def get_all_people_keypoints_and_scores(data):
    """
    Hàm này lấy keypoint và keypoint_score của tất cả mọi người trong tất cả các frame.

    Args:
        data: Dictionary lưu trữ dữ liệu tracking.

    Returns:
        Dictionary chứa keypoint và keypoint_score của tất cả mọi người trong tất cả các frame.
    """

    all_people_data = {}
    all_people_data['keypoint'] = {}
    all_people_data['keypoint_score'] = {}

    for frame_id in data['frame_index'].keys():
        all_people_data['keypoint'][frame_id] = {}
        all_people_data['keypoint_score'][frame_id] = {}
        for person_id in range(len(data['frame_index'][frame_id])):
            keypoints, keypoint_scores = get_person_keypoints_and_scores(data, person_id, frame_id)
            all_people_data['keypoint'][frame_id][person_id] = keypoints
            all_people_data['keypoint_score'][frame_id][person_id] = keypoint_scores

    return all_people_data

data = read_data('/media/ivsr/data2/pyskl/demo/results/example2pp_0.txt')

all_people_data = get_all_people_keypoints_and_scores(data)

# Lấy keypoint và keypoint_score của người thứ 2 trong frame thứ 3
person_id = 2
frame_id = 3
keypoints = all_people_data['keypoint'][frame_id][person_id]
keypoint_scores = all_people_data['keypoint_score'][frame_id][person_id]

print(keypoints)
print(keypoint_scores)

