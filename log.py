# import cv2

# vidcap = cv2.VideoCapture('/media/ivsr/data2/pyskl/video_data/0/S003C003P007R001A008_rgb.avi')
# fps = vidcap.get(cv2.CAP_PROP_FPS)

# print(f"{fps} frames per second")
"""
visualize metric
"""
import json
import matplotlib.pyplot as plt

# Đọc file ".log.json"
# log_file = '/media/ivsr/newHDD1/pyskl/work_dirs/posec3d/c3d_light_ntu60_xsub/joint/20231018_172632.log.json'
log_file = '/media/ivsr/data2/pyskl/work_dirs/stgcn++/stgcn++_ntu120_xsub_hrnet/j/20240406_140402.log.json'
with open(log_file, 'r') as f:
    lines = f.readlines()

# Khởi tạo danh sách lưu giá trị accuracy và loss
accuracy_values = []
loss_values = []
acc_val = []
mean_acc = []
# Xử lý từng dòng trong file
for line in lines:
    # Chuyển đổi dòng thành đối tượng JSON
    data = json.loads(line)


    if "mode" in data and data["mode"] == "train":
        accuracy = data.get('top1_acc') #data.get de ko loi key
        loss = data.get('loss')
        accuracy_values.append(accuracy)
        loss_values.append(loss)
    if "mode" in data and data["mode"] == "val":
        accuracy = data.get('top1_acc')
        mean_acc = data.get("mean_class_accuracy")
        acc_val.append(accuracy)

plt.figure(figsize=(10, 5))
plt.plot(loss_values)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(accuracy_values)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(acc_val)
plt.xlabel('Iterations')
plt.ylabel('Accuracy_val')
plt.title('Accuracy_val')
plt.show()

# plt.figure(figsize=(10, 5))
plt.plot(mean_acc)
plt.xlabel('Iterations')
plt.ylabel('mean_acc')
plt.title('mean_acc')
plt.show()