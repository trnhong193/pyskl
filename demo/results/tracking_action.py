tracker = 'bytetrack.yaml'#'botsort.yaml

import cv2, time, argparse,torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import argparse
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
import warnings
import tracking_pose_test as hihi
from scipy.optimize import linear_sum_assignment

from pyskl.apis import inference_recognizer, init_recognizer
def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument("--video_path", help='video file/url',required=True)
    parser.add_argument("--kptSeqNum", default=None, required=False)
    parser.add_argument("--max_miss", type=int, default=4, required=False)
    parser.add_argument("--GPU_no", type=int, default=0, required=False)
    parser.add_argument(
        '--config',
        default='/media/ivsr/data2/pyskl/configs/stgcn++/stgcn++_ntu60_xsub_hrnet/j_old.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default='/media/ivsr/data2/pyskl/work_dirs/stgcn++part1/stgcn++_ntu60_xsub_hrnet/j/epoch_15.pth',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/nturgbd_120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    args = parser.parse_args()
    return args


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    
    #Plot the skeleton and keypointsfor coco datatset
    kptThres = 0.1
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps
    kp_ten =[]
    conf_ten =[]
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        
        kp_ten.append([x_coord, y_coord])
        
        conf_ten.append(kpts[3* kid + 2])
        
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < kptThres:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
    
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<kptThres or conf2<kptThres:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
    return im












args = parse_args()
video_path = args.video_path
GPU_no = args.GPU_no
max_miss = args.max_miss
kptSeqNum = args.kptSeqNum