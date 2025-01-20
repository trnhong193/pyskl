# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
import warnings
# import tracking_pose_test as hihi
from scipy.optimize import linear_sum_assignment

from pyskl.apis import inference_recognizer, init_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    def inference_detector(*args, **kwargs):
        pass

    def init_detector(*args, **kwargs):
        pass
    warnings.warn(
        'Failed to import `inference_detector` and `init_detector` from `mmdet.apis`. '
        'Make sure you can successfully import these if you want to use related features. '
    )

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    def init_pose_model(*args, **kwargs):
        pass

    def inference_top_down_pose_model(*args, **kwargs):
        pass

    def vis_pose_result(*args, **kwargs):
        pass

    warnings.warn(
        'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
        '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
    )


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('out_filename', help='output filename')
    
    #GCN=False
    # parser.add_argument(
    #     '--config',
    #     default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
    #     help='skeleton action recognition config file path')
    # parser.add_argument(
    #     '--checkpoint',
    #     default='https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth',
    #     help='skeleton action recognition checkpoint file/url')
    
    #GCN = True
    parser.add_argument(
        '--config',
        default='/media/ivsr/data2/pyskl/configs/stgcn++/stgcn++_ntu60_xsub_hrnet/j_old.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default='/media/ivsr/data2/pyskl/work_dirs/stgcn++part1/stgcn++_ntu60_xsub_hrnet/j/epoch_15.pth',
        help='skeleton action recognition checkpoint file/url')
    
    
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_1x_coco-person.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
                 'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
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


# def frame_extraction(video_path, short_side):
#     """Extract frames given video_path.

#     Args:
#         video_path (str): The video_path.
#     """
#     # Load the video, extract frames into ./tmp/video_name
#     target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
#     os.makedirs(target_dir, exist_ok=True)
#     # Should be able to handle videos up to several hours
#     frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
#     vid = cv2.VideoCapture(video_path)
#     frames = []
#     frame_paths = []
#     flag, frame = vid.read()
#     cnt = 0
#     new_h, new_w = None, None
#     while flag:
#         if new_h is None:
#             h, w, _ = frame.shape
#             new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

#         frame = mmcv.imresize(frame, (new_w, new_h))

#         frames.append(frame)
#         frame_path = frame_tmpl.format(cnt + 1)
#         frame_paths.append(frame_path)

#         cv2.imwrite(frame_path, frame)
#         cnt += 1
#         flag, frame = vid.read()

#     return frame_paths, frames,cnt


def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret


def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1], poses[j])

        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)
    if num_joints is None:
        return None, None
    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose
   
    return result[..., :2], result[..., 2]


def create_fake_anno(pose_results, num_frame, h, w, clip_len=10):
    # if num_frame % 10 != 0:
    #     return None
    
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame
    )
    num_keypoint = 17
    keypoint = np.zeros((1, num_frame, num_keypoint, 2), dtype=np.float16)
    keypoint_score = np.zeros((1, num_frame, num_keypoint), dtype=np.float16)
    
    for i, poses in enumerate(pose_results):
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            keypoint[j, i] = pose[:, :2]
            keypoint_score[j, i] = pose[:, 2]

    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score
    
    return fake_anno
# def create_fake_anno(pose_results, clip_len=10):
       
#     return dict(
#         frame_dir='',
#         label=-1,
#         img_shape=(h, w),
#         original_shape=(h, w),
#         start_index=0,
#         modality='Pose',
#         total_frames=num_frame)
#     )


def main():
    args = parse_args()
    
    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']
    
    # model action
    model = init_recognizer(config, args.checkpoint, args.device)

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()] 
    
    # # Load the video, extract frames into ./tmp/video_name
    # target_dir = osp.join('./tmp', osp.basename(osp.splitext(args.video)[0]))
    # os.makedirs(target_dir, exist_ok=True)
    # # Should be able to handle videos up to several hours
    # frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    
    #bat dau vid
    vid = cv2.VideoCapture(args.video)
    
    frames = []
    frame_paths = []
    num_frame =0
    # cnt = 0 #frame_idx
    new_h, new_w = None, None
    pose_results_buffer = []  #luu keypoint, keypoint_score va bbox tam thoi
    results_buffer = [] #luu action tam thoi
    predict_per_nframe =5 
    
    while vid.isOpened():
        flag, frame = vid.read()
        num_frame += 1 
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (args.short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        # frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(f'webcam_frame_{num_frame}.jpg')
        # frame_paths.append(frame_path)
        # num_frame = len(frame_paths)
        cv2.imwrite(frame_paths[-1], frame)
        # cv2.imwrite(frame_path, frame)

        # print(frame_paths)
        # print('\n')
        # print(num_frame)
        
        # Get Human detection results
        det_results = detection_inference(args, frame_paths)
        torch.cuda.empty_cache()

        #get human pose results
        pose_results = pose_inference(args, frame_paths, det_results)
        torch.cuda.empty_cache() 
        
        # pose_results_buffer.append(pose_results)
        # print(pose_results_buffer)
        # print('\n')
        # fake_anno = create_fake_anno(pose_results,num_frame,h,w)
        # fake_anno = dict(
        #     frame_dir='',
        #     label=-1,
        #     img_shape=(h, w),
        #     original_shape=(h, w),
        #     start_index=0,
        #     modality='Pose',
        #     total_frames=num_frame)

        # num_keypoint = 17
        # keypoint = np.zeros((1, num_frame, num_keypoint, 2),
        #                     dtype=np.float16)
        # keypoint_score = np.zeros((1, num_frame, num_keypoint),
        #                             dtype=np.float16)
        # for i, poses in enumerate(pose_results):
        #     for j, pose in enumerate(poses):
        #         pose = pose['keypoints']
        #         keypoint[j, i] = pose[:, :2]
        #         keypoint_score[j, i] = pose[:, 2]
        # fake_anno['keypoint'] = keypoint
        # fake_anno['keypoint_score'] = keypoint_score
        
        # if fake_anno['keypoint'] is None:
        #     action_label = ''
        # else:
        #     results = inference_recognizer(model, fake_anno)
        #     action_label = label_map[results[0][0]]
            
        if num_frame % (predict_per_nframe) ==0:
            
            fake_anno = create_fake_anno(pose_results_buffer,num_frame,h,w)
            if fake_anno is not None:
                results = inference_recognizer(model, fake_anno)
                action_label = label_map[results[0][0]]
                results_buffer.append(action_label)
                pose_results_buffer = [] #reset pose_results_buffer sau moi lan tich luy
            else:
                results_buffer.append('No person detected')
        
        
        
        
        
        
            
        pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                    args.device)
        vis_frames = [
            vis_pose_result(pose_model, frame_paths[i], pose_results[i])
            for i in range(num_frame)
        ]
        
        print(results_buffer)
        for i, label in enumerate(results_buffer):
            for frame in vis_frames:
                cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)
                
        
    # Combine frames into a video clip
    clip = mpy.ImageSequenceClip(frames, fps=24)

    # Save the video output
    clip.write_videofile(args.out_filename, codec='libx264')

    # Release resources
    vid.release()
    cv2.destroyAllWindows()
        # cv2.imshow("aaaa", frame)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break
        # vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
        # vid.write_videofile(args.out_filename, remove_temp=True)

        # tmp_frame_dir = osp.dirname(frame_paths[0])
        # shutil.rmtree(tmp_frame_dir)      
             
if __name__ == '__main__':
    main()