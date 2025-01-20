import os
from pyskl.smp import *
import mmcv
from mmcv import load, dump
from sklearn.model_selection import train_test_split


folders="/media/ivsr/data2/pyskl/video_data"
data =[]
data2 = []
for folder in os.listdir(folders):
    label = os.path.basename(folder)
    path_folder = os.path.join(folders,folder)
    for file in os.listdir(path_folder):
            # vid_name= os.path.basename(file).split('.')[0].split('_')[0]
            vid_name1= os.path.basename(file)
            # vid_name2 = vid_name1[: len(vid_name1)-4]
            vid_name =os.path.join(label,vid_name1)
            
            data.append((vid_name, label))
            # data2.append((vid_name2, label))

# print(data)
# for vn in data2: 
#       print(vn[0])

# for x in data:
#     print(f"x[0]: {x[0]} && x[1]: {x[1]}")
#     print('\n')
def mwlines(lines, fname):
    with open(fname, 'w') as fout:
        fout.write('\n'.join(lines))   

      
'''
tao ntu.list'''
# tmpl = '/media/ivsr/data2/pyskl/video_data/{}'
# lines = [(tmpl + ' {}').format(x[0], x[1]) for x in data]
# mwlines(lines, 'demo/ntu_small.list')

'''
tao anno.pkl'''
#tao file annos.pkl
# python3 tools/data/custom_2d_skeleton.py --video-list demo/ntu_small.list  --out demo/ntu_small_annos.pkl -
# -non-dist

'''
tao hrnet.pkl''' 
# train, test= train_test_split(data, test_size=0.2, random_state=42)
# annotations = load('demo/results/ntu_small_annos.pkl')
'''
only for ntu'''
# for x in annotations:
#     x['frame_dir']=x['frame_dir'].split('_')[0]
# split = dict()
# # split['train'] = [(x.split('/')[1]).split('_')[0] for x in train]
# split['train'] = [x[0].split('_')[0].split('/')[1] for x in train]
# split['test'] = [x[0].split('_')[0].split('/')[1] for x in test]
'''
only for ntu'''
# dump(dict(split=split, annotations=annotations), 'demo/results/ntu_small_true.pkl')


a = load('demo/results/ntu_small_true.pkl')
# for key in a.keys():
    # print(key)
anno = a["annotations"]
# for item in anno:
#     frame_dir = item['frame_dir']
#     print(frame_dir)
print(a["split"])
# print(a["annotations"])