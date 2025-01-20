import os
from pyskl.smp import *
import mmcv
from mmcv import load, dump
from sklearn.model_selection import train_test_split

#
folders="/media/ivsr/newHDD1/nturgbd_rgb_s001/nturgbd_videos"
data =[]
# path = os.listdir(folders)
# print(path)
b=[]
label = []
for folder in os.listdir(folders):
    # label1 = os.path.basename(folder)
    a1 =len(folder)
    label = folder[(a1-11):(a1-8)]
    # b.append(load_label)
    vid_name1 = os.path.join(folder,folders)
    vid_name = folder[:(a1-4)]
    # print(len(b))           
    # print(label)       
    data.append((vid_name, label))
    # print(data)

# print(len(data))

# for vn in data: 
      # print(vn[0])
def mwlines(lines, fname):
    with open(fname, 'w') as fout:
        fout.write('\n'.join(lines))   

# print(train)
# for x in train:
      

tmpl = '/media/ivsr/newHDD1/nturgbd_rgb_s001/nturgb+d_rgb/{}'

lines = [(tmpl + ' {}').format(x[0], x[1]) for x in data]
print(lines[0])
# mwlines(lines, 'ntu1.list')

# train, test= train_test_split(data, test_size=0.2, random_state=42)
# annotations = load('aa_ntu11.pkl')
# # print(annotations)
# split = dict()
# split['train'] = [x[0] for x in train]
# split['test'] = [x[0] for x in test]
# dump(dict(split=split, annotations=annotations), 'final_ntu11_hrnet_new.pkl')
# a = load('test_hrnet.pkl')
# for key in a.keys():
#     print(key)
# print(key(a))
# print(a["split"])
# print(a["annotations"])