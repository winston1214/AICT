# @Author winston1214
# similar to coco format
# <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

import os
import pandas as pd
import numpy as np
train_path = 'VisDrone2019-DET-train/annotations/'
train_ann = os.listdir(train_path)
for idx,i in enumerate(train_ann):
    data = open(train_path + train_ann[idx],'r')
    ls = data.readlines()
    ann_ls = list(map(lambda x: x[:-1],ls))
    ann_ls = list(map(lambda x: x.split(','),ann_ls))
    bbox_left = list(map(lambda x: x[0],ann_ls))
    bbox_top = list(map(lambda x: x[1],ann_ls))
    bbox_width = list(map(lambda x: x[2],ann_ls))
    bbox_height = list(map(lambda x: x[3],ann_ls))
    category = list(map(lambda x: x[5],ann_ls)) # 0 : pedestrian , 1 : person
    img = 'VisDrone2019-DET-train/images/'+i.split('.')[0]+'.jpg'
    img = cv2.imread(img)
    h = img.shape[0]
    w = img.shape[1]
    center_x = (np.array(bbox_left,dtype=int) + np.array(bbox_width,dtype=int)/2)/w
    center_y = (np.array(bbox_top,dtype=int) + np.array(bbox_height,dtype=int)/2)/h
    yolo_width = np.array(bbox_width,dtype=int)/w
    yolo_height = np.array(bbox_height,dtype=int)/h
    for num,point in enumerate(category):
        if point == '0' or point == '1':
            with open('VisDrone2019-DET-train/yolo_annotations/'+i,'a') as f:
                f.write('{} {} {} {} {}\n'.format(point,center_x[num],center_y[num],yolo_width[num],yolo_height[num]))
