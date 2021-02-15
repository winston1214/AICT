# xmax,ymax,xmin,ymin

import os
import numpy as np
import cv2
label = os.listdir('label_2')
train_path = 'label_2/' # train annotation path
train_ann = os.listdir(train_path)
tmp = []
for idx,i in enumerate(train_ann):
    print('file_number',idx)
    data = open(train_path+train_ann[idx],'r')
    ls = data.readlines()
    ann_ls = list(map(lambda x: x[:-1],ls)) # remove \n
    ann_ls = list(map(lambda x: x.split(' '),ann_ls))
    x1 = np.array(list(map(lambda x: float(x[4]),ann_ls)))
    y1 = list(map(lambda x: float(x[5]),ann_ls))
    x2 = list(map(lambda x: float(x[6]),ann_ls))
    y2 = list(map(lambda x: float(x[7]),ann_ls))
    category = list(map(lambda x: x[0],ann_ls))
    img = 'image_2/'+i.split('.')[0]+'.png'
    img = cv2.imread(img) # because each image has a different shape.
   
    h = img.shape[0]
    w = img.shape[1]
    
    for num,point in enumerate(category):
        if point == 'Pedestrian' or point == 'Car' or point =='Person':
            
            center_x = np.mean([x1[num],x2[num]])/w
            center_y = np.mean([y1[num],y2[num]])/h
            yolo_height = abs(y2[num]-y1[num])/h
            yolo_width = abs(x2[num]-x1[num])/w
            
            if point == 'Pedestrian' or point =='Person':
                point=0
            else:
                point=1
            print(point, center_x,center_y, yolo_height,yolo_width)
            with open('yolo_label/'+i,'a') as f:
                f.write('{} {} {} {} {}\n'.format(point,center_x,center_y,yolo_width,yolo_height))
