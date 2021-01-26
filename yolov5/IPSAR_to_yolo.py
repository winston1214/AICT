import numpy as np
import os
from xml.etree.ElementTree import parse
label_path = 'trainImages/labels/'
label_ls = os.listdir(label_path)
bbox = []
for label in label_ls:
    tree = parse('trainImages/labels/'+label)
    root = tree.getroot()
    ann = root.findall('object')
    size = root.findall('size')
    if len(size) == 0: car.append(label)
    else:
        w = int(size[0].findtext('width'))
        h = int(size[0].findtext('height'))
        name = root.findall('filename')[0].text
        name = 'train_'+name
        for i in ann:
            cls = i.find('name').text
            xmin = int(i.find('bndbox').findtext('xmin'))
            xmax = int(i.find('bndbox').findtext('xmax'))
            ymin = int(i.find('bndbox').findtext('ymin'))
            ymax = int(i.find('bndbox').findtext('ymax'))
            center_x = np.mean([xmin,xmax])/w
            center_y = np.mean([ymin,ymax])/h
            width = (xmax-xmin)/w
            height = (ymax-ymin)/h
            bbox.append([center_x,center_y,width,height])
        for k in bbox:
            k = list(map(str,k))
            with open('annotations/{}.txt'.format(name),'a') as f:
                f.write('0 {}\n'.format(' '.join(k)))
