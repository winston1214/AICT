'''
<?xml version="1.0" encoding="utf-8"?>
<annotation>
   <object>
      <name>human</name>
      <pose>unspecified</pose>
      <truncated>0</truncated>
      <difficult>0</difficult>
      <bndbox>
         <xmin>3471</xmin>
         <xmax>3540</xmax>
         <ymin>1195</ymin>
         <ymax>1275</ymax>
      </bndbox>
   </object>
   <folder>SAR</folder>
   <filename>BLA_0001</filename>
   <source>
      <database>HERIDAL database</database>
   </source>
   <size>
      <width>4000</width>
      <height>3000</height>
      <depth>3</depth>
   </size>
</annotation>
'''
   
import numpy as np
import os
from xml.etree.ElementTree import parse
label_path = 'trainImages/labels/'
label_ls = os.listdir(label_path)

for label in label_ls:
    bbox = []
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
