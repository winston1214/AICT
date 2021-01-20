# @Author winston1214

'''
json file organization
{'annotations': [{'id': '0',
   'type': 'bbox',
   'attributes': {'person_pose': 'sitting'},
   'points': [[1602, 978], [1750, 978], [1750, 1073], [1602, 1073]],
   'label': 'person'}],
 'attributes': {},
 'filename': 'sample.jpg',
 'parent_path': 'file_path/',
 'metadata': {'height': 2160, 'width': 3840},
 'Metadata': {'mission-id': '03',
  'status': 'MISSION',
  'drone-id': None,
  'weather': 'SUN',
  'time': 'PM',
  'altitude': 20,
  'angle': 90,
  'places': 'land',
  'location': None,
  'date': '2020-12-08',
  'hour': '13-29',
  'de-identification': 'N'}}
 '''
import json
import numpy as np
with open('sample.json') as f:
  data = json.load(f)
bbox = data['annotations'][0]['points']
img_height = data['metadata']['height']
img_width = data['metadata']['width']
center_x = np.mean([bbox[0][0],bbox[2][0]])
center_y = np.mean([bbox[0][1],bbox[2][1]])
width = bbox[2][0] - bbox[0][0]
height = bbox[2][1] - bbox[0][1]
yolo_center_x = center_x/img_width
yolo_center_y = center_y/img_height
yolo_width = width/img_width
yolo_height = height/img_height
name = data['filename'].split('.')[0]
with open('{}.txt'.format(name),'w') as f:  
  f.write('{} {} {} {} {}'.format(0,yolo_center_x,yolo_center_y,yolo_width,yolo_height)) # pesron index number 0
