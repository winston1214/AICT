
# YOLOv5

Ref https://github.com/ultralytics/yolov5

## Outline
- Environment
Ubuntu 18.04.5 , TitanXP
- Training
I divide the class into two classes and train them using the COCO dataset. (0: Person :boy: , 1: Car :car:)
- Test
The test was conducted with Kitty dataset. (img)

1. COCO convert to YOLO
:point_right: https://bitbucket.org/yymoto/coco-to-yolo/src/master/

```bash
$ java -jar cocotoyolo.jar "coco/annotations/instances_train2017.json" "/usr/home/madmax/coco/images/train2017/" "person, car" "coco/yolo"
```
= java -jar cocotoyolo.jar "json file(annotations) path" "img path" "class" "save path"

2. Setting directory

- Modify yaml file

```
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ../coco128/images/train2017/
val: ../coco128/images/train2017/

# number of classes
nc: 80

# class names
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
        'teddy bear', 'hair drier', 'toothbrush']
```

:point_up: Raw data.yaml file

If you want to designate the number of classes as something other than 80, you need to modify it accordingly.

**nc** : Number of classes
**train** : Path of train images
**val** : Path of validation images
**names** : Set of class names


- Set directory Path



```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```
<img src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png" width="900">



