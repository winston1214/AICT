
# YOLOv5

한국어 버전의 설명 : https://bigdata-analyst.tistory.com/194  #  yolov5 소개

한국어 버전(더욱 상세함): https://bigdata-analyst.tistory.com/195 -- yolov5 train & test                   

                

Original https://github.com/ultralytics/yolov5

## Outline
- Environment
Ubuntu 18.04.5 , GPU : TitanXP , Creating a virtual environment
- Training
I divide the class into two classes and train them using the COCO dataset. (0: Person :boy: , 1: Car :car:)
- Test
The test was conducted with Kitty dataset. (img)

## 1. COCO convert to YOLO
:point_right: https://bitbucket.org/yymoto/coco-to-yolo/src/master/

```bash
$ java -jar cocotoyolo.jar "coco/annotations/instances_train2017.json" "/usr/home/cvai-server/coco/images/train2017/" "person, car" "coco/yolo"
```
= java -jar cocotoyolo.jar "json file(annotations) path" "img path" "class" "save path"

## 2. Setting Enviornment
- Set yolov5

```bash
$ git clone https://github.com/ultralytics/yolov5.git

$ cd yolov5

$ pip install requirements.txt

```

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

All_images.txt will be created if you have completed the previous process.

You should separate train.txt and validation.txt from this file to create two txt files.

And you can put that txt file in train_path(or validation_path).


- Set directory Path
<img src = https://user-images.githubusercontent.com/47775179/93779003-71d47e00-fc61-11ea-8af4-6a23595f9f25.PNG>

Set up a directory as shown in the picture above.

## 3. Training

***Basic ver***
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```
I carried out as follows

```bash
$ python3 train.py --data ./data/coco.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --batch 64 --img 400 --epochs 50 --name ep50
```

**data** : Path of data.yaml

**weights** : Path of pretrained file path (default='')

**batch** : batch_size

**cfg** : Path of yolov5 architecture (4 type of version)
        s : small , m: medium , l : large , x: xlarge
        
**names** : Last save name

[yolov5 ver]<img src=https://user-images.githubusercontent.com/26833433/90187293-6773ba00-dd6e-11ea-8f90-cd94afc0427f.png>


# 4. Test

```bash
$ python3 detect.py --source ~/test_img/2011_09_26_drive_0091_sync/image_01/data/ --weights ./runs/exp5_ep50/weights/my_best.pt
```

**source** : Path of Test images

**weights** : Path of weights file


I was converted png to mp4
:point_down: :point_down: **Click there! You can see test video**



<center><a href="http://www.youtube.com/watch?feature=player_embedded&v=zzaC7ID1fD4
" target="_blank"><img src="http://img.youtube.com/vi/zzaC7ID1fD4/0.jpg" 
alt="Test img" /></a></center>

:point_up: :point_up: **Click there! You can see test video**


Yolov5 Test avi Link : https://youtu.be/zzaC7ID1fD4

### 4-1 ROI optical flow

<a href="https://www.youtube.com/watch?v=JI56dX7vO0E
" target="_blank"><img src="https://www.youtube.com/watch?v=JI56dX7vO0E/0.jpg" 
alt="Test img" /></a>

Yolov5 Test sparse roi optical flow : https://youtu.be/JI56dX7vO0E
