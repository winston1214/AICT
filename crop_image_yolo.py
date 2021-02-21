# @Author Hyunwook Ahn
import json
import numpy as np
import cv2
import os
place = os.listdir(json_path)[-24:] # 탐지할 파일 리스트
for i in range(24):
    os.mkdir('crop_'+place[i]) # crop 한 이미지를 저장할 폴더
    os.mkdir(json_path+'crop_'+place[i]) # crop한 txt를 저장한 폴더
    for p in os.listdir(json_path+place[i]):
        with open('{}{}/{}'.format(json_path,place[i],p), encoding='utf-8') as f:
            data = json.load(f)
        # for idx, _ in enumerate(data['annotations'])
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

        # ---------------------------여기서부터 시작 modify crop size-------------------------
        #def cropping(width, height, bbox, img_height,img_width, crop_size = 832)
        crop_size = 832

        #비율 조정(직사각형)
#         if img_width > img_height:
#             crop_size_w = crop_size
#             crop_size_h = int(crop_size *(img_height/img_width))
#         else:
#             crop_size_w = int(crop_size *(img_width/img_height))
#             crop_size_h = crop_size
        crop_size_w, crop_size_h = (832,832)
        

        #--------------------------code---------------------------------

        Abbox = bbox
        #탐지된 영역 나머지의 너비, 높이
        try: # 이미지 사이즈가 넘어가는 경우가 있어 try except 구문 
            width_r = crop_size_w - width
            height_r = crop_size_h - height
            #랜덤으로 crop
            x_r = np.random.randint(width_r)
            y_r = np.random.randint(height_r)

            #x부분의 예외처리(crop되는부분이 이미지 바깥으로 가지 않도록)
            if bbox[0][0] - width_r < 0: x_r = np.random.randint(0, bbox[0][0])
            elif bbox[2][0] + width_r > img_width: x_r = np.random.randint(0, (img_width - bbox[2][0]))    

            #y분의 예외처리(crop되는부분이 이미지 바깥으로 가지 않도록)
            if bbox[0][1] - height_r < 0: y_r = np.random.randint(0, bbox[0][1])
            elif bbox[2][1] + height_r > img_height: y_r = np.random.randint(0, (img_height - bbox[2][1]))

            #Crop될 부분 Abbox로 설정
            Abbox[0][0] = bbox[0][0] - x_r
            Abbox[0][1] = bbox[0][1] - y_r
            Abbox[2][0] = bbox[2][0] + (width_r - x_r)
            Abbox[2][1] = bbox[2][1] + (height_r - y_r)

            #전체 이미지의 변화로 중심점의 상대적 위치 변경
            center_x = center_x - Abbox[0][0]
            center_y = center_y - Abbox[0][1]

            ##잘린 이미지 저장
            name = data['filename'].split('.')[0]

            ##읽을 이미지 경로 이름 입력
            f_name = str(place[i])+'/' + name +'.jpg'

            img_array = np.fromfile(f_name, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            cropped_image = img[Abbox[0][1]:Abbox[2][1],Abbox[0][0]:Abbox[2][0],...]

            img_name = 'crop_' + data['filename'].split('.')[0]

            ##저장할 이미지 경로 입력
            f_img_name = 'crop_'+place[i]+'/' + img_name +'.jpg'
            result, encoded_img = cv2.imencode(f_img_name, cropped_image)
            if result:
                with open(f_img_name, mode='w+b') as f:
                    encoded_img.tofile(f)

            ## modifiy jsonfile
            ## image_size
            cropped_image_width = crop_size_w
            cropped_image_height = crop_size_h

            ## Bounding box Points
            ## Cbbox = bbox
            ## Cbbox[0] = bbox[0]-Abbox[0]
            ## Cbbox[1] = bbox[1]-Abbox[0]
            ## Cbbox[2] = bbox[2]-Abbox[0]
            ## Cbbox[3] = bbox[3]-Abbox[0]


            #전체 이미지 비율 대신 Cropped 된 이미지에 대한 비율 -> 마지막 라인 주석처리 필요
            yolo_center_x = center_x/crop_size_w
            yolo_center_y = center_y/crop_size_h
            yolo_width = width/crop_size_w
            yolo_height = height/crop_size_h

            #-------------------end-----------------------------

            name = data['filename'].split('.')[0]
            ##yolo text 입력 위치
            with open('{}{}/{}.txt'.format(json_path,'crop_'+place[i],name),'a') as f:  
                f.write('{} {} {} {} {}'.format(0,yolo_center_x,yolo_center_y,yolo_width,yolo_height))
        except:pass

