import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import math
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from optical_flow import optical_flow, dense_optical_flow

input_data = []
img_ls = []

vector_qx = []  # 백터 저장을 위한 queue _hyeonuk
vector_qy = []
velocity_arr = []
distance_arr = []
location = 0


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    over_object=0
    idx = 0
    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32

        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = Path(path), '', im0s
            im0 = cv2.resize(im0, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            # im0 = cv2.GaussianBlur(im0,(9, 9), 0) #Gaussian filter(9 by 9), 0 means sigma is auto-determined _hyeonuk

            img_ls.append(im0)
            idx += 1
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image

                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                        label = '%s' % (names[int(cls)])
                        input_data.append([int((x1 + x2) // 2), int((y1 + y2) // 2)])
                        distance_mm = (1750 * 18) / ((abs(y1 - y2)) * 0.025)

                        #distance 계산 _hyeonuk
                        velocity = 0
                        if y2 >= im0.shape[0] - 2 or over_object > 1:
                            over_object += 1
                            distance_mm = (1 - 1 / (over_object + 1)) * (1750 * 18) / (
                                        (abs(y1 - y2) * (1 + 0.04 * over_object) + abs(x1 - x2)*0.01*over_object) * 0.025) + \
                                          (1 / (over_object + 1)) * distance_mm
                        else:
                            if over_object > 0:
                                over_object -= 1

                        label = '%s %.2fm' % (names[int(cls)], distance_mm / 1000)

                        if len(input_data) == 1:  # initial location left:-1, middle:0, right:1
                            if (x2 <= im0.shape[1] // 2):
                                location = -1
                            elif (x1 >= im0.shape[1] // 2):
                                location = 1
                            elif (x2 >= im0.shape[1] // 2 and x1 <= im0.shape[1] // 2):
                                location = 0

                        if len(img_ls) >= 2:  # changed 10 to 2 _hyeonuk
                            before = img_ls[-2]
                            cur = img_ls[-1]

                            mo_x, mo_y = dense_optical_flow(xyxy, before, cur)

                            # 대표 백터 accumulating, size 10 queue by x, y_hyeonuk
                            # motion vector의 y성분이 양수인 경우 음수로 변경 _ problem, 만약 앞으로 가는 사람은 어떻게 인지할 것인
                            if mo_y > 0:  # 사람이 온다고 가정
                                mo_y = -mo_y

                            if len(vector_qx) > 5:
                                vector_qx.pop(0)
                                vector_qy.pop(0)
                                vector_qx.append(mo_x)
                                vector_qy.append(mo_y)
                                x_sum = sum(vector_qx)
                                y_sum = sum(vector_qy)
                                x_mean = x_sum  # / len(vector_qx)
                                y_mean = y_sum  # / len(vector_qy)

                                if distance_mm > 1000:
                                    distance_arr.pop(0)
                                    distance_arr.append(distance_mm)

                                velocity = distance_arr[-1] - distance_arr[0]
                                if len(velocity_arr) > 30:
                                    velocity_arr.pop(0)
                                velocity_arr.append(velocity)

                                velocity = np.mean(np.array(velocity_arr))

                            else:
                                vector_qx.append(mo_x)
                                vector_qy.append(mo_y)
                                x_sum = sum(vector_qx)
                                y_sum = sum(vector_qy)
                                x_mean = x_sum  # / len(vector_qx)
                                y_mean = y_sum  # / len(vector_qy)

                                distance_arr.append(distance_mm)
                                velocity_arr.append(velocity)

                            velocity=velocity*(vid_cap.get(cv2.CAP_PROP_FPS))/3
                            # print('%.2f' % x_mean, '%.2f' % y_mean)

                            # 대표백터 magnitude angle로 변경 _hyeonuk
                            mag = np.sqrt((math.pow(x_mean, 2)) + (math.pow(y_mean, 2)))  # 수정 예
                            ang_posmean = math.atan(y_mean / x_mean)

                            # arctan(-90~90) -> 0~360 변환
                            if x_mean > 0 and y_mean > 0:  # 1사분면
                                ang = ang_posmean
                            elif x_mean < 0 and y_mean > 0:  # 2사분면
                                ang = np.pi + ang_posmean
                            elif x_mean < 0 and y_mean < 0:  # 3사분면
                                ang = np.pi + ang_posmean
                            else:  # 4사분면
                                ang = 2 * np.pi + ang_posmean

                            degree = math.degrees(ang)
                            if location == -1:  # (x2<=im0.shape[1]//2): # left
                                if (degree > 270) and (degree <= 315) and (distance_mm / 1000 < 1 ):  # magnitude 조건 추가 예정
                                    cv2.putText(im0, 'Warning!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255),3)
                                    cv2.circle(im0, (int(im0.shape[1]/2 - x_mean * 2),int(im0.shape[0] - 30 + y_mean)),
                                               color=(255,255,0), radius = 10, thickness = -1)
                                    cv2.putText(im0, 'TTC : {:.2f}s'.format(-distance_mm/velocity), (130, 150), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
                                    degree = 315
                                else:
                                    cv2.putText(im0, 'Safe!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 3)
                            elif location == 1:  # (x1>=im0.shape[1]//2) : # right
                                if (degree <= 270) and (degree > 225)and (distance_mm / 1000 < 1 ):
                                    cv2.putText(im0, 'Warning!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255),3)
                                    cv2.circle(im0,
                                               (int(im0.shape[1] / 2 - x_mean * 2), int(im0.shape[0] - 30 + y_mean)),
                                               color=(255, 255, 0), radius=10, thickness=-1)
                                    cv2.putText(im0, 'TTC : {:.2f}s'.format(-distance_mm / velocity), (130, 150),
                                                cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
                                    degree = 225

                                else:
                                    cv2.putText(im0, 'Safe!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 3)
                            elif location == 0:  # (x2 >= im0.shape[1]//2 and x1<=im0.shape[1]//2): # center
                                if (np.mean([225, 270]) <= degree) and degree <= np.mean([270, 315])and (distance_mm / 1000 < 1 ):
                                    cv2.putText(im0, 'Warning!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255),3)
                                    cv2.circle(im0,
                                               (int(im0.shape[1] / 2 - x_mean * 2), int(im0.shape[0] - 30 + y_mean)),
                                               color=(255, 255, 0), radius=10, thickness=-1)
                                    cv2.putText(im0, 'TTC : {:.2f}s'.format(-distance_mm / velocity), (130, 150),
                                                cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
                                    degree = 270
                                else:
                                    cv2.putText(im0, 'Safe!', (130, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 3)

                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            new_x = center_x + mag * np.cos(math.radians(degree))
                            new_y = center_y - mag * np.sin(math.radians(degree))
                            cv2.arrowedLine(im0, (int(center_x), int(center_y)), (int(new_x), int(new_y)), (0, 0, 255),
                                            5)

                            # try:
                            #     ang,mag = optical_flow(xyxy,img_ls[-2],img_ls[-1])
                            #     if ang <0 : ang += 360
                            #     center_x = (x1+x2)//2
                            #     center_y = (y1+y2)//2
                            #     new_x = center_x + mag*np.cos(math.radians(ang))
                            #     new_y = center_y - mag*np.sin(math.radians(ang))
                            #     cv2.arrowedLine(im0,(int(center_x),int(center_y)),(int(new_x),int(new_y)),(0,0,255),7)
                            # except:pass

                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # if (len(input_data) >= 2): # motion vector
                #     before = input_data[num-int(n):-int(n)]
                #     current = input_data[-int(n):]
                #     num += int(n)
                #     before,current = sorted(before),sorted(current) # 거리가 가장 가까운거 정렬로 임시방편..
                #
                #
                #     # 가장 가까운 점들끼리 연결 코드 작성
                #
                #     if len(before) == len(current):
                #         for i in range(len(current)):
                #             distance = np.sqrt(math.pow(before[i][0]-current[i][0],2) + math.pow(before[i][1] - current[i][1],2))
                #             #cv2.arrowedLine(im0,(before[i][0],before[i][1]),(current[i][0],current[i][1]),(0,0,255),3)
                #             if distance >= 100: # Regulation
                #                 pass
                #             else:
                #                 cv2.circle(im0,(before[i][0],before[i][1]),5,(0,0,255),-1)
                #                 cv2.line(im0,(before[i][0],before[i][1]),(current[i][0],current[i][1]),(0,255,0),7)
                #                 cv2.circle(im0,(current[i][0],current[i][1]),5,(0,0,255),-1)
                # else:num+=int(n)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)

            if save_img:
                if dataset.mode == 'images':
                    # cv2.imwrite(save_path, im0)
                    cv2.imshow('ros', im0)

                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    im0 = cv2.resize(im0, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
