#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge,CvBridgeError
import numpy as np
import torch
from sensor_msgs.msg import Image
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import argparse
import time
from pathlib import Path

# from vector import calc_vector
class ROS:
    def __init__(self):
        self.image_topic = '/pylon_camera_node1/image_raw'
        # self.h = 0
        # self.w = 0
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.image_topic,Image,self.image_callback)
        self.path = 'ros.jpg'
        rospy.spin()
    def image_callback(self,msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg,'bgr8')
        except CvBridgeError as e:
            print(e)
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        ros_img = source == 'ros'

        #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run

        set_logging()
        if ros_img:
            view_img = True
            cudnn.benchmark = True 
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()


        im0s = self.cv_image # bgr
        img = letterbox(im0s, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device) # rgb
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for _, det in enumerate(pred):  # detections per image
            # if webcam:  # batch_size >= 1
            #     p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            # else:
            p, s, im0 = Path(self.path), '', im0s

            # save_path = str(save_dir / p.name)
            # txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
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
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #     # with open(txt_path + '.txt', 'a') as f:
                    #     #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)


            print('%sDone. (%.3fs)' % (s, t2 - t1))

            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == 27:  # q to quit
                    cv2.destroyAllWindows()

            # Save results (image with detections)
            
            # if save:
            #     if dataset.mode == 'images':
            #         # cv2.imwrite(save_path, im0)
            #         cv2.imshow('ros',im0)
                    
            #     else:
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer

            #             fourcc = 'mp4v'  # output video codec
            #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #             self.w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #             self.h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            #         vid_writer.write(im0)
        print('Done. (%.3fs)' % (time.time() - t0))
if __name__=="__main__":
    # Initialize node
    rospy.init_node("detector_manager_node")

    # Define detector object
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
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
    ROS()
            