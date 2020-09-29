import cv2
import numpy as np
def dense_optical_flow(x,before,img):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3])) # start, end
    hsv = np.zeros_like(before)
    prvs = cv2.cvtColor(before,cv2.COLOR_BGR2GRAY)
    hsv[...,1] = 255
    # next_ = img[c1[1]:c2[1],c1[0]:c2[0]] # ROI만 계산
    next_ = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next_,None,pyr_scale = 0.5,levels=3,winsize=15,iterations=3,poly_n=5,poly_sigma=1.2,flags=10) #velocity
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2 # angle
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # hsv = hsv[c1[1]:c2[1],c1[0]:c2[0]]
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return rgb