import cv2
import numpy as np
import math #for cosine, sine calculation _hyeonuk
import matplotlib.pyplot as plt
from collections import Counter
def optical_flow(x,before,img): # Lucas Kanade
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3])) # start, end
    final_frame_gray = cv2.cvtColor(before,cv2.COLOR_BGR2GRAY)
    
    lk_params = dict( winSize  = (30,30), maxLevel = 100,criteria = ( cv2.TERM_CRITERIA_COUNT |  cv2.TERM_CRITERIA_EPS, 10, 0.05))
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.05,
                       minDistance = 10,
                       blockSize = 7 )
    p0 = cv2.goodFeaturesToTrack(final_frame_gray,**feature_params)
    idx=0
    for i in p0: # range out delete
        if (i[0][0]<=c1[0]) or (i[0][0]>=c2[0]) or (i[0][1]<=c1[1]) or(i[0][1]>=c2[1]):
            p0 = np.delete(p0,idx,axis=0)
        else:
            idx+=1

    final_frame_gray2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    try:
        p1,st,err = cv2.calcOpticalFlowPyrLK(final_frame_gray,final_frame_gray2,p0,None,**lk_params)
        # good_new = p1[st==1]
        # good_old = p0[st==1]
        # angle_array = []
        # mag_array = []
        mask = np.zeros_like(before)

        for f2,f1 in zip(p1,p0):
            a,b = f2.ravel()
            c,d = f1.ravel()
            # cv2.arrowedLine(img, (a, b), (c, d), (0, 0, 255), 2)
            cv2.line(mask,(a,b),(c,d),(0,0,255),2)
        return mask
    except:pass


def dense_optical_flow(x,before,img):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3])) # start, end
    hsv = np.zeros_like(before, dtype=np.float64)
    output = np.zeros_like(before, dtype=np.float64)

    prvs = cv2.cvtColor(before,cv2.COLOR_BGR2GRAY)
    next_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hsv[...,1],output[...,1] = 255,255
    # img = img[c1[1]:c2[1],c1[0]:c2[0]] # ROI만 계산

    #grid, find Homography matrix, stabilize _hyeonuk
    # height, width = img.shape[:2]
    # step = 64
    # idx_y, idx_x = np.mgrid[height / 4:3 * height / 4:step, width / 4:3 * width / 4:step].astype(np.int)
    # indices = np.stack((idx_x, idx_y), axis=-1).reshape(-1, 1, 2)
    # prevPt = np.float32(indices)
    #
    # nextPt, status, err = cv2.calcOpticalFlowPyrLK(prvs, next_, prevPt, None,
    #                                                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.05))
    # Hmatrix, tmp = cv2.findHomography(prevPt, nextPt, method=cv2.LMEDS)
    # stab_img = cv2.warpPerspective(prvs, Hmatrix, (width, height))


    flow = cv2.calcOpticalFlowFarneback(prvs,next_,None,pyr_scale = 0.5,levels=6,winsize=15,iterations=3,poly_n=5, \
                                        poly_sigma=1.1,flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    #velocity, level 3->6, flag 10->Gaussian, poly_sigma 1.2->1.1, prv->stab_img _hyeonuk

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1]) # ang = radian

    hsv[...,0] = ang#*180/np.pi/2 # radian type
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    hsv[...,2] = mag

    output[c1[1]:c2[1],c1[0]:c2[0]] = hsv[c1[1]:c2[1],c1[0]:c2[0]]
    # rgb = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    # return rgb

    bbox = output[c1[1]:c2[1],c1[0]:c2[0]]

    angle_ls = bbox[...,0]
    mag_ls = bbox[...,2]

    # 각 픽셀들에 대한 백터합 평균 _hyeonuk

    angle_ls = angle_ls.flatten()
    mag_ls = mag_ls.flatten()
    # print(angle_ls)
    #
    # return angle_ls
    # print(min(angle_ls), max(angle_ls))
    # print(min(angle_ls), max(mag_ls))

    x_ls = [a * math.cos(b) for a, b in zip(mag_ls, angle_ls)]
    y_ls = [a * math.sin(b) for a, b in zip(mag_ls, angle_ls)]

    x_mean = np.mean(x_ls)
    y_mean = np.mean(y_ls)

    return x_mean, y_mean #return 값을 magnitude, angle이 아닌 x, y형태로 받음. _hyeonuk
