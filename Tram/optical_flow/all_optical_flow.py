import cv2
import numpy as np
def all_optical_flow(before,img):
    final_frame_gray = cv2.cvtColor(before,cv2.COLOR_BGR2GRAY)
    
    lk_params = dict( winSize  = (30,30), maxLevel = 100,criteria = ( cv2.TERM_CRITERIA_COUNT |  cv2.TERM_CRITERIA_EPS, 10, 0.05))
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
    p0 = cv2.goodFeaturesToTrack(final_frame_gray,**feature_params)
    try:
        # if type(p0) != type(None):
        final_frame_gray2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
        p1,st,err = cv2.calcOpticalFlowPyrLK(final_frame_gray,final_frame_gray2,p0,None,**lk_params)
        
        tmp=[]
        for i, (f2,f1) in enumerate(zip(p1,p0)):
            a,b = f2.ravel() 
            c,d = f1.ravel()
            tmp.append([i,a,b,c,d])
            cv2.arrowedLine(img,(a,b),(c,d),(0,0,255),2)
        return tmp
    except:pass