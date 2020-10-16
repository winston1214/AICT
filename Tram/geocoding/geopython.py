import math
import numpy as np
class geocoding():
    def __init__(self,lat1,lat2,lon1,lon2):
        self.lat1 = lat1 # start lat
        self.lat2 = lat2 # end lat
        self.lon1 = lon1 # start lon
        self.lon2 = lon2 # end lon
        self.x1 = self.lat1*math.pi/180
        self.x2 = self.lat2*math.pi/180
        self.dx = (self.lat2-lat1) * math.pi/180 # seta_dev
        self.dy = (self.lon2-lon1) * math.pi/180 # lambda_dev
        self.R = 6378.1*1000 # meter
        
    def distance(self):  
        a = math.sin(self.dx/2)**2 + math.cos(self.x1)*math.cos(self.x2)*math.sin(self.dy/2)*math.sin(self.dy/2)
        c = 2*math.atan2(np.sqrt(a),np.sqrt(1-a))
        d = R*c
        return d # meter
    def bearing(self):
        y = math.sin(self.dy) * math.cos(self.x2)
        x = math.cos(self.x1)*math.sin(self.x2)-math.sin(self.x1)*math.cos(self.x2)*math.cos(self.dy)
        seta = math.atan2(y,x)
        brng = (seta*180/math.pi+360)%360
        return brng # 우리가 아는 그 각도
    def points(self,d): # d = km
        lat1 = math.radians(self.lat1) # radian
        lon1 = math.radians(self.lon1) # radian
        brng = math.radians(self.bearing()) # radian
        lat2 = math.asin( math.sin(lat1)*math.cos(d/self.R) +math.cos(lat1)*math.sin(d/self.R)*math.cos(brng))
        lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/self.R)*math.cos(lat1),math.cos(d/self.R)-math.sin(lat1)*math.sin(lat2))
        lat2 = math.degrees(lat2)
        lon2 = math.degrees(lon2)
        return lat2,lon2