## Auxiliar Library for the MQR tracker

# Python Libraries
import cv2
import numpy as np
import sys


def metrica(x):
    ## Helper function for the circular median
    if x<180:
        return x
    else:
        return 360-x

def Flujo_Caja(frame1,frame2,r):
    # Helper function for estimating the OPFlow of one box
    
    frame1=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame2=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    h, w = frame1.shape[:2]
    flow_Deep=np.zeros((h,w,2),np.float32)
    
    
    #inst_Deep = cv2.optflow.createOptFlow_DeepFlow()
    inst_Deep = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    
    inst_Deep.calc(frame1,frame2,flow_Deep)
    
    
    real_mag,real_ang=cv2.cartToPolar(flow_Deep[:,:,0], flow_Deep[:,:,1], angleInDegrees=True)
    
    # Crop image
    mag = real_mag[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    ang = real_ang[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    
    m=np.median(mag)
    
    #Circular median based on angular distance in the color spac
    ang_2 = np.reshape(ang,-1)
    N = ang_2.shape[0]
    ang_sorted = np.sort(ang_2, kind='quicksort',axis=0)
    ang_diff = np.diff(ang_sorted,axis=0)
    ang_diff = np.reshape(ang_diff,-1)
    d = metrica(abs(ang_sorted[N-1]-ang_sorted[0]))
    ang_diff = np.append(ang_diff,d)
    
    rotogado = ang_sorted[np.argmax(ang_diff)+1:]
    rotogado = np.append(rotogado,ang_sorted[:np.argmax(ang_diff)+1])
    a = rotogado[int(N/2)]
    
    a = a * np.pi/180
    
    #Computing the OF based on the circular median. 
    u = m * np.cos(a)
    v = m * np.sin(a)
    
    cv2.normalize(real_mag,real_mag,0,1,cv2.NORM_MINMAX)
    
    hsv = np.zeros((h, w, 3),np.float32)
    hsv[...,0] = real_ang
    hsv[...,1] = real_mag
    hsv[...,2] = np.ones((h, w), np.float32)
    
    bgr_Deep= cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #(bgr_Deep[...,0]+bgr_Deep[...,1]+bgr_Deep[...,2])/3
    
    rgb = np.zeros((h, w, 3),np.float32)
    rgb[...,0] = real_mag
    rgb[...,1] = real_mag
    rgb[...,2] = real_mag
    
    # Updating the box
    p1 = (int(r[0]+u), int(bbox[1]+v))
    p2 = (int(bbox[0] + bbox[2]+ u ), int(bbox[1] + bbox[3]+ v ))
    cv2.rectangle(rgb, p1, p2, (0,0,255), 2, 1)
    
    return u,v

class TrackerMQR():
    
    frame=[];bbox=[];
    
    def init(self,image,r):
        
        self.frame = image
        self.bbox = r
    
    
    def update(self,new_frame):
        
        u,v = Flujo_Caja(self.frame,new_frame,self.bbox)
        
        self.bbox = (self.bbox[0] + u,self.bbox[1] + v,\
                    self.bbox[2],self.bbox[3])
        self.frame = new_frame
        
        ok = True
        
        return ok, self.bbox
