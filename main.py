#Desired Libraries
import os
import sys
import numpy as np
import cv2
from scipy import signal
from scipy import fftpack
import pickle
import time
import errno
from common import anorm2, draw_str
from numpy.linalg import inv

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.serialization import load_lua

import MQR

np.set_printoptions(threshold=np.nan,suppress=True,precision=5,linewidth=120)

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data import VOC_CLASSES as labels
Labels_of_interest = (  # always index 0
                      'bicycle', 'bus', 'car',
                      'motorbike')


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd
net = build_ssd('test', 300, 21)    # initialize SSD
net.load_weights('weights/ssd300_mAP_77.43_v2.pth')

newspace = False

def filter(img):
    
    #Aperture for the different filters
    PI = np.pi
    sigma = [4,8,16,32,64,128,256]
    alpha = [0,PI/6,PI/3,PI/2,-PI/3,-PI/6]

    h,w = img.shape[0:2]
    heigh=int(h)
    weight=int(w)
    print(h,w)
    
    #Load filter bank
    file = open('../Filters.pkl','rb')
    Filtros = pickle.load(file)
    
    #Change image to the new space for filtering. 
    if newspace:
        pix = np.reshape(img,(-1,3))
        Matrix = np.array([[1,-1,0],[1/2,1/2,-1],[1/3,1/3,1/3]])
        r = np.matmul(Matrix,np.transpose(pix))
        img = np.reshape(np.transpose(r),img.shape)


    #Each orientation has a different weight.  
    Filtered=[]
    for i in range(len(alpha)):
        Q = np.zeros((heigh,weight,3))
        for j in range(len(sigma)):
            w = (8/3 * sigma[j])**(-1/10)
            q_R = signal.fftconvolve(img[:,:,0],Filtros[i][j], mode='same')
            q_G = signal.fftconvolve(img[:,:,1],Filtros[i][j], mode='same')
            q_B = signal.fftconvolve(img[:,:,2],Filtros[i][j], mode='same')
            Q[:,:,0]=Q[:,:,0] + w * q_R
            Q[:,:,1]=Q[:,:,1] + w * q_G
            Q[:,:,2]=Q[:,:,2] + w * q_B
        Filtered.append(Q)


    # New weighted sum for each filter.  
    R = np.zeros((heigh,weight,3))
    for Q in Filtered:
        R[:,:,0] = R[:,:,0]+Q[:,:,0]/((np.sum(Q[:,:,0]**2))**(1/2))
        R[:,:,1] = R[:,:,1]+Q[:,:,1]/((np.sum(Q[:,:,1]**2))**(1/2))
        R[:,:,2] = R[:,:,2]+Q[:,:,2]/((np.sum(Q[:,:,2]**2))**(1/2))

    R[:,:,0]=R[:,:,0]/6
    R[:,:,1]=R[:,:,1]/6
    R[:,:,2]=R[:,:,2]/6

    R[:,:,0]=R[:,:,0]-np.min(R[:,:,0])
    R[:,:,0]=R[:,:,0]/np.max(R[:,:,0])

    R[:,:,1]=R[:,:,1]-np.min(R[:,:,1])
    R[:,:,1]=R[:,:,1]/np.max(R[:,:,1])

    R[:,:,2]=R[:,:,2]-np.min(R[:,:,2])
    R[:,:,2]=R[:,:,2]/np.max(R[:,:,2])

    return R

def Color_Matrices():

    #Creating the color Matrices, this needs 
    # improvement for scalability.
    
    A = np.array([0,0,0,1])
    B = np.array([1,1,1,1])

    vertices=[]
    vertices.append(np.array([0,0,1,1]))
    vertices.append(np.array([1/2,0,1,1]))
    vertices.append(np.array([1,0,1,1]))
    vertices.append(np.array([1,0,1/2,1]))
    vertices.append(np.array([1,0,0,1]))
    vertices.append(np.array([1,1/2,0,1]))
    vertices.append(np.array([1,1,0,1]))
    vertices.append(np.array([1/2,1,0,1]))
    vertices.append(np.array([0,1,0,1]))
    vertices.append(np.array([0,1,1/2,1]))
    vertices.append(np.array([0,1,1,1]))
    vertices.append(np.array([0,1/2,1,1]))

    matrices=[]
    for i in range(len(vertices)):
        if i==len(vertices)-1:
            m=np.zeros((4,4))
            m[:,0]=A
            m[:,1]=B
            m[:,2]=vertices[i]
            m[:,3]=vertices[0]
            m=inv(m)
        else:
            m=np.zeros((4,4))
            m[:,0]=A
            m[:,1]=B
            m[:,2]=vertices[i]
            m[:,3]=vertices[i+1]
            m=inv(m)
        matrices.append(m)

    return matrices

def Histograma(im,r,matrices):
    # This function build the color model of a region r. 
    
    # Normalize and crop image
    im = im/255
    im = im[int(r[1]):int(r[3]), int(r[0]):int(r[2])]
    original_shape=im.shape

    #Array for percentages.
    Porcentajes = np.zeros(12)

    #Building the Color Model 

    im = np.reshape(im,(-1,3))   #Image as list of pixels. 
    u = np.ones((im.shape[0],1)) 
    pixeles = np.transpose(np.concatenate((im,u),axis=1))   #Add ones to pixel list

    #Remove achromatics and sum them all. 
    maximos = np.max(pixeles[0:3,:],axis=0)
    minimos = np.min(pixeles[0:3,:],axis=0)
    M = (maximos-minimos)
    acrom = np.sum(M<0.04)
    v_acrom = np.sum(pixeles[0,M<0.04])/(M<0.04).size
    pixeles=np.transpose(np.transpose(pixeles)[M>0.04]) #Threshold for achromatics e=0.04


    r = np.matmul(matrices,pixeles)                         #Product with chromatic pixels
    found = 1 * (r<0).any(axis=1)                             #Find the position of positive numbers
    tetrahedros = np.where(found==0)[0]

    #Build the Histogram. 
    for i in range(len(Porcentajes)):
        Porcentajes[i] = np.sum(tetrahedros==i)

    Porcentajes=np.append(Porcentajes,acrom)

    return Porcentajes/np.sum(Porcentajes), v_acrom


def overlap(bb1,bb2,A1,A2):
    #Ratio of overlaping area between bounding boxes. 

    xa1 = bb1[0]; xa2 = bb1[2]; ya1 = bb1[1]; ya2 = bb1[3]
    xb1 = bb2[0]; xb2 = bb2[2]; yb1 = bb2[1]; yb2 = bb2[3]

    Dx = min(xa2,xb2) - max(xa1,xb1);
    Dx = max(0.0,Dx);

    Dy = min(ya2,yb2) - max(ya1,yb1);
    Dy = max(0.0,Dy);

    A = min(A1,A2)

    ratio = Dx*Dy/A;

    return ratio, A


def overlap_ID(bb1,bb2,ID1,ID2):
    #Ratio of overlapping area of boundin boxes relative to their IDs

    xa1 = bb1[0]; xa2 = bb1[2]; ya1 = bb1[1]; ya2 = bb1[3]
    xb1 = bb2[0]; xb2 = bb2[2]; yb1 = bb2[1]; yb2 = bb2[3]

    A1=(bb1[2]-bb1[0])*(bb1[3]-bb1[1])
    A2=(bb2[2]-bb2[0])*(bb2[3]-bb2[1])


    Dx = min(xa2,xb2) - max(xa1,xb1);
    Dx = max(0.0,Dx);

    Dy = min(ya2,yb2) - max(ya1,yb1);
    Dy = max(0.0,Dy);

    A = min(A1,A2)

    ratio = Dx*Dy/A;

    ID = max(ID1,ID2)

    return ratio, ID

def Detect(rgb_image,net,matrices):
    # Filter Image
    #filtered_image=filter(rgb_image)

    image=cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    Hist=[]
    Coord=[]
    Areas=[]
    Acrom=[]
    Lab=[]

    x = cv2.resize(rgb_image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()

    y = net(xx)

    detections = y.data

    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        label_name = labels[i-1]

        if label_name in Labels_of_interest:

            j = 0
            n_detections=0;

            while detections[0,i,j,0] >= 0.6:

                n_detections+=1
                score = detections[0,i,j,0]
                label_name = labels[i-1]

                pt = (detections[0,i,j,1:]*scale).cpu().numpy()

                if pt[0]>100:

                    h,v_acrom=Histograma(rgb_image,pt,matrices)
                    Hist.append(h)
                    Acrom.append(v_acrom)
                    Coord.append(pt)
                    Areas.append((pt[2]-pt[0])*(pt[3]-pt[1]))
                    Lab.append(label_name)

                j+=1

    k=0;
    while k < len(Hist):
        l=k+1
        while l < len(Hist):
            ovr, A_min = overlap(Coord[k],Coord[l],Areas[k],Areas[l])

            if (ovr>0.7):

                idx=Areas.index(A_min)
                Hist.pop(idx)
                Coord.pop(idx)
                Lab.pop(idx)
                Acrom.pop(idx)
                Areas.pop(idx)

            else:
                l+=1
        k+=1


    return Hist,Coord,Lab,Acrom

def Dist(A,B):
    #Implementation of the histogram distance. 
    prefixsum = np.cumsum(A - B)
    H_dist = np.sum(abs(np.cumsum(A - B)))

    if prefixsum[prefixsum >= 0].size:
        d = np.min(prefixsum[prefixsum >= 0])
        temp = prefixsum - d
        H_dist2 = np.sum(abs(temp))
    else:
        H_dist2 = H_dist

    if prefixsum[prefixsum < 0].size:
        d = np.max(prefixsum[prefixsum < 0])
        temp = prefixsum + d
        H_dist3 = np.sum(abs(temp))
    else:
        H_dist3 = H_dist

    return min(H_dist,H_dist2,H_dist3)

def main():

    #Setting the time in the video to start counting. 
    minute = 0
    second = 0   

    #Read the video.
    cap = cv2.VideoCapture("../video_5.mpg")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frame = (minute*60 + second)*fps
    cap.set(cv2.CAP_PROP_POS_FRAMES,n_frame)

    matrices = Color_Matrices()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    flow=np.zeros((frame_height,frame_width,2),np.float32)
    inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    hsv = np.zeros((frame_height, frame_width, 3),np.float32)
    bgr = np.zeros((frame_height, frame_width, 3),np.float32)


    out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'),\ 
                            15,(frame_width,frame_height))


    l = ['car','bus','motorbike','bicycle']   #Labels of Interest. 
    
    #Variables for storing the Records

    Histograms=[] ;Acromatics=[]; Coordinates=[]; Labels=[]; ID=[]; Found=[]
    Histograms_unk=[] ;Acromatics_unk=[]; Coordinates_unk=[]; Found_unk=[]

    Count=np.zeros(4)

    n_frame=0

    while(True):
        
        # Capture frame-by-frame
        ret, image = cap.read()

        if (ret==True):

            # Change the color-space
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            # If we are receiving the first frame everything is unknown. 
            if(n_frame % 1 == 0):

                if n_frame > 0:
                    inst.calc(prevgray,gray,flow)
                    flow = cv2.medianBlur(flow,3)

                frame_data=Detect(rgb_image,net,matrices)

                print(n_frame)
                
                #If everything is new, then start filling the record database
                if not Histograms and not Histograms_unk:

                    for i in range(len(frame_data[0])):
                        if frame_data[1][i][2]<frame_width-100:
                            Histograms.append(frame_data[0][i])
                            Coordinates.append(frame_data[1][i])
                            Labels.append(frame_data[2][i])
                            Acromatics.append(frame_data[3])
                            Found.append(True)
                        else:
                            Histograms_unk.append(frame_data[0][i])
                            Coordinates_unk.append(frame_data[1][i])
                            Found_unk.append(True)

                    # ID's assignation
                    for i in range(len(Labels)):
                        ID.append(Count[l.index(Labels[i])])
                        Count[l.index(Labels[i])] += 1

                else:

                    print("-------------Record of coordinates---------------")
                    print(np.array(Histograms))
                    print("-------------Record of unkonow coordinates--------")
                    print(np.array(Histograms_unk))
                    print

                    # If there's some historic information, use the matching
                    # for the new detections

                    for i in range(len(frame_data[0])):
                        #print("-------------Current Box:----------------")
                        distances=[]
                        
                        #The new detections will remain as unknown until they
                        #are within a desired region. A bus can be misslabeled
                        #as a car until it completely inside the scene.
                        
                        if frame_data[1][i][2]<frame_width-100:  
                            for j in range(len(Histograms)):

                                h1 = frame_data[0][i][0:12]/(1-frame_data[0][i][0:12])
                                h1 = h1/np.sum(h1)

                                h2 = Histograms[j][0:12]/(1-Histograms[j][0:12])
                                h2 = h2/np.sum(h2)

                                d = 0.3 * Dist(h1,h2) + \
                                    0.7 * abs(frame_data[0][i][12] - 
                                    Histograms[j][12])
                                distances.append(d)

                            j=0
                            while j<(len(Histograms_unk)):

                                h1 = frame_data[0][i][0:12]/(1-frame_data[0][i][0:12])
                                h1 = h1/np.sum(h1)

                                h2 = Histograms_unk[j][0:12]/(1-Histograms_unk[j][0:12])
                                h2 = h2/np.sum(h2)

                                d = 0.3 *Dist(h1,h2) + \
                                    abs(frame_data[0][i][12] - 
                                        0.7 * Histograms_unk[j][12])
                                    
                                distances.append(d)
                                j += 1

                            if distances:
                                
                                min_dist = min(distances)
                                index = np.argmin(np.array(distances))

                                if index < len(Histograms):

                                    center = ((Coordinates[index][0] +
                                                Coordinates[index][2])/2,
                                                (Coordinates[index][1] +
                                                Coordinates[index][3])/2)
                                            
                                    new_center = ((frame_data[1][i][0] +
                                                    frame_data[1][i][2])/2,
                                                    (frame_data[1][i][1] +
                                                    frame_data[1][i][3])/2)

                                    centers_dist=((new_center[0]-center[0])**2 +
                                                (new_center[1]-center[1])**2)**(1/2)

                                    if (min_dist<4)and(centers_dist<100):
                                        Histograms[index]=frame_data[0][i]
                                        Coordinates[index]=frame_data[1][i]
                                        Labels[index]=frame_data[2][i]
                                        Found[index]=True
                                    else:
                                        print("New Vehicle")
                                        Histograms.append(frame_data[0][i])
                                        Coordinates.append(frame_data[1][i])
                                        Labels.append(frame_data[2][i])
                                        Found.append(True)
                                        ID.append(Count[l.index(frame_data[2][i])])
                                        Count[l.index(frame_data[2][i])]+=1
                                else:
                                    index = index-len(Histograms)
                                    center = ((Coordinates_unk[index][0] + \
                                                Coordinates_unk[index][2])/2,
                                                (Coordinates_unk[index][1] + \
                                                Coordinates_unk[index][3])/2)
                                            
                                    new_center=((frame_data[1][i][0]
                                                +frame_data[1][i][2])/2,
                                                (frame_data[1][i][1] + 
                                                frame_data[1][i][3])/2)

                                    centers_dist=((new_center[0]-center[0])**2 +\ 
                                                    (new_center[1]-center[1])**2)**(1/2)

                                    if (min_dist<4)and(centers_dist<100):
                                        print(index)
                                        Histograms_unk.pop(index)
                                        Coordinates_unk.pop(index)
                                        Found_unk.pop(index)

                                    print("New Vehicle")
                                    Histograms.append(frame_data[0][i])
                                    Coordinates.append(frame_data[1][i])
                                    Labels.append(frame_data[2][i])
                                    Found.append(True)
                                    ID.append(Count[l.index(frame_data[2][i])])
                                    Count[l.index(frame_data[2][i])]+=1

                        else:
                            for j in range(len(Histograms_unk)):
                                d = Dist(frame_data[0][i][0:12],
                                        Histograms_unk[j][0:12]) + \
                                        abs(frame_data[0][i][12] - \
                                        Histograms_unk[j][12])
                                        
                                distances.append(d)

                            if distances:
                                min_dist=min(distances)
                                index=np.argmin(np.array(distances))

                                center = ((Coordinates_unk[index][0] + \
                                            Coordinates_unk[index][2])/2,
                                            (Coordinates_unk[index][1] + \
                                            Coordinates_unk[index][3])/2)
                                            
                                new_center = ((frame_data[1][i][0] +
                                                frame_data[1][i][2])/2,
                                                (frame_data[1][i][1] +
                                                frame_data[1][i][3])/2)

                                centers_dist = ((new_center[0]-center[0])**2 + \
                                                (new_center[1]-center[1])**2)**(1/2)

                                if (min_dist<4)and(centers_dist<100):
                                    Histograms_unk[index]=frame_data[0][i]
                                    Coordinates_unk[index]=frame_data[1][i]
                                    Found_unk[index]=True

                                else:
                                    print("New Vehicle")
                                    Histograms_unk.append(frame_data[0][i])
                                    Coordinates_unk.append(frame_data[1][i])
                                    Found_unk.append(True)

                print("-------------Found Vehicles---------------")
                print(np.array(Found))
                print(np.array(Found_unk))


                #-------- Finding Lost Vehicles --------------------------------
                i=0
                while i<(len(Found)):

                    pt = Coordinates[i]
                    x1 = np.int(pt[0]); y1 = np.int(pt[1]); 
                    x2 = np.int(pt[2]); y2 = np.int(pt[3])
                    r = (x1,y1,x2,y2)

                    if Found[i]==True and x1>100:
                        Found[i]=False
                        i+=1

                    else:

                        u,v = MQR.Flujo_caja(gray,prevgray,r)
                        new_r = (x1 + u, y1 + v, x2 + u, y2 + v)
                        
                        new_hist = Histograma(im,new_r,matrices)

                        Histograms[i] = new_hist
                        pt[0] = pt[0]+u ; pt[1] = pt[1]+v;  
                        pt[2] = pt[2]+u ; pt[3] = pt[3]+v;
                        x1 = np.int(pt[0]); y1 = np.int(pt[1]); 
                        x2 = np.int(pt[2]); y2 = np.int(pt[3])
                        Coordinates[i]=pt

                        if x1<50:
                            Histograms.pop(i)
                            Coordinates.pop(i)
                            Labels.pop(i)
                            Found.pop(i)
                            ID.pop(i)
                        else:
                            i+=1

                #-------- Deleting Overlaping Vehicles  -----------------------

                k=0;
                while k < len(Histograms):
                    n=k+1
                    while n < len(Histograms):
                        ovr, ID_max = overlap_ID(Coordinates[k],Coordinates[n],\
                        ID[k],ID[n])

                        if (ovr>0.7):
                            idx=ID.index(ID_max)
                            Histograms.pop(idx)
                            Coordinates.pop(idx)
                            Count[l.index(Labels[idx])]-=1
                            Labels.pop(idx)
                            ID.pop(idx)
                            Found.pop(idx)

                        else:
                            n+=1
                    k+=1

                #-------- Drawing Bounding Boxes  -----------------------------
                i=0
                while i<(len(Found)):
                    pt = Coordinates[i]
                    x1 = np.int(pt[0]); y1 = np.int(pt[1]); 
                    x2 = np.int(pt[2]); y2 = np.int(pt[3])

                    if Found[i]==True and x1>100:

                        display_txt = '%s: %.2f'%(Labels[i],ID[i])
                        cv2.rectangle(image,(x1,y1),(x2,y2),((0, 255, 0)),2)
                        draw_str(image,(x1,y1),display_txt)

                    else:
                        if Labels[i]=='car':
                            display_txt = '%s: %.2f'%(Labels[i],ID[i])
                            cv2.rectangle(image,(x1,y1),(x2,y2),((0,255,0)),2)
                            draw_str(image,(x1,y1),display_txt)
                        elif Labels[i]=='bus':
                            display_txt = '%s: %.2f'%(Labels[i],ID[i])
                            cv2.rectangle(image,(x1,y1),(x2,y2),((255,0,0)),2)
                            draw_str(image,(x1,y1),display_txt)
                        elif Labels[i]=='motorbike':
                            display_txt = '%s: %.2f'%(Labels[i],ID[i])
                            cv2.rectangle(image,(x1,y1),(x2,y2),((122,122,255)),2)
                            draw_str(image,(x1,y1),display_txt)
                        else:
                            display_txt = '%s: %.2f'%(Labels[i],ID[i])
                            cv2.rectangle(image,(x1,y1),(x2,y2),((255,0,255)),2)
                            draw_str(image,(x1,y1),display_txt)

                    i+=1

                mag,ang=cv2.cartToPolar(flow[:,:,0], flow[:,:,1], angleInDegrees=True)
                cv2.normalize(mag,mag,0,1,cv2.NORM_MINMAX)
                hsv[...,0] = ang
                hsv[...,1] = mag
                hsv[...,2] = np.ones((frame_height, frame_width), np.float32)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


                out.write(image)
                prevgray = gray
                
                n_frame+=1
            else:
                n_frame+=1

        else:
            break


    out.release()
    print(Count)

if __name__=='__main__':
    main()
