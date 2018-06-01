#-*- coding:utf-8 -*-

from sklearn.neighbors import LocalOutlierFactor
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2


#FInd Biggest size of the shape
def max_shape(array, _type ):
    point = np.array(array)
    if _type in ['area', 'rect']:
       areas = []
       for i,rect in enumerate(point):
           areas.append(rect[2]*rect[3])
       return point[areas.index(max(areas))]
    elif _type in ['r', 'radius']:
         #input array included radius in third index
         max_radius = np.max([r[2] for r in point])
         return [r for r in point if r[2]==max_radius][0]

#Draw vector
def drawAxis(img, start_pt, vec, colour, length):
    CV_AA = 20

    end_pt = (int(start_pt[0] + length * vec[0]), int(start_pt[1] + length * vec[1]))

    cv2.circle(img, (int(start_pt[0]), int(start_pt[1])), 5, colour, 1)
    cv2.line(img, (int(start_pt[0]), int(start_pt[1])), end_pt, colour, 1, CV_AA);

    angle = math.atan2(vec[1], vec[0])

    qx0 = int(end_pt[0] - 9 * math.cos(angle + math.pi / 4))
    qy0 = int(end_pt[1] - 9 * math.sin(angle + math.pi / 4))
    cv2.line(img, end_pt, (qx0, qy0), colour, 1, CV_AA)

    qx1 = int(end_pt[0] - 9 * math.cos(angle - math.pi / 4))
    qy1 = int(end_pt[1] - 9 * math.sin(angle - math.pi / 4))
    cv2.line(img, end_pt, (qx1, qy1), colour, 1, CV_AA)


#Detect pin vector  (x, y, r) -> coordinate of the center 
def pointing_line(gray, x, y, R, perimeter):
    epsilon = 0.4  
    r = int(epsilon * R)

    img = gray[x-r:x+r, y-r:y+r]
    #thresh
    ret,thresh = cv2.threshold(img,50,255, cv2.THRESH_BINARY)
    #if thresh is not None:
    #conv
       #kernel = np.ones((9,9),np.uint8)
       #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    #countor
    _, contours, hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if contours != []:
       perimeter = []
    #max perimeter
       for cnt in contours[1:]:
           perimeter.append(cv2.arcLength(cnt,True))
    if perimeter != []:          
       maxindex= perimeter.index(max(perimeter))
    
       X = np.array(contours[maxindex+1], dtype=np.float).reshape((contours[maxindex+1].shape[0], contours[maxindex+1].shape[2]))
       mean, eigenvectors = cv2.PCACompute(X, mean=np.array([], dtype=np.float), maxComponents=1)
       vec = (-eigenvectors[0][0], -eigenvectors[0][1])
       drawAxis(gray, (x,y), vec, (0, 255, 0), R)
       

#Detect Gauge
def Gauge(show=True):

    LENGTH = 30
    Center = deque(maxlen=LENGTH)
    cap = cv2.VideoCapture(0)
    clf = LocalOutlierFactor(n_neighbors=2)
    perimeter = []
  
    while (cap.isOpened()):
       
        #camera
        ret, frame = cap.read()
        Cascade = cv2.CascadeClassifier('./only_pressure_gauge14_80.xml')

        #gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detection
        point = Cascade.detectMultiScale(gray, 1.1, 3)
        if len(point) > 0:
           rect = max_shape(point, 'area')

           #max square  _x, _y --> coordinate of the detected rectangle
           _x, _y = int(rect[0]), int(rect[1])
           _w, _h = int(rect[2]), int(rect[3])          
           detection = gray[_y:_y+_h, _x:_x+_w]
            
           #detect circle x, y, X, Y --> coordinate of the biggest circle
           circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
           if circles is not None:
              circles = np.round(circles[0, :]).astype("int")
              x, y ,r  = max_shape(circles, 'radius')
              Center.append([x,y])
              center = np.array(Center)
           #find LOF and find Center
              if len(center) > LENGTH-1:                 
                 predict = clf.fit_predict(center) 
                 true_center = center[predict==1]   
                 X, Y = np.mean(true_center[:,0]), np.mean(true_center[:,1])
              else:
                 X, Y = np.mean(center[:,0]), np.mean(center[:,1])
              X, Y = int(X), int(Y)
              if X in range(_x,_x+_w) and Y in range(_y,_y+_h):             
                 cv2.circle(gray, (X, Y), r, (0, 255, 0), 4)
                 cv2.rectangle(gray, (X - 5, Y - 5), (X + 5, Y + 5), (0, 128, 255), -1)
           #draw Line
                 pointing_line(gray, X, Y, r, perimeter)      

              cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    if show:
       cap.release()
       cv2.destroyAllWindows()


if __name__ == '__main__':
   Gauge()

