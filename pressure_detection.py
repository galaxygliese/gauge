#-*- coding:utf-8 -*-

from sklearn.neighbors import LocalOutlierFactor
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
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

#Detect Gauge
def Gauge(show=True):

    LENGTH = 30
    Center = deque(maxlen=LENGTH)
    cap = cv2.VideoCapture(0)
    clf = LocalOutlierFactor(n_neighbors=2)
   
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
                 edges = cv2.Canny(gray[_x:_x+_w, _y:_y+_h],50,150,apertureSize = 3)
                 minLineLength = 500
                 maxLineGap = 10
                 lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
                 print(lines)
                 if lines is not None:
                    for x1,y1,x2,y2 in lines[0]:
                        cv2.line(gray,(x1,y1),(x2,y2),(0,255,0),2)
 
        cv2.imshow('frame', gray)	
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    if show:
       cap.release()
       cv2.destroyAllWindows()


if __name__ == '__main__':
   Gauge()

