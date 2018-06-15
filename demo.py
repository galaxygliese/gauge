#-*- coding:utf-8 -*-

from sklearn.neighbors import LocalOutlierFactor
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import timeit
import cv2


class detectGauge:

      def __init__(self):
          super(detectGauge).__init__()
          self.cascade_path = "./only_pressure_gauge14_80.xml"
          self.clf = LocalOutlierFactor(n_neighbors=2)
          self.LENGTH = 30 #LOF array length
          self.d = 5 #center rectangle's w and h       
          self.X, self.Y = None, None
          self.Radius = []
          self.Center = deque(maxlen=self.LENGTH)
   
      def max_shape(self, array, _type ):
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
 
         
      def run(self, frame):
          #gray
          self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         
          #cascade
          self.cascade = cv2.CascadeClassifier(self.cascade_path)
          #detection
          self.point = self.cascade.detectMultiScale(self.gray, 1.1, 3)

          if len(self.point) > 0:
           rect = self.max_shape(self.point, 'area')

           #max square  _x, _y --> coordinate of the detected rectangle
           _x, _y = int(rect[0]), int(rect[1])
           _w, _h = int(rect[2]), int(rect[3])          
            
           #detect circle x, y, X, Y --> coordinate of the biggest circle
           circles = cv2.HoughCircles(self.gray, cv2.HOUGH_GRADIENT, 1.2, 100)
           if circles is not None:
              circles = np.round(circles[0, :]).astype("int")
              x, y ,r  = self.max_shape(circles, 'radius')
              self.Center.append([x,y])
              self.Radius.append(r)
              center = np.array(self.Center)
           #find LOF and find Center
              if len(center) > self.LENGTH-1:                 
                 predict = self.clf.fit_predict(center) 
                 true_center = center[predict==1]   
                 X, Y = np.mean(true_center[:,0]), np.mean(true_center[:,1])
              else:
                 X, Y = np.mean(center[:,0]), np.mean(center[:,1])
              if (_x < X < _x+_w) and (_y < Y < _y+_h):                             
                  self.X, self.Y, self.r = int(X), int(Y), int(r)

      def show(self):
          if (self.X, self.Y) != (None, None): 
             cv2.circle(self.gray, (self.X, self.Y), self.r, (0, 255, 0), 4)
             cv2.rectangle(self.gray, (self.X - self.d, self.Y - self.d), (self.X + self.d, self.Y + self.d), (0, 255, 0), -1)
             cv2.imshow('frame', self.gray)
 
              

if __name__ == '__main__':
   from billiard import Process, forking_enable
   def start():
       forking_enable(0) # Is all you need!
       camProcess = Process(target=cam, args=(0,))
       camProcess.start()
   def cam(cam_id):
       #initial time
       time = 15
       start = timeit.default_timer()
       cap = cv2.VideoCapture(cam_id)
       ret, frame = cap.read()
       init_detect = detectGauge()
       detect = detectGauge() 

       #center coordinate
       X, Y, R = None, None, None
       while(cap.isOpened()):
           ret, frame = cap.read()
           if ret:
              stop = timeit.default_timer()
              if stop-start <= time :
                init_detect.run(frame)
                init_detect.show()
                #get final parameters
                X, Y = init_detect.X, init_detect.Y
                R = max(init_detect.Radius) if any(init_detect.Radius) else 0
              else:
                detect.run(frame[Y-R:Y+R, X-R:X+R])
                detect.show()
           k = cv2.waitKey(10)
           if k == 27:
              break
           if k == ord('q'):
              break
       cap.release()
       cv2.destroyAllWindows()

   start()
   cam(0)

