import cv2
import numpy as np

cmd=cv2.VideoCapture(0)
while True:
   _,frame=cmd.read()
   gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
   cv2.imshow("frame",gray)
   key=cv2.waitKey(1)
   if key==27:
     break
cmd.release()
cv2.destroyAllWindows()
