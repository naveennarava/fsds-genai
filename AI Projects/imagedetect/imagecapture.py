import numpy as np
import cv2

cmd=cv2.VideoCapture(0)
while True:
   _,frame=cmd.read()
   cv2.imshow("frame",frame)
   lowred= np.array([161,155,84])
   highred=np.array([179,255,255]) 
   redmask=cv2.inRange(frame,lowred,highred)
   redoutput=cv2.bitwise_and(frame,frame,mask=redmask)
   cv2.imshow("Frame",frame)
   cv2.imshow("Redmask",redmask)
   key=cv2.waitKey(1)
   if key==27:
    break
cmd.release() 
cv2.destroyAllWindows

