import cv2
import numpy as np
import mediapipe as mp
import time

mpholistic=mp.solutions.holistic 
hmodel=mpholistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5 
)
draw=mp.solutions.drawing_utils
capture=cv2.VideoCapture(0)
#FPS (Frames Per Second) means how many images (frames) your program processes or displays each second
while capture.isOpened():
    ret,frame=capture.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue
    start=time.time()
    rgbframe=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    rgbframe.flags.writeable=False
    results=hmodel.process(rgbframe)
    rgbframe.flags.writeable=True
    bgrframe=cv2.cvtColor(rgbframe,cv2.COLOR_RGB2BGR)
    #Draw landmarks
    draw.draw_landmarks(bgrframe,results.face_landmarks,mpholistic.FACEMESH_TESSELATION)
    draw.draw_landmarks(bgrframe,results.left_hand_landmarks,mpholistic.HAND_CONNECTIONS)
    draw.draw_landmarks(bgrframe,results.right_hand_landmarks,mpholistic.HAND_CONNECTIONS)
    end=time.time()
    totalTime=end-start
    fps=1/totalTime
    cv2.putText(bgrframe,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("Mediapipe Holistic",bgrframe)
    key=cv2.waitKey(1)
    if key==27:
        break
capture.release()
cv2.destroyAllWindows()    