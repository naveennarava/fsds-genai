import mediapipe as mp
import cv2
import numpy as np
import PIL.Image as Image
import urllib
import matplotlib.pyplot as plt


#img1='D:\AI\cam.jpg'
img1='https://www.sisinternational.com/wp-content/uploads/2025/08/Photography-2.jpg'
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils
def load_imagefromurl(img1):
    #req=cv2.imread(img1)#frm local path
    req=urllib.request.urlopen(img1)#read image from url
    #This is useful because OpenCV’s cv2.imdecode() expects a NumPy array of bytes, not a Python bytes object.
    arr=np.array(bytearray(req.read()),dtype=np.uint8)#Reads raw binary data from a file-like object Converts that binary data into a mutable sequence of bytes.
    #Now OpenCV’s decoder reads that 1D byte array, decompresses it (like unzipping), and constructs a 2D (grayscale) or 3D (color) array of pixel values.
    arr=cv2.imdecode(arr,-1)#Decode an image from a buffer in memory. now it cnverts into 2d image
    arr=cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)
    return arr 
img1 = load_imagefromurl(img1) 
    
objectron = mp_objectron.Objectron(
    static_image_mode=True,
    max_num_objects=5,
    min_detection_confidence=0.2,
    model_name='Camera') 

results=objectron.process(img1)  
if not results.detected_objects:
    print("No object detected")
copyimage= img1.copy()   
for detected_object in results.detected_objects:
    mp_drawing.draw_landmarks(copyimage, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
    mp_drawing.draw_axis(copyimage, detected_object.rotation, detected_object.translation)    
fig,ax=plt.subplots(figsize=(10,10))
ax.imshow(copyimage)           
ax.axis('off')
plt.show()
    