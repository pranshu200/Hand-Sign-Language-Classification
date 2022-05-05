#%%
#Import required libraries:
try:
    import sys,os
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import warnings
    import matplotlib.pyplot as plt
    import cv2
    from keras.preprocessing import image
    import time
    print("----Libraries Loaded----")
except:
    print("----Libraries Not Loaded----")

#%%
#Load saved model
from tensorflow.keras.models import load_model
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

load_model = model_from_json(open('C:/PRANSHU/PROJECTS/Hand-Sign-Language-Classification/Saved Model/HandSignModel1.json','r').read())  
load_model.load_weights('C:/PRANSHU/PROJECTS/Hand-Sign-Language-Classification/Saved Model/HandSignModel1_Weights.h5')
print("------Saved Model Loaded------")
#%%
cap=cv2.VideoCapture(0)
cap.set(3,600)
cap.set(4,600)
while True:
    ret, frame = cap.read()
    cv2image = cv2.flip(frame, 1)
    x1 = int(0.5*frame.shape[1])
    y1 = 60
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    cv2image = cv2image[y1:y2, x1:x2]
    roi_gray=cv2.resize(cv2image,(64,64))  
    img_pixels = image.img_to_array(roi_gray)  
    img_pixels = np.expand_dims(img_pixels, axis = 0)  
    img_pixels /= 255  
    pred = load_model.predict(img_pixels)
    classes= {1: 'Bye', 2: 'CallMe', 3: 'God', 4: 'GoodLuck', 5: 'Hello', 6: 'I', 7:'ILoveYou', 8: 'No', 9: 'Ok', 10: 'Peace', 11: 'Smile', 12: 'ThumbsDown', 13: 'ThumbsUp', 14: 'Water', 15: 'Yes'}
    print(classes[np.argmax(pred)+1])
    cv2.imshow("Webcam recording", frame)
    time.sleep(1)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# %%
