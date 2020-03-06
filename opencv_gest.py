import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from keras.models import  load_model
import os
os.chdir('/users/arunkarthik/downloads/sign_lang')

loaded_model = load_model('gesture_rec.h5')

cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    flip = cv2.flip(frame,2)
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    roi = frame[y1:y2, x1:x2]

    roi = cv2.resize(roi, (28, 28),cv2.INTER_AREA)
    cvt = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
    cv2.imshow('cvt',cvt)
    val = np.resize(cvt,(-1,28,28))
    print(val.shape)
    result = loaded_model.predict(val.reshape(val.shape[0],28,28,1))
    
    predition = {
        'A':result[0][0],
        'b': result[0][1],
        'c': result[0][2],
        'd': result[0][3],
        'e': result[0][4],
    }
    prediction = sorted(predition.items(), key=operator.itemgetter(1), reverse=True)

    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 12, (0, 255, 255), 12)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(40) == 27:
        break
cv2.destroyAllWindows()
cap.release()