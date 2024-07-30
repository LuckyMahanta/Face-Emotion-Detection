import cv2
from keras.models import model_from_json
import numpy as np
json_file= open("emotiondetector.json","r")
model_json=json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")
hear_file= cv2.data.haarcascades+ "haarcascade_frontalface_default.xml"
#to detect our face from camera
face_cascade= cv2.CascadeClassifier(hear_file)

def extract_features(image):
    feature= np.array(image)
    feature= feature.reshape(1,48,48,1)
    return feature/255.0

webcam = cv2.VideoCapture(0)
labels= {0:"angry", 1:"disgust", 2:"fear", 3:"happy", 4:"neutral", 5:"sad", 6:"surprise"}
while True:
    i,im= webcam.read()
    gray= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(im, 1.3, 5)
    try:
        for (p,q,r,s) in faces:
            image= gray[q: q+s, p:p+r]
            cv2.rectangle(im, (p,q),(p+r, q+s),(255,0,0),2)
            image= cv2.resize(image,(48,48))
            #resized the image to 48x48
            img = extract_features(image)
            pred= model.predict(img)
            prediction_label= labels[pred.argmax()]
            cv2.putText(im,' %s' %(prediction_label),(p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255))
            print(prediction_label," is the emotion as this point of time.")
        cv2.imshow("Output", im)
        #cv2.waitKey(27)
        if cv2.waitKey(1) & 0xFF==ord('q') :
            break
        
    except cv2.error:
        pass
            