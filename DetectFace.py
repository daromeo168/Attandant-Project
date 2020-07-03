import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier('src\cascade\data\haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('src/cascade/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C://Users//MSI//Desktop//Attendant Project//src//trainner.yml")

labels = {"Person_name": 1}
with open("src/labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        #print(x,y,w,h)

        #region of interest
        #capturing face in a square area of color gray
        roi_gray = gray[y:y+h, x:x+h]
        #same with frame
        roi_color = frame[y:y + h, x:x + h]
        #Recognizer
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            name=labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


        img_item = "test-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0) #BGR color
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x, y), (width, height), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()