import cv2
import sys
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)

i=1;

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags= cv2.CASCADE_SCALE_IMAGE
    )
    
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        i=i+1;
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print (faces)

        #face extraction from the video
        face_region = frame[y:y+h, x:x+w]
        resized_image = cv2.resize(face_region, (200, 200)) 
        cv2.imshow("Face", resized_image)
        cv2.imwrite('recode/image'+str(i)+'.png', resized_image)

        #for eye detection 
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_color)
        for (ex,ey,ew,eh) in eyes:
             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

