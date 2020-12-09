import functions
import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    timer = cv2.getTickCount()
    _, img = cap.read()
    img = cv2.flip(img, 1)
    faces, eyes = functions.true_eyes_and_faces(img)
    mouths = functions.detect_mouths(img)
    noses = functions.detect_noses(img)
    functions.add_rectangle(img, faces)
    true_mouths, faces_mouths_noses = functions.best_mouth_to_face(mouths, faces)
    functions.add_rectangle(img, true_mouths, (0, 255, 0))  # groen
    true_noses = functions.best_nose_to_face(noses, faces, faces_mouths_noses)
    functions.add_rectangle(img, true_noses, (0, 255, 255))  # geel
    functions.mask_due_haar(img, faces_mouths_noses)
    fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))
    cv2.putText(img, str(fps), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
