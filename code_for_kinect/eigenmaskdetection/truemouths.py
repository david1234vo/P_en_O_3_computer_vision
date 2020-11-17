import functions
import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    img1 = img.copy()
    faces, eyes = functions.true_eyes_and_faces(img)
    mouths = functions.detect_mouths(img)
    functions.add_rectangle(img, faces)  # blauw
    functions.add_rectangle(img, eyes, (0, 0, 255))  # rood
    true_mouths = functions.best_mouth_to_face(mouths,faces)
    functions.add_rectangle(img, true_mouths, (0, 255, 0))  # groen
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
