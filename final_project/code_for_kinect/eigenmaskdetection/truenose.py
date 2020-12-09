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
    faces1 = functions.detect_faces(img)
    faces, eyes = functions.true_eyes_and_faces(img)
    mouths = functions.detect_mouths(img)
    noses = functions.detect_noses(img)
    functions.add_rectangle(img1, mouths)
    functions.add_rectangle(img, faces)  # blauw
    functions.add_rectangle(img, eyes, (0, 0, 255))  # rood
    true_mouths, faces_mouths_noses = functions.best_mouth_to_face(mouths, faces)
    true_noses = functions.best_nose_to_face(noses, faces, faces_mouths_noses)
    functions.add_rectangle(img, true_noses, (0, 255, 255))  # geel
    functions.add_rectangle(img, true_mouths, (0, 255, 0))  # groen
    pos = []
    for face in faces:
        middle_pos = functions.middle_position(face)
        pos.append(middle_pos)
    for (x, y) in pos:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imshow('img', img)
    cv2.imshow('img1', img1)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
