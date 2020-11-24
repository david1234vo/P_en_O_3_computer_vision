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
    img = functions.resize_image(img)
    print(img.shape)
    img = cv2.flip(img, 1)

    fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))
    cv2.putText(img, str(fps), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    cv2.imshow('img', img)

    # cv2.imshow('img1', img1)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
