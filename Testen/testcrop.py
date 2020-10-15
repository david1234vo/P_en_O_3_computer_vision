import time
import numpy as np
import cv2
import random
pos=None
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img_shape = img.shape
    y_img = img_shape[0]
    x_img = img_shape[1]
    res = img_shape[0] / img_shape[1]
    if not pos:
        pos_y = int(y_img / 2)
        pos_x = int(x_img / 2)
        pos=(pos_x,pos_y)
    else:
        pos_y = int(pos[0])
        pos_x = int(pos[1])
    hoog = int(104)
    breed = int(res * hoog)

    y_min = pos_y - hoog
    y_max = pos_y + hoog
    x_min = pos_x - breed
    x_max = pos_x + breed

    cv2.circle(img, pos, 5, (0, 0, 255), -1)

    crop = img[x_min:x_max, y_min:y_max]

    cv2.imshow('crop', crop)
    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


