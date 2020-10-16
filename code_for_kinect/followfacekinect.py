import time
import numpy as np
import cv2
import random

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
upper_body = cv2.CascadeClassifier('haarcascade_upperbody.xml')

bw_threshold = 80


def crop_image(img, range):
    img_shape = img.shape
    x_img = img_shape[0]
    y_img = img_shape[1]
    edge = int((y_img-x_img)/2)
    res = img_shape[0] / img_shape[1]

    if range == [0,0,x_img,y_img]:
        crop = img[0:x_img,edge:y_img-edge]
    else:
        pos_y = int(range[0])
        pos_x = int(range[1])

        width = int(range[2])
        height = int(range[3])
        y_min = pos_y
        y_max = pos_y + width
        x_min = pos_x
        x_max = pos_x + height

        crop = img[x_min:x_max, y_min:y_max]

    scale_percent = 100  # percent of original size
    width_resize = int(x_img * scale_percent / 100)
    height_resize = int(x_img * scale_percent / 100)
    dim = (width_resize, height_resize)
    resized = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)

    return resized


def position_face(img):
    img_shape = img.shape
    x_img = img_shape[0]
    y_img = img_shape[1]

    # Convert to grayscale
    gray_pos = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces_pos = face_cascade.detectMultiScale(gray_pos, 1.1, 4)

    if isinstance(faces_pos, np.ndarray):
        range_face = list(faces_pos[0])
        pos_face = (faces_pos[0][1], faces_pos[0][0])
        mid_face = (pos_face[0] + int(faces_pos[0][3] / 2), pos_face[1] + int(faces_pos[0][2] / 2))
    else:
        pos_face = (int(x_img / 2), int(y_img / 2))
        mid_face = pos_face
        range_face=list([0,0,x_img,y_img])
    if not 0 <= pos_face[0] <= x_img or not 0 <= pos_face[1] <= y_img:
        mid_face = (int(x_img / 2), int(y_img / 2))

    #for (x_cv, y_cv, w, h) in faces_pos:
        #cv2.rectangle(img, (x_cv, y_cv), (x_cv + w, y_cv + h), (255, 0, 0), 2)
    #cv2.circle(img, (range_face[0],range_face[1]), 5, (0, 0, 255), -1)
    # Display
    cv2.imshow('img', img)

    return range_face


cap = cv2.VideoCapture(0)  # Get video from kinect

while 1:
    # Get individual frame
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    range=position_face(img)
    crop=crop_image(img,range)
    # Show frame with results
    cv2.imshow('Mask Detection', crop)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release video
cap.release()
# time.sleep(10)
cv2.destroyAllWindows()
