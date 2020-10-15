import time
import numpy as np
import cv2
import random

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
upper_body = cv2.CascadeClassifier('haarcascade_upperbody.xml')

# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 80

# User message
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (255, 255, 255)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing MASK"
not_weared_mask = "Please wear MASK to defeat Corona"


def crop_image(img, pos=None):
    img_shape = img.shape
    y_img = img_shape[0]
    x_img = img_shape[1]
    res = img_shape[0] / img_shape[1]
    if not pos:
        pos_y = int(img_shape[0] / 2)
        pos_x = int(img_shape[1] / 2)
    else:
        pos_y = int(pos[0])
        pos_x = int(pos[1])
    breed = int(104)
    hoog = int(res * breed)
    print(pos_y, pos_x)
    y_min = pos_y - hoog
    print(y_min)
    if y_min < 1:
        y_min = 1
    y_max = pos_y + hoog
    print(y_max)
    if y_max > y_img:
        y_min = y_img
    x_min = pos_x - breed
    if x_min < 1:
        x_min = 1
    x_max = pos_x + breed
    if x_max > x_img:
        x_max = x_img

    crop = img[y_min:y_max, x_min:x_max]
    print(crop.shape)

    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
    return resized


def position_face(img):
    img_shape = img.shape
    y_img = img_shape[0]
    x_img = img_shape[1]
    # Convert to grayscale
    grijs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    gezicht = face_cascade.detectMultiScale(grijs, 1.1, 4)
    if isinstance(gezicht, np.ndarray):
        pos_face = (gezicht[0][0], gezicht[0][1])
    else:
        pos_face = (int(x_img / 2), int(y_img / 2))
    if not 0 < pos_face[0] < y_img or not 0 < pos_face[1] < x_img:
        pos_face = (int(x_img / 2), int(y_img / 2))
    print(pos_face)
    for (x, y, w, h) in gezicht:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.circle(img, pos_face, 5, (0, 0, 255), -1)
    # Display
    cv2.imshow('img', img)
    return pos_face


# Read video
cap = cv2.VideoCapture(0)  # op een of andere manier data van kinect binnenkrijgen
a = True
while 1:
    # Get individual frame
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    # binnenkrijgen van diepte kinect
    depth = 2600

    # binnenkrijgen locatie van gezicht

    # als verder dan 2,5m inzoomen
    if depth > 2500:
        position = position_face(img)
        img = crop_image(img, position)

    # Convert Image into gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert image in black and white
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)

    # cv2.imshow('black_and_white', black_and_white)

    # detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Face prediction for black and white
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)

    if (len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img, "No face found...", org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    elif (len(faces) == 0 and len(faces_bw) == 1):
        # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
        cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    else:
        # Draw rectangle on gace
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect lips counters
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)

        # Face detected but Lips not detected which means person is wearing mask
        if (len(mouth_rects) == 0):
            cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
        else:
            for (mx, my, mw, mh) in mouth_rects:

                if (y < my < y + h):
                    # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                    # person is not waring mask
                    cv2.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness,
                                cv2.LINE_AA)

                    # cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                    break

    # Show frame with results
    cv2.imshow('Mask Detection', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release video
cap.release()
time.sleep(10)
cv2.destroyAllWindows()
