import time
import numpy as np
import cv2
import random

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
upper_body = cv2.CascadeClassifier('haarcascade_upperbody.xml')

bw_threshold = 80


def crop_image(original, face):
    original_shape = original.shape
    x_original = original_shape[0]
    y_original = original_shape[1]
    edge = int((y_original - x_original) / 2)
    res = original_shape[0] / original_shape[1]

    if range == [0, 0, x_original, y_original]:
        crop = img[0:x_original, edge:y_original - edge]
    else:
        pos_y = int(face[0])
        pos_x = int(face[1])

        width = int(face[2])
        height = int(face[3])
        y_min = pos_y
        y_max = pos_y + width
        x_min = pos_x
        x_max = pos_x + height

        crop = img[x_min:x_max, y_min:y_max]

    scale_percent = 100  # percent of original size
    width_resize = int(x_original * scale_percent / 100)
    height_resize = int(x_original * scale_percent / 100)
    dim = (width_resize, height_resize)
    resized = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)

    return resized


def position_faces(original):
    original_shape = original.shape
    x_original = original_shape[0]
    y_original = original_shape[1]

    # Convert to grayscale
    gray_pos = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces_pos = face_cascade.detectMultiScale(gray_pos, 1.1, 4)
    if isinstance(faces_pos, np.ndarray):
        range_face = faces_pos.tolist()
    else:
        range_face = [[0, 0, x_original, y_original]]

    for (x_cv, y_cv, w, h) in faces_pos:
        cv2.rectangle(img, (x_cv, y_cv), (x_cv + w, y_cv + h), (255, 0, 0), 2)
    cv2.imshow('img', original)

    return range_face


cap = cv2.VideoCapture(0)  # Get video from kinect

while 1:
    # Get individual frame
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    faces = position_faces(img)
    for face in faces:
        zoom = crop_image(img, face)
        cv2.imshow('Mask Detection', zoom)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

# Release video
cap.release()
# time.sleep(10)
cv2.destroyAllWindows()
