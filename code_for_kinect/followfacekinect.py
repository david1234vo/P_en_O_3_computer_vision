import time
import numpy as np
import cv2
import random

face_cascade = cv2.CascadeClassifier('C:/Users/lucas/PycharmProjects/P_en_O_3_computer_vision/Cascades/haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier(    'C:/Users/Jasper/PycharmProjects/PO3/P_en_O_3_computer_vision/Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(
    'C:/Users/Jasper/PycharmProjects/PO3/P_en_O_3_computer_vision/Cascades/haarcascade_mcs_mouth.xml')
upper_body = cv2.CascadeClassifier('haarcascade_upperbody.xml')

bw_threshold = 80

def get_corners(rect):
    (x, y, w, h) = rect
    corners = np.array([x, y, x + w, y + h])
    return corners


def crop_image(original, face):
    original_shape = original.shape
    x_original = original_shape[0]
    y_original = original_shape[1]
    edge = int((y_original - x_original) / 2)
    res = original_shape[0] / original_shape[1]

    if range == [0, 0, x_original, y_original]:
        crop = original[0:x_original, edge:y_original - edge]
    else:
        (y_min, x_min, y_max, x_max) = get_corners(face)

        crop = original[x_min:x_max, y_min:y_max]

    scale_percent = 100  # percent of original size
    width_resize = int(y_original * scale_percent / 100)
    height_resize = int(x_original * scale_percent / 100)
    dim = (width_resize, height_resize)
    resized = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)

    return resized


def position_faces(original):
    original_shape = original.shape
    x_original = original_shape[0]
    y_original = original_shape[1]

    # Convert to grayscale
    gray_pos = cv2.cvtColor(original.astype('uint8'), cv2.COLOR_GRAY2BGR)
    # Detect the faces
    faces_pos = face_cascade.detectMultiScale(original, 1.1, 4)
    if isinstance(faces_pos, np.ndarray):
        range_face = faces_pos.tolist()
    else:
        range_face = [[0, 0, x_original, y_original]]

    # for (x_cv, y_cv, w, h) in faces_pos:
    #     cv2.rectangle(original, (x_cv, y_cv), (x_cv + w, y_cv + h), (255, 0, 0), 2)
    # cv2.imshow('img', original)

    return range_face


def zoomimg(original_img):

    returnimgs = []
    img = cv2.flip(original_img, 1)
    faces = position_faces(img)
    for face in faces:
        zoom = crop_image(img, face)
        returnimgs.append(zoom)
    return returnimgs
