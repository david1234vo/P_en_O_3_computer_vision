from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2

face_cascade = cv2.CascadeClassifier(
    'C:/Users/david/PycharmProjects/peno/P_en_O_3_computer_vision/Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'C:/Users/david/PycharmProjects/peno/P_en_O_3_computer_vision/Cascades/haarcascade_mcs_eyepair_big.xml')




#cap = cv2.VideoCapture(0)


def detect_persons(img):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog.detectMultiScale(img, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)
    rects = np.array([[4.8 * x, 4.8 * y, 4.8 * (w), 4.8 * (h)] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.3)
    rects = [[int(sub_element) for sub_element in element] for element in rects]
    # print(rects, pick)
    return rects


def add_rectangle(img, array, type_rect, color=(255, 0, 0)):
    if type_rect == 0:
        for (xm, ym, xM, yM) in array:
            cv2.rectangle(img, (xm, ym), (xM, yM), color, 2)
    if type_rect == 1:
        for (x, y, w, h) in array:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def get_corners(rect):
    (x, y, w, h) = rect
    corners = (x, y, x + w, y + h)
    return corners


def crop_image(original, rect):
    original_shape = original.shape
    x_original = original_shape[0]
    y_original = original_shape[1]
    edge = int((y_original - x_original) / 2)
    res = original_shape[0] / original_shape[1]
    (y, x, h, w) = rect
    crop = original[x:x+w, y:y+h] # w//2 voor hogere fps, maar iets minder gezichten herkennen
    img = crop.copy()
    return img


def within_face(rect, face):
    (xmin, ymin, xmax, ymax) = get_corners(rect)
    (xMin, yMin, xMax, yMax) = get_corners(face)
    if xmin >= xMin:
        if xmax <= xMax:
            if ymin >= yMin:
                if ymax <= yMax:
                    return True
    return False


def all_rectangles_within_face(array, faces):
    new_arraylst = []
    for face in faces:
        for i in range(0, len(array)):
            rect = array[i]
            if within_face(rect, face):
                new_arraylst.append(rect)
    new_array = np.array(new_arraylst)
    return new_array


def all_faces_with_a_pair(eyes, faces):
    new_faceslst = []
    for face in faces:
        a = 0
        for pair in eyes:
            if within_face(pair, face):
                a += 1
        if a != 0:
            new_faceslst.append(face)
    new_faces = np.array(new_faceslst)
    return new_faces


def true_eyes_and_faces(img):
    faces = detect_faces(img)
    eyes = detect_eyes(img)
    true_faces = all_faces_with_a_pair(eyes, faces)
    true_eyes = all_rectangles_within_face(eyes, faces)
    return true_faces, true_eyes


def detect_faces(img):  # als gray zwz berekend wordt kan invoer ook gray zijn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces


def detect_eyes(img):  # als gray zwz berekend wordt kan invoer ook gray zijn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    return eyes


def resize_image(img, scale_percent):
    original_shape = img.shape
    x_original = original_shape[0]
    y_original = original_shape[1]
    width_resize = int(y_original * scale_percent / 100)
    height_resize = int(x_original * scale_percent / 100)
    dim = (width_resize, height_resize)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def detect_persons_with_faces(image):
    all_persons=[]
    faceso = []
    persons = detect_persons(imutils.resize(image, width=min(400, image.shape[1])))
    for person in persons:
        crop = crop_image(image, person)
        #print(crop)
        crop = resize_image(crop,100)
        #print(crop)
        faces, eyes = true_eyes_and_faces(crop)
        if faces != []:
            all_persons.append(person)
        for face in faces:
            faceso.append((face[0]+person[0],face[1]+person[1],face[2],face[3]))
    return all_persons, faceso

def detect_persons_with_rescale(image):
    return detect_persons(imutils.resize(image, width=min(400, image.shape[1])))

