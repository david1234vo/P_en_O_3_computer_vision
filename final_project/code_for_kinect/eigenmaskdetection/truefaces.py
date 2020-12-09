import cv2
import numpy as np
import functions

# Werkt voor 1 gezicht

# Load the cascade
face_cascade = cv2.CascadeClassifier(
    'C:/Users/Jasper/PycharmProjects/PO3/P_en_O_3_computer_vision/Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'C:/Users/Jasper/PycharmProjects/PO3/P_en_O_3_computer_vision/Cascades/haarcascade_mcs_eyepair_big.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)


# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')


while True:
    # Read the frame
    _, img = cap.read()
    img = cv2.flip(img, 1)
    img1 = img.copy()

    faces = functions.detect_faces(img)
    eyes = functions.detect_eyes(img)
    true_eyes, true_faces = functions.true_eyes_and_faces(img)

    functions.add_counter(img1,faces)
    functions.add_counter(img1,eyes,(100,50),(0,255,0))
    functions.add_counter(img,true_faces)
    functions.add_counter(img,true_eyes,(100,50),(0,255,0))

    functions.add_rectangle(img1,faces)
    functions.add_rectangle(img1,eyes,(0,255,0))
    functions.add_rectangle(img,true_faces)
    functions.add_rectangle(img,true_eyes,(0,255,0))

    # Display
    cv2.imshow('origineel', img1)
    cv2.imshow('echte gezichten', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
