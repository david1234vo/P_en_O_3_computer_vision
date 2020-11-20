import cv2
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier(
    'C:/Users/Jasper/PycharmProjects/PO3/P_en_O_3_computer_vision/Cascades/haarcascade_frontalface_default.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 105

while True:
    # Read the frame
    _, img = cap.read()
    img = cv2.flip(img, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)


    number_of_faces = len(faces)
    cv2.putText(img, str(number_of_faces), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 0), 2)
    for (x, y, w, h) in faces_bw:
        cv2.rectangle(black_and_white, (x, y), (x + w, y + h), (0, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
