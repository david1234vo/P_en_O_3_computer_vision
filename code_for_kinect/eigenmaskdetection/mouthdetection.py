import cv2
import numpy as np

# Load the cascade
mouth_cascade = cv2.CascadeClassifier(
    'C:/Users/Jasper/PycharmProjects/PO3/P_en_O_3_computer_vision/Cascades/haarcascade_mcs_mouth.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the mouths
    mouths = mouth_cascade.detectMultiScale(gray, 1.1, 4)
    print(mouths)
    number_of_mouths = len(mouths)
    cv2.putText(img, str(number_of_mouths), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    # Draw the rectangle around each mouth
    for (x, y, w, h) in mouths:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
