import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    edges = cv.Canny(frame, 100, 200)
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_RGB2RGBA)
    # Display the resulting frame
    cv.imshow('frame',edges)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()