import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    fgmask = fgbg.apply(frame)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    cv2.imshow('frame',res)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()