# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True, help="path to images directory")
# args = vars(ap.parse_args())
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture('color_2_without_skelet.mp4')
# cap = cv2.VideoCapture(0)
# loop over the image paths

frame_rate = 30
prev = 0


while True:
    timer = cv2.getTickCount()
    _, image = cap.read()
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    # image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    # show some information on the number of bounding boxes
    # filename = imagePath[imagePath.rfind("/") + 1:]
    # print("[INFO] {}: {} original boxes, {} after suppression".format(
    # filename, len(rects), len(pick)))
    # show the output images
    fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))
    cv2.putText(image, str(fps), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
