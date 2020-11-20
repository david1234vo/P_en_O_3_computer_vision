import numpy as np
import cv2


# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/



# def non_max_suppression_fast(boxes, overlapThresh):
#     # if there are no boxes, return an empty list
#     if len(boxes) == 0:
#         return []
#     # if the bounding boxes integers, convert them to floats --
#     # this is important since we'll be doing a bunch of divisions
#     if boxes.dtype.kind == "i":
#         boxes = boxes.astype("float")
#     # initialize the list of picked indexes
#     pick = []
#     # grab the coordinates of the bounding boxes
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
#     # compute the area of the bounding boxes and sort the bounding
#     # boxes by the bottom-right y-coordinate of the bounding box
#     area = (x2 - x1 + 1) * (y2 - y1 + 1)
#     idxs = np.argsort(y2)
#     # keep looping while some indexes still remain in the indexes
#     # list
#     while len(idxs) > 0:
#         # grab the last index in the indexes list and add the
#         # index value to the list of picked indexes
#         last = len(idxs) - 1
#         i = idxs[last]
#         pick.append(i)
#         # find the largest (x, y) coordinates for the start of
#         # the bounding box and the smallest (x, y) coordinates
#         # for the end of the bounding box
#         xx1 = np.maximum(x1[i], x1[idxs[:last]])
#         yy1 = np.maximum(y1[i], y1[idxs[:last]])
#         xx2 = np.minimum(x2[i], x2[idxs[:last]])
#         yy2 = np.minimum(y2[i], y2[idxs[:last]])
#         # compute the width and height of the bounding box
#         w = np.maximum(0, xx2 - xx1 + 1)
#         h = np.maximum(0, yy2 - yy1 + 1)
#         # compute the ratio of overlap
#         overlap = (w * h) / area[idxs[:last]]
#         # delete all indexes from the index list that have
#         idxs = np.delete(idxs, np.concatenate(([last],
#                                                np.where(overlap > overlapThresh)[0])))
#     # return only the bounding boxes that were picked using the
#     # integer data type
#     return boxes[pick].astype("int")








# cap = cv2.VideoCapture(0)
# # cap.set(cv2.cv2.CAP_PROP_FPS, 30)
#
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#
# while cap.isOpened():
#     _, frame = cap.read()
#     rectangles, w = hog.detectMultiScale(frame,
#                                          winStride=(10, 10),
#                                          padding=(32, 32),
#                                          scale=1.05,
#                                          useMeanshiftGrouping=False,
#                                          )
#
#     for x, y, w, h in rectangles:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=1)
#
#     cv2.imshow('full body detection using HOGDescriptor', frame)
#
#     if cv2.waitKey(1) & 0XFF == ord('+'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()



cap = cv2.VideoCapture(0)
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')


while cap.isOpened():
    _, frame = cap.read()
    rectangles = upperbody_cascade.detectMultiScale(frame,
                                                    scaleFactor=1.05,
                                                    minNeighbors=7,
                                                    minSize=(30, 30)
                                                    )
    print(rectangles)
    for x, y, w, h in rectangles:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0, 255, 0), thickness= 2)

    cv2.imshow('upper body detection using haarcascades', frame)


    if cv2.waitKey(1) & 0xFF == ord('+'):
        break
cap.release()
cv2.destroyAllWindows()

