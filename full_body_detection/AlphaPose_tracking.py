import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn

import numpy as np
import cv2
from torchvision import transforms
from matplotlib import pyplot as plt



#### live feed ####
# def overlapping_area(box1, box2):
#     dx = min(box1[2], box2[2]) - max(box1[0], box2[0])
#     dy = min(box1[3], box2[3]) - max(box1[1], box2[1])
#     return dx*dy
#
# def filter_boxes(boxes, max_overlap = 0.5):
#     if len(boxes) < 2:
#         return boxes
#
#     new_boxes = list()
#
#     for box in boxes:         # area > 10000
#         if (box[2]-box[0])*(box[3]-box[1]) > 20:
#             new_boxes += [box]
#
#     if len(boxes) < 2:
#         return new_boxes



model = keypointrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=2, num_keypoints=17, pretrained_backbone=True)
model.eval()

img_transform = transforms.ToTensor()


cap = cv2.VideoCapture(0)
cap.set(cv2.cv2.CAP_PROP_FPS, 1)

while cap.isOpened():
    _, frame = cap.read()

    image_tensor = img_transform(frame)
    with torch.no_grad():
        output = model([image_tensor])

    image_array = np.array(frame)
    for box in output[0]["boxes"]:      #"boxes" or "keypoints"
        box = box.int()

        if (box[2]-box[0])*(box[3]-box[1]) > 10000:         #overlappende dozen nog fixen
            image_array = cv2.rectangle(image_array, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)

    cv2.imshow('rect', image_array)

    if cv2.waitKey(1) & 0xFF == ord('+'):
        break


cap.release()
cv2.destroyAllWindows()





