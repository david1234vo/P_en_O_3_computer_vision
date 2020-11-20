import functions
import cv2
import numpy as np
import time

print(cv2.getTextSize('Please wear a mask!', cv2.FONT_HERSHEY_SIMPLEX, 1, 1))

cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')


while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    img1 = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = functions.detect_faces(img)
    position = []
    for face in faces:
        side_pos = functions.get_corners(face)
        middle_pos = functions.middle_position(face)
        position.append((side_pos + middle_pos))

    for (xmin, ymin, xmax, ymax, xmid, ymid) in position:
        gezicht_factor_y = 0.4
        y_gezicht = int(ymin + gezicht_factor_y * (ymid - ymin))
        gezicht_factor_x = 0.9
        x_gezicht = int(xmin + gezicht_factor_x * (xmid - xmin))
        voorhoofd = img[y_gezicht][x_gezicht]
        voorhoofd = functions.npcolor_to_color(voorhoofd)
        mond_factor_y = 0.3
        y_mond = int(ymid + mond_factor_y * (ymax - ymid))
        mond_factor_x = 0.6
        x_mond = int(xmin + mond_factor_x * (xmid - xmin))
        mond = img[y_mond][x_mond]
        mond = functions.npcolor_to_color(mond)
        print(mond)
        verschil = functions.color_difference(voorhoofd, mond)
        print('verschil: ', verschil)
        print('')
        cv2.circle(img, (x_gezicht, y_gezicht), 5, voorhoofd, -1)
        cv2.circle(img, (x_mond, y_mond), 5, mond, -1)
        breedte = xmax - xmin
        scale = breedte / 334
        print(scale)
        cv2.putText(img, 'Please wear a mask!', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255))
    print('')
    functions.add_rectangle(img, faces)
    cv2.imshow('img', img)

    # cv2.imshow('img1', img1)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
