import cv2
import numpy

def crop_image(img):
    img_shape=img.shape
    mid_y=int(img_shape[0]/2)
    mid_x=int(img_shape[1]/2)
    h=int(104)
    w=int(4/3*h)
    crop = img[mid_y-h:mid_y+h,mid_x-w:mid_x+w]

    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(crop, dim, interpolation = cv2.INTER_AREA)
    return resized


