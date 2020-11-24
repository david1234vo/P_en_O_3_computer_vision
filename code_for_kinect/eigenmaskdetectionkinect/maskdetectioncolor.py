from code_for_kinect.eigenmaskdetectionkinect import functions


def maskdetectioncolor(img, resizefactor=100):
    img = functions.resize_image(img, resizefactor)
    faces, eyes = functions.true_eyes_and_faces(img)
    mask_info = functions.mask_due_color(img, faces, resizefactor)
    return mask_info

