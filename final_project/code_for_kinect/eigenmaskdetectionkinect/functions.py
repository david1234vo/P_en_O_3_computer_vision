import cv2
import numpy as np
import math
import scipy

face_cascade = cv2.CascadeClassifier(
    'C:/Users/david/PycharmProjects/peno/P_en_O_3_computer_vision/Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'C:/Users/david/PycharmProjects/peno/P_en_O_3_computer_vision/Cascades/haarcascade_mcs_eyepair_big.xml')
mouth_cascade = cv2.CascadeClassifier(
    'C:/Users/david/PycharmProjects/peno/P_en_O_3_computer_vision/Cascades/haarcascade_mcs_mouth.xml')
smile_cascade = cv2.CascadeClassifier(
    'C:/Users/david/PycharmProjects/peno/P_en_O_3_computer_vision/Cascades/haarcascade_smile.xml')
nose_cascade = cv2.CascadeClassifier(
    'C:/Users/david/PycharmProjects/peno/P_en_O_3_computer_vision/Cascades/haarcascade_mcs_nose.xml')


def get_corners(rect):
    (x, y, w, h) = rect
    corners = (x, y, x + w, y + h)
    return corners


def within_face(rect, face):
    (xmin, ymin, xmax, ymax) = get_corners(rect)
    (xMin, yMin, xMax, yMax) = get_corners(face)
    if xmin >= xMin:
        if xmax <= xMax:
            if ymin >= yMin:
                if ymax <= yMax:
                    return True
    return False


def mouth_within_face(mouth, face):
    (xmin, ymin, xmax, ymax) = get_corners(mouth)
    (xMin, yMin, xMax, yMax) = get_corners(face)
    if xmin >= xMin:
        if xmax <= xMax:
            if ymin >= yMin:
                y_scale = 0.3
                h = ymax - ymin
                if ymax < yMax + y_scale * h:
                    return True
    return False


def all_rectangles_within_face(array, faces):
    new_arraylst = []
    for face in faces:
        for i in range(0, len(array)):
            rect = array[i]
            if within_face(rect, face):
                new_arraylst.append(rect)
    new_array = np.array(new_arraylst)
    return new_array


def all_mouths_within_face(mouths, faces):
    new_mouthslst = []
    for face in faces:
        for i in range(0, len(mouths)):
            mouth = mouths[i]
            if mouth_within_face(mouth, face):
                new_mouthslst.append(mouth)
    new_mouths = np.array(new_mouthslst)
    return new_mouths


def all_faces_with_a_pair(eyes, faces):
    new_faceslst = []
    for face in faces:
        a = 0
        for pair in eyes:
            if within_face(pair, face):
                a += 1
        if a != 0:
            new_faceslst.append(face)
    new_faces = np.array(new_faceslst)
    return new_faces


def true_eyes_and_faces(img):
    faces = detect_faces(img)
    eyes = detect_eyes(img)
    true_faces = all_faces_with_a_pair(eyes, faces)
    true_eyes = all_rectangles_within_face(eyes, faces)
    return true_faces, true_eyes


def detect_faces(img):  # als gray zwz berekend wordt kan invoer ook gray zijn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces


def detect_eyes(img):  # als gray zwz berekend wordt kan invoer ook gray zijn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    return eyes


def detect_mouths(img):  # als gray zwz berekend wordt kan invoer ook gray zijn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mouths = mouth_cascade.detectMultiScale(gray, 1.1, 4)
    return mouths


def detect_smiles(img):  # als gray zwz berekend wordt kan invoer ook gray zijn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
    return smiles


def detect_noses(img):  # als gray zwz berekend wordt kan invoer ook gray zijn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noses = nose_cascade.detectMultiScale(gray, 1.1, 4)
    return noses


def add_counter(img, array, pos=(50, 50), color=(255, 0, 0)):
    number = len(array)
    cv2.putText(img, str(number), pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color)


def add_rectangle(img, array, color=(255, 0, 0)):
    for (x, y, w, h) in array:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def crop_image(original, face):
    original_shape = original.shape
    (x_original, y_original, _) = original_shape
    edge = int((y_original - x_original) / 2)
    res = original_shape[0] / original_shape[1]

    if range == [0, 0, x_original, y_original]:
        crop = original[0:x_original, edge:y_original - edge]
    else:
        (y_min, x_min, y_max, x_max) = get_corners(face)

        crop = original[x_min:x_max, y_min:y_max]

    scale_percent = 100  # percent of original size
    width_resize = int(x_original * scale_percent / 100)
    height_resize = int(x_original * scale_percent / 100)
    dim = (width_resize, height_resize)
    resized = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)

    return resized


def overlap(rect1, rect2):
    dx = min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])
    dy = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return 0


def area(rect, type_rect):
    if type_rect == 0:
        dx = rect[2] - rect[0]
        dy = rect[3] - rect[1]
    if type_rect == 1:
        dx = rect[2]
        dy = rect[3]
    return dx * dy


def relative_overlap(rect1, rect2):
    overlap_area = overlap(rect1, rect2)
    max_area = max(area(rect1), area(rect2))
    rel_overlap = overlap_area / max_area
    return rel_overlap


def good_overlap(array1, array2):
    rect1 = get_corners(array1)
    rect2 = get_corners(array2)
    rel_overlap = relative_overlap(rect1, rect2)
    if 0.5 < rel_overlap < 0.8:
        pass
    if rel_overlap > 0.6:
        return True


def true_mouths_and_smiles(mouths, smiles, true_faces):
    true_mouth = []
    true_smile = []
    good_mouths = all_mouths_within_face(mouths, true_faces)
    good_smiles = all_rectangles_within_face(smiles, true_faces)
    for mouth in good_mouths:
        for smile in good_smiles:
            if good_overlap(mouth, smile):
                true_mouth.append(mouth)
                true_smile.append(smile)
    return true_mouth, true_smile


def middle_position(rect):
    (x, y, w, h) = rect
    return (int(x + w / 2), int(y + h / 2))


def best_mouth_to_face(mouths, faces):
    faces_mouths_noses = []
    best_mouths = []
    good_mouths = all_mouths_within_face(mouths, faces)
    for face in faces:
        best_mouths_in_face = []
        for mouth in good_mouths:
            (xf, yf, wf, hf) = face
            (xm, ym, wm, hm) = mouth
            if middle_position(face)[1] < ym < yf + hf:
                rel_dist = relative_distance_to_midpoint(mouth, face, 0)
                if rel_dist < 0.1:
                    best_mouths_in_face.append(mouth)

        if len(best_mouths_in_face) > 1:
            areas = []
            for best_mouth in best_mouths_in_face:
                ar = area(best_mouth, 1)
                areas.append(ar)
            max_ar = max(areas)
            pos = areas.index(max_ar)
            ultimate_mouth = best_mouths_in_face[pos]
        elif len(best_mouths_in_face) == 1:
            ultimate_mouth = best_mouths_in_face[0]
        else:
            ultimate_mouth = None
        if ultimate_mouth is not None:
            best_mouths.append(ultimate_mouth)
        mouth_in_face = [face, ultimate_mouth, None]

        faces_mouths_noses.append(mouth_in_face)

    return best_mouths, faces_mouths_noses


def relative_distance_to_midpoint(rect, face, axis):
    if axis == 0:
        x_rect = middle_position(rect)[0]
        x_face = middle_position(face)[0]
        dx = abs(x_face - x_rect)
        w = face[2]
        rel_dist = dx / w
        return rel_dist
    if axis == 1:
        y_rect = middle_position(rect)[1]
        y_face = middle_position(face)[1]
        dy = abs(y_face - y_rect)
        h = face[3]
        rel_dist = dy / h
        return rel_dist


def best_nose_to_face(noses, faces, faces_mouths_noses):
    best_noses = []
    good_noses = all_rectangles_within_face(noses, faces)
    for face in faces:
        best_noses_in_face = []
        for nose in good_noses:
            rel_distx = relative_distance_to_midpoint(nose, face, 0)
            if rel_distx < 0.2:
                rel_disty = relative_distance_to_midpoint(nose, face, 1)
                if rel_disty < 0.2:
                    best_noses_in_face.append(nose)

        if len(best_noses_in_face) > 1:
            ultimate_nose = best_noses_in_face[0]
        elif len(best_noses_in_face) == 1:
            ultimate_nose = best_noses_in_face[0]
        else:
            ultimate_nose = None

        for face_mouth_nose in faces_mouths_noses:
            if face_mouth_nose[0].all() == face.all():
                face_mouth_nose[2] = ultimate_nose

        if ultimate_nose is not None:
            best_noses.append(ultimate_nose)

    return best_noses


def best_mouth_to_nose(best_mouths, noses, faces):
    for face1 in faces:
        for nose in noses:
            for mouth1 in best_mouths:
                mouth2 = mouth1
                nose_in_face = within_face(nose, face1)
                mouth_in_face = within_face(mouth1, face1)
                if nose_in_face and mouth_in_face:
                    area_mouth = area(mouth1)
                    area_nose = area(nose)
                    if area_mouth >= area_nose:
                        pass  # del slechte mouth


def lengthen_faces(faces):
    new_faces = []
    factor = 1.2
    for (x, y, w, h) in faces:
        new_faces.append((x, y, w, int(h * factor)))
    return new_faces


def color_difference(color1, color2):  # BGR
    (b1, g1, r1) = color1
    (b2, g2, r2) = color2
    diff = abs(math.sqrt(2 * (b2 - b1) ** 2 + (g2 - g1) ** 2 + (r2 - r1) ** 2))
    return diff


def npcolor_to_color(color):
    (b, g, r) = color
    return (int(b), int(g), int(r))


def mask_due_color(img, faces=None,resizefactor=100):
    mask_info = []
    if not isinstance(faces, np.ndarray):
        faces = detect_faces(img)
    coordinates = []
    for face in faces:
        (x, y, w, h) = 100*face//resizefactor
        side_pos = get_corners(face)
        middle_pos = middle_position(face)
        diff_mouth, diff_nose, coordinate = mouth_to_forehead_difference(img, (side_pos + middle_pos))
        if diff_mouth < 75:
            # display_wear_a_mask(img, pos, 0)
            mask_info.append((x, y, w, h, 2))
        else:
            if diff_nose < 75:
                # display_wear_properly(img, pos, 0)
                mask_info.append((x,y,w,h,1))
            else:
                mask_info.append((x,y,w,h,0))
    # print(mask_info)
    return mask_info

def quantize(img, NUM_CLUSTERS = 5):
    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    return vecs

def mask_due_quantize(img, faces=None,resizefactor=100):
    mask_info = []
    if not isinstance(faces, np.ndarray):
        faces = detect_faces(img)
    coordinates = []
    for face in faces:
        (x, y, w, h) = 100*face//resizefactor
        side_pos = get_corners(face)
        middle_pos = middle_position(face)
        diff_mouth, diff_nose, coordinate = mouth_to_forehead_difference(img, (side_pos + middle_pos))
        if diff_nose < 75:
            if diff_mouth < 75:
                # display_wear_a_mask(img, pos, 0)
                mask_info.append((x,y,w,h,2, coordinate))
            else:
                # display_wear_properly(img, pos, 0)
                mask_info.append((x,y,w,h,1, coordinate))
        else:
            mask_info.append((x,y,w,h,0, coordinate))
    # print(mask_info)
    return mask_info


def mouth_to_forehead_difference(img, pos):
    (xmin, ymin, xmax, ymax, xmid, ymid) = pos
    face_factor_y = 0.4
    y_face = int(ymin + face_factor_y * (ymid - ymin))
    face_factor_x = 0.9
    x_face = int(xmin + face_factor_x * (xmid - xmin))
    forehead = img[y_face][x_face]
    forehead = npcolor_to_color(forehead)
    mouth_factor_y = 0.45
    y_mouth = int(ymid + mouth_factor_y * (ymax - ymid))
    mouth_factor_x = 0.6
    x_mouth = int(xmin + mouth_factor_x * (xmid - xmin))
    mouth = img[y_mouth][x_mouth]
    mouth = npcolor_to_color(mouth)
    nose_factor_y = 0.1
    y_nose = int(ymid + nose_factor_y * (ymax - ymid))
    nose_factor_x = 0.6
    x_nose = int(xmin + nose_factor_x * (xmid - xmin))
    nose = img[y_nose][x_nose]
    nose = npcolor_to_color(nose)
    diff_mouth = color_difference(forehead, mouth)
    diff_nose = color_difference(forehead, nose)
    cv2.circle(img, (x_face, y_face), 5, forehead, -1)
    cv2.circle(img, (x_mouth, y_mouth), 5, mouth, -1)
    cv2.circle(img, (x_nose, y_nose), 5, nose, -1)
    # print("forehead:", forehead, "/mouth:", mouth, "/nose:", nose, "/diff_mouth:", diff_mouth, "/diff_nose:", diff_nose)
    return diff_mouth, diff_nose, [(x_face, y_face, forehead), (x_mouth, y_mouth, mouth), (x_nose, y_nose, nose)]


def display_wear_a_mask(img, pos, type_rect):
    if type_rect == 0:
        length_face = pos[2] - pos[0]
    if type_rect == 1:
        length_face = pos[2]
    length_text = cv2.getTextSize('Please wear a mask!', cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]
    scale = length_face / length_text
    cv2.putText(img, 'Please wear a mask!', (pos[0], pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255))


def display_wear_properly(img, pos, type_rect):
    if type_rect == 0:
        length_face = pos[2] - pos[0]
    if type_rect == 1:
        length_face = pos[2]
    length_text1 = cv2.getTextSize('Please wear your mask', cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]
    length_text2 = cv2.getTextSize('over your nose!', cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]
    scale = length_face / length_text1
    cv2.putText(img, 'Please wear your mask', (pos[0], pos[1] - 22), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255))
    cv2.putText(img, 'over your nose!', (pos[0], pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255))


def resize_image(img, scale_percent):
    original_shape = img.shape
    x_original = original_shape[0]
    y_original = original_shape[1]
    width_resize = int(y_original * scale_percent / 100)
    height_resize = int(x_original * scale_percent / 100)
    dim = (width_resize, height_resize)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    #resized = cv2.pyrDown(img)   # te gebruiken voor 50% comprimeren werkt anders niet
    return resized


def mask_due_haar(img, faces_mouths_noses):
    for face_mouth_nose in faces_mouths_noses:
        if face_mouth_nose[1] is not None:
            display_wear_a_mask(img, face_mouth_nose[0], 1)
        else:
            if face_mouth_nose[2] is not None:
                display_wear_properly(img, face_mouth_nose[0], 1)
