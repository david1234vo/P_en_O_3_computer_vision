import cv2
import numpy as np
import math

face_cascade = cv2.CascadeClassifier(
    'C:/Users/Jasper/PycharmProjects/PO3/P_en_O_3_computer_vision/Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'C:/Users/Jasper/PycharmProjects/PO3/P_en_O_3_computer_vision/Cascades/haarcascade_mcs_eyepair_big.xml')
mouth_cascade = cv2.CascadeClassifier(
    'C:/Users/Jasper/PycharmProjects/PO3/P_en_O_3_computer_vision/Cascades/haarcascade_mcs_mouth.xml')
smile_cascade = cv2.CascadeClassifier(
    'C:/Users/Jasper/PycharmProjects/PO3/P_en_O_3_computer_vision/Cascades/haarcascade_smile.xml')
nose_cascade = cv2.CascadeClassifier(
    'C:/Users/Jasper/PycharmProjects/PO3/P_en_O_3_computer_vision/Cascades/haarcascade_mcs_nose.xml')

# Rectangles come in 2 variations:
# Type 1: (x, y, width, height) -> upper-left coordinate, a width and a height
# Type 2: (xmin, ymin, xmax, ymax) -> upper-left coordinate and lower-right coordinate
type_1 = 0
type_2 = 1


def get_type_2(rect):
    """Calculates type 2 from type 1, input: type 1"""
    (x, y, w, h) = rect
    type2 = (x, y, x + w, y + h)
    return type2


def get_type_1(rect):
    """Calculates type 1 from type 2, input: type 2"""
    (xmin, ymin, xmax, ymax) = rect
    type1 = (xmin, ymin, xmax - xmin, ymax - ymin)
    return type1


def within_face(rect, face):
    """Checks if a rectangle is completely within a face, input: type 1"""
    (xmin, ymin, xmax, ymax) = get_type_2(rect)
    (xMin, yMin, xMax, yMax) = get_type_2(face)
    if xmin >= xMin:
        if xmax <= xMax:
            if ymin >= yMin:
                if ymax <= yMax:
                    return True
    return False


def mouth_within_face(mouth, face):
    """Checks if a mouth in 'within' a face -> the lowest point of a mouth can be a certain amount lower
     than the lowest point of the face, input: type 1"""
    (xmin, ymin, xmax, ymax) = get_type_2(mouth)
    (xMin, yMin, xMax, yMax) = get_type_2(face)
    if xmin >= xMin:
        if xmax <= xMax:
            if ymin >= yMin:
                y_scale = 0.3
                h = ymax - ymin
                if ymax < yMax + y_scale * h:
                    return True
    return False


def all_rectangles_within_face(array, faces):
    """Returns all rectangles that are completely within one of the faces, input: type 1"""
    new_arraylst = []
    for face in faces:
        for i in range(0, len(array)):
            rect = array[i]
            if within_face(rect, face):
                new_arraylst.append(rect)
    new_array = np.array(new_arraylst)
    return new_array


def all_mouths_within_face(mouths, faces):
    """Returns all mouths that are 'within' one of the faces, input: type 1"""
    new_mouthslst = []
    for face in faces:
        for i in range(0, len(mouths)):
            mouth = mouths[i]
            if mouth_within_face(mouth, face):
                new_mouthslst.append(mouth)
    new_mouths = np.array(new_mouthslst)
    return new_mouths


def all_faces_with_a_pair(eyes, faces):
    """Returns all faces with at least one pair of eyes within it, input: type 1"""
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
    """Returns only the faces with at least one pair of eyes and those eyes"""
    faces = detect_faces(img)
    eyes = detect_eyes(img)
    true_faces = all_faces_with_a_pair(eyes, faces)
    true_eyes = all_rectangles_within_face(eyes, faces)
    return true_faces, true_eyes


def detect_faces(img):  # als gray zwz berekend wordt kan invoer ook gray zijn
    """Returns an array with all recognised faces using Haarcascades"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces


def detect_eyes(img):  # als gray zwz berekend wordt kan invoer ook gray zijn
    """Returns an array with all recognised eyes using Haarcascades"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    return eyes


def detect_mouths(img):  # als gray zwz berekend wordt kan invoer ook gray zijn
    """Returns an array with all recognised mouths using Haarcascades"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mouths = mouth_cascade.detectMultiScale(gray, 1.1, 4)
    return mouths


def detect_smiles(img):  # als gray zwz berekend wordt kan invoer ook gray zijn
    """Returns an array with all recognised smiles using Haarcascades"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
    return smiles


def detect_noses(img):  # als gray zwz berekend wordt kan invoer ook gray zijn
    """Returns an array with all recognised noses using Haarcascades"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noses = nose_cascade.detectMultiScale(gray, 1.1, 4)
    return noses


def add_counter(img, array, pos=(50, 50), color=(255, 0, 0)):
    """Prints the length of an array on an opencv-image"""
    number = len(array)
    cv2.putText(img, str(number), pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color)


def add_rectangle(img, array, color=(255, 0, 0)):
    """Draws a rectangle on an opencv-image, input: type 1"""
    for (x, y, w, h) in array:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def crop_image(original, rect):
    """Crops an image to the size of a given rectangle"""
    original_shape = original.shape
    (x_original, y_original, _) = original_shape
    edge = int((y_original - x_original) / 2)
    res = original_shape[0] / original_shape[1]

    if range == [0, 0, x_original, y_original]:
        crop = original[0:x_original, edge:y_original - edge]
    else:
        (y_min, x_min, y_max, x_max) = get_type_2(rect)

        crop = original[x_min:x_max, y_min:y_max]

    scale_percent = 100  # percent of original size
    width_resize = int(x_original * scale_percent / 100)
    height_resize = int(x_original * scale_percent / 100)
    dim = (width_resize, height_resize)
    resized = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)

    return resized


def overlap(rect1, rect2):
    """Calculates the overlapping area of two rectangles, input: type 2"""
    dx = min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])
    dy = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return 0


def area(rect, type_rect):
    """Calculates the area of a rectangle, input: type 1 and type 2 -> specify"""
    if type_rect == 0:
        dx = rect[2]
        dy = rect[3]
    elif type_rect == 1:
        dx = rect[2] - rect[0]
        dy = rect[3] - rect[1]
    else:
        return False
    return dx * dy


def relative_overlap(rect1, rect2):
    """Calculates the relative overlap between two rectangles compared to the biggest rectangle of the two, input: type 2"""
    overlap_area = overlap(rect1, rect2)
    max_area = max(area(rect1, type_2), area(rect2, type_2))
    rel_overlap = overlap_area / max_area
    return rel_overlap


def good_overlap(array1, array2):
    """Returns True if two rectangles have a relative overlap of more than 60%, input: type 1"""
    rect1 = get_type_2(array1)
    rect2 = get_type_2(array2)
    rel_overlap = relative_overlap(rect1, rect2)
    if rel_overlap > 0.6:
        return True


def true_mouths_and_smiles(mouths, smiles, true_faces):
    """Returns all mouths and smiles that have a good overlap with each other, input: type 1"""
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


def middle_position(rect, type_rect):
    """Returns the middle position of a rectangle, input: type 1 and type 2 -> specify"""
    if type_rect == 0:
        (x, y, w, h) = rect
    elif type_rect == 1:
        (x, y, w, h) = get_type_1(rect)
    else:
        return False
    return (int(x + w / 2), int(y + h / 2))


def best_mouth_to_face(mouths, faces):
    """Picks the best mouth in a face based on position and area, input: type 1"""
    faces_mouths_noses = []
    best_mouths = []
    good_mouths = all_mouths_within_face(mouths, faces)
    for face in faces:
        best_mouths_in_face = []
        for mouth in good_mouths:
            (xf, yf, wf, hf) = face
            (xm, ym, wm, hm) = mouth
            if middle_position(face,type_1)[1] < ym < yf + hf:
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
    """Calculates the relative distance between the middle points of a face and a rectangle compared to the size of the face,
    input: specify axis -> x or y, type 1"""
    if axis == 0:
        x_rect = middle_position(rect,type_1)[0]
        x_face = middle_position(face,type_1)[0]
        dx = abs(x_face - x_rect)
        w = face[2]
        rel_dist = dx / w
        return rel_dist
    if axis == 1:
        y_rect = middle_position(rect,type_1)[1]
        y_face = middle_position(face,type_1)[1]
        dy = abs(y_face - y_rect)
        h = face[3]
        rel_dist = dy / h
        return rel_dist


def best_nose_to_face(noses, faces, faces_mouths_noses):
    """Picks the best nose in a face based on position and area, input: type 1"""
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
    """Picks the best mouth to a nose based on position, input: type 1"""
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
    """Lengthen the y axis of a face by a factor, input: type 1"""
    new_faces = []
    factor = 1.2
    for (x, y, w, h) in faces:
        new_faces.append((x, y, w, int(h * factor)))
    return new_faces


def color_difference(color1, color2):
    """Calculates the difference between 2 BGR-colors"""
    (b1, g1, r1) = color1
    (b2, g2, r2) = color2
    diff = abs(math.sqrt(2 * (b2 - b1) ** 2 + (g2 - g1) ** 2 + (r2 - r1) ** 2))
    return diff


def npcolor_to_color(color):
    """Converts a numpy-color to a BGR-color"""
    (b, g, r) = color
    return (int(b), int(g), int(r))


def mask_due_color(img, faces=None):
    """Returns if it recognises a good, a bad or no mask on all given faces based on color difference,
     input: type 1"""
    positions = []
    if not isinstance(faces, np.ndarray):
        faces = detect_faces(img)
    for face in faces:
        side_pos = get_type_2(face)
        middle_pos = middle_position(face,type_1)
        positions.append((side_pos + middle_pos))

    for pos in positions:
        diff_mouth, diff_nose = mouth_to_forehead_difference(img, pos)
        if diff_mouth < 75:
            display_wear_a_mask(img, pos, 0)
        else:
            if diff_nose < 75:
                display_wear_properly(img, pos, 0)

    pass


def mouth_to_forehead_difference(img, pos):
    """Calculates the color difference between a point on the forehead, a point next to the nose and a point next to the mouth,
     input: extension of type 1"""
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
    return diff_mouth, diff_nose


def display_wear_a_mask(img, pos, type_rect):
    """Writes 'Please wear a mask' on an opencv-image"""
    if type_rect == 0:
        length_face = pos[2]
    if type_rect == 1:
        length_face = pos[2] - pos[0]
    length_text = cv2.getTextSize('Please wear a mask!', cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]
    scale = length_face / length_text
    cv2.putText(img, 'Please wear a mask!', (pos[0], pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255))


def display_wear_properly(img, pos, type_rect):
    """Writes 'Please wear your mask over your nose' on an opencv-image"""
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
    """Resizes an opencv image with a given scale"""
    original_shape = img.shape
    x_original = original_shape[0]
    y_original = original_shape[1]
    width_resize = int(y_original * scale_percent / 100)
    height_resize = int(x_original * scale_percent / 100)
    dim = (width_resize, height_resize)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def mask_due_haar(img, faces_mouths_noses):
    """Returns if it recognises a good, a bad or no mask on all given faces based on mouth and nose recognition
     with Haarcascades, input: type 1"""
    for face_mouth_nose in faces_mouths_noses:
        if face_mouth_nose[1] is not None:
            display_wear_a_mask(img, face_mouth_nose[0], 1)
        else:
            if face_mouth_nose[2] is not None:
                display_wear_properly(img, face_mouth_nose[0], 1)
