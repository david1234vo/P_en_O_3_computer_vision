
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import ctypes
import math
import time
import os
import copy
import random
import sys
import cv2
import pygame
from pygame.locals import *
from code_for_kinect.followfacekinect import *
from code_for_kinect.eigenmaskdetectionkinect.functions import *
from code_for_kinect.eigenmaskdetectionkinect.maskdetectioncolor import *
from code_for_kinect.eigenpersondetection.persondetectionslice import *
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import dlib
import imutils
from imutils import face_utils
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model





class TopDownViewRuntime(object):

    def init(self):
        #schaal van topdown en color camera
        self.topdown_scale = 1/10
        self.color_scale = 4/8

        #grootte van topdown surface (width, height)
        self.topdown_surface_size = (1000, 600)

        self.display_pygame = False


        self._done = False
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_Depth)
        self._bodies = None
        self.frame = 0
        self.topdown_position = (20,20)
        self.new_depth_frame = False
        self.new_body_frame = False

        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.gray = (200, 200, 200)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.orange = (255, 165, 0)
        self.blue = (0, 0, 255)

    def __init__(self):
        self.init()

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def draw_depth_frame(self, frame, target_surface):
        if frame is None:
            return
        target_surface.lock()
        f8=np.uint8(frame.clip(1,4000)/16.)
        frame8bit=np.dstack((f8,f8,f8))
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address

    def draw_infrared_frame(self, frame, target_surface):
        if frame is None:
            return
        target_surface.lock()
        f8=np.uint8(frame.clip(1,4000)/16.)
        frame8bit=np.dstack((f8,f8,f8))
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address
        target_surface.unlock()

    def get_window_size(self):
        return (float(self._kinect.color_frame_desc.Width), float(self._kinect.color_frame_desc.Height))

    def convert_to_coordinates(self, locations, window_size = None):
        if len(locations) == 0:
            return
        lis = True
        to_return = []
        if type(locations[0]) != list and type(locations[0]) != tuple:
            locations = [locations]
            lis = False
        for location in locations:
            if window_size is None:
                window_size = self.get_window_size()
            horizontal_factor = 1/1000 * 0.9
            vertical_factor = -1/1000 * 0.9
            x, y, depth, _ = location
            width, height = window_size
            horizontal_coordinate = (x-width/2)*depth*horizontal_factor
            vertical_coordinate = (y-height/2)*depth*vertical_factor
            to_return.append([horizontal_coordinate, vertical_coordinate, depth])
        if not lis:
            return to_return[0]
        else:
            return to_return

    def color_difference(self, color1, color2):  # BGR
        (b1, g1, r1) = color1
        (b2, g2, r2) = color2
        diff = abs(math.sqrt(2 * (b2 - b1) ** 2 + (g2 - g1) ** 2 + (r2 - r1) ** 2))
        return diff

    def convert_to_coordinate(self, location, window_size = None):
        if window_size is None:
            window_size = self.get_window_size()
        horizontal_factor = 1/1000 * 0.9
        vertical_factor = -1/1000 * 0.9
        x, y, depth = location
        width, height = window_size
        horizontal_coordinate = (x-width/2)*depth*horizontal_factor
        vertical_coordinate = (y-height/2)*depth*vertical_factor
        return [horizontal_coordinate, vertical_coordinate, depth]

    def get_distance(self, location1, location2):
        x1, y1, z1 = location1[0:3]
        x2, y2, z2 = location2[0:3]
        argument = (x1-x2)**2+(y1-y2)**2+(int(z1)-int(z2))**2
        if argument < 0:
            return 0
            print("argument zero")

        return(math.sqrt(argument))

    def get_distances(self, location, list_location, return_zero = False):
        to_return = []
        for second_location in list_location:
            d = self.get_distance(location, second_location)
            if d != 0 or return_zero:
                to_return.append(d)
        return to_return

    def coordinate_to_pixel(self, location, extra = 0):
        return (int(location[0]*self.topdown_scale + self.topdown_surface_size[0]/2 + extra), int(location[2]*self.topdown_scale + extra))

    def get_middle(self, location1, location2):
        x1, y1, z1 = location1
        x2, y2, z2 = location2
        return ((x1+x2)/2,(y1+y2)/2,(z1+z2)/2)

    def get_head_location(self):
        if self._bodies is not None: 
            self.head_locations = []
            for i in range(0, self._kinect.max_body_count):
                body = self._bodies.bodies[i]
                if not body.is_tracked: 
                    continue 
                joints = body.joints 
                joint_points = self._kinect.body_joints_to_color_space(joints)
                joint_points_depth = self._kinect.body_joints_to_depth_space(joints)

                if self.new_body_frame and self.new_depth_frame:
                    try:
                        head_joint = joint_points[PyKinectV2.JointType_Head]
                        head_joint_depth = joint_points_depth[PyKinectV2.JointType_Head]
                        depth_value = self.depth_frame[int(head_joint_depth.y), int(head_joint_depth.x)]
                        if depth_value != 0:
                            self.head_locations.append([head_joint.x, head_joint.y, depth_value])
                    except Exception as e:
                        if "infinity" not in str(e):
                            print("error before return:", e)

    def between_zero_and(self,number, max_number):
        a = int(min(max(number, 0), max_number))
        # print("between", a, number, max_number)
        return a

    def get_hands_location(self):
        """
            returns a list containing tuples which all contain two tuples, respectively storing
            the xyz coordinates of the left hand and the right hand
        """

        hands_locations = []
        if self._bodies is not None:
            for i in range(0, self._kinect.max_body_count):
                body = self._bodies.bodies[i]
                if body.is_tracked:
                    joints = body.joints
                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    joint_points_depth = self._kinect.body_joints_to_depth_space(joints)

                    hand_joint = joint_points[PyKinectV2.JointType_HandLeft]
                    hand_joint_depth = joint_points_depth[PyKinectV2.JointType_HandLeft]
                    if self.between_zero_and(hand_joint_depth.y, 423) == int(hand_joint_depth.y) and self.between_zero_and(hand_joint_depth.x, 511) == int(hand_joint_depth.x):
                        lx = hand_joint.x
                        ly = hand_joint.y
                        lz = int(self.depth_frame[int(hand_joint_depth.y), int(hand_joint_depth.x)])
                    else:
                        lx, ly, lz = 0,0,0
                        # print("left", hand_joint_depth.x, hand_joint_depth.y)

                    hand_joint = joint_points[PyKinectV2.JointType_HandRight]
                    hand_joint_depth = joint_points_depth[PyKinectV2.JointType_HandRight]
                    if self.between_zero_and(hand_joint_depth.y, 423) == int(hand_joint_depth.y) and self.between_zero_and(
                            hand_joint_depth.x, 511) == int(hand_joint_depth.x):
                        rx = hand_joint.x
                        ry = hand_joint.y
                        rz = int(self.depth_frame[int(hand_joint_depth.y), int(hand_joint_depth.x)])
                    else:
                        rx, ry, rz = 0, 0, 0
                        # print("right", hand_joint_depth.x, hand_joint_depth.y)

                    hands_locations.append(((i, lx, ly, lz), (i, rx, ry, rz)))

        return hands_locations

    def hands_too_close(self, distance_allowed):
        """
            prints whether or not the hands of any two people are to close
        """
        hand_locations_without_depth = self.get_hands_location()

        hand_locations = [[[element[0][0]]+self.convert_to_coordinate(element[0][1:]), [element[1][0]]+self.convert_to_coordinate(element[1][1:])] for element
                          in hand_locations_without_depth]

        hand_locations = [[[int(s) for s in element[0]], [int(s) for s in element[1]]] for element
                          in hand_locations]

        # if hand_locations != []:
        #     hand_locations.append([[100, -1143, 33, 1724], [100, -350, 22, 1753]])
        # else:
        #     hand_locations = [[[100, -1143, 33, 1724], [100, -350, 22, 1753]]]

        # print(hand_locations, hand_locations_without_depth)

        # cv2.imshow("hands", self.color_frame)

        # print(hand_locations)
        #
        # for left, right in hand_locations:
        #     d = int(self.color_difference(left, right))
        #     print("distance between", left, "and", right, "is", d)
            # for left2, right2 in hand_locations.copy():
            #     # print(left, right, left2, right2)
            #     if (left != left2 or not (left == left2 and [int(s) for s in left] == [0, 0, 0])) and (
            #             right != right2 or not (right == right2 and [int(s) for s in right] == [0, 0, 0])):
            #         distances = [int(self.color_difference(left, left2)), int(self.color_difference(right, right2)),
            #                      int(self.color_difference(left, right2)), int(self.color_difference(right, left2))]
            #         if left == (0, 0, 0):
            #             distances[0] = 0
            #             distances[2] = 0
            #         if right == (0, 0, 0):
            #             distances[1] = 0
            #             distances[3] = 0
            #         if left2 == (0, 0, 0):
            #             distances[0] = 0
            #             distances[3] = 0
            #         if right2 == (0, 0, 0):
            #             distances[1] = 0
            #             distances[2] = 0
            #         # print("distance", left, right, left2, right2, distances)
            #         m = min(distances)
            #         # if 0 < m < 1500:
            #         print(m)

        # print("hand_locations:", hand_locations)
        if len(hand_locations) > 1:
            for i in range(len(hand_locations)-1):
                for k in range(0, 2):
                    id1, x1, y1, z1 = hand_locations[i][k]

                    for j in range(len(hand_locations[i:])):
                        for q in range(0, 2):
                            id2, x2, y2, z2 = hand_locations[j][q]


                            if id1 != id2:
                                current_distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
                                # if current_distance > 0: print("distance between", x1, y1, z1,"/", x2, y2, z2, "is", current_distance)
                                if 0 < current_distance < distance_allowed:
                                    print(current_distance, "too close: hands at", [int(x) for x in hand_locations[i][k]], "and", [int(x) for x in hand_locations[j][q]])
                                    font = pygame.font.SysFont('Comic Sans MS', 80)
                                    textsurface = font.render("hands at distance "+str(round(current_distance, 2)), False, (0, 0, 255))
                                    text_coordinate = (50,50)
                                    self.color_surface.blit(textsurface, text_coordinate)

        # for left, right in hand_locations:
        #     for left2, right2 in hand_locations.copy():
        #         if (left != left2) and (right != right2):
        #             distances = [int(self.color_difference(left, left2)), int(self.color_difference(right, right2)), int(self.color_difference(left, right2)), int(self.color_difference(right, left2))]
        #             if left == (0,0,0):
        #                 distances[0] = 0
        #                 distances[2] = 0
        #             if right == (0,0,0):
        #                 distances[1] = 0
        #                 distances[3] = 0
        #             if left2 == (0,0,0):
        #                 distances[0] = 0
        #                 distances[3] = 0
        #             if right2 == (0,0,0):
        #                 distances[1] = 0
        #                 distances[2] = 0
        #             print("distance2", left, right, distances)




    def nearest_nonzero_idx(self, a, x, y):
        idx = np.argwhere(a)
        idx = idx[~(idx == [x, y]).all(1)]
        return idx[((idx - [x, y]) ** 2).sum(1).argmin()]

    def get_chest_location(self):
        if self.person_positions is not None:
            chest_locations = []
            for position in self.person_positions:
                x, y, w, h = position

                chest_location = [x + w // 2, y + h // 3]
                xd = int((chest_location[0] - 960) * 0.3673 + 256)
                yd = int((chest_location[1] - 540) * 0.3673 + 212)
                self.chest_depth = 0
                if 0 <= yd < self.depth_frame.shape[0] and 0 <= xd < self.depth_frame.shape[1]:
                    depth_y, depth_x = self.nearest_nonzero_idx(self.depth_frame, yd, xd)
                    depth = self.depth_frame[depth_y, depth_x]
                    self.chest_depth = depth
                else:
                    print("not in frame", xd, yd)
                if self.chest_depth != 0:
                    chest_location.append(self.chest_depth)
                    chest_locations.append(chest_location)
            return chest_locations

    def get_position_from_frame(self, frame_coordinate):
        frame_x, frame_y = frame_coordinate
        depth = self._kinect._mapper.MapCameraPointToDepthSpace(frame_coordinate) 
        print(frame_x, frame_y, depth)

    def quantize(self, img, NUM_CLUSTERS=5):
        ar = np.asarray(img)
        shape = ar.shape
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
        vecs, dist = scipy.cluster.vq.vq(ar, codes)
        return np.reshape(vecs, shape[:2]), codes, vecs



    def retrieve_data(self, draw = True, print_output = False):

        if (self.sensor and self._kinect.has_new_color_frame()) or (not self.sensor and self.frame_name in self.color_files):
            if self.sensor:
                self.color_frame = self._kinect.get_last_color_frame()
            else:
                self.color_frame = np.load(self.folder_path+"color/"+self.frame_name)
            if draw:
                self.draw_color_frame(self.color_frame, self.color_surface)
                pygame.draw.rect(self.color_surface, self.black, ((0,0), self.color_surface.get_size()), 80)
            self.first_frame = True

            if self.topdown:
                view = pygame.surfarray.array3d(self.color_surface)
                view = view.transpose([1, 0, 2])
                img_BGR = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                self.person_positions = detect_persons_with_rescale(img_BGR)
                if self.display_debug:
                    for (x_person, y_person, width_person, height_person) in self.person_positions:
                        pygame.draw.rect(self.color_surface, self.black, ((x_person, y_person), (width_person, height_person)), 10)

            if self.record: np.save(self.folder_path+"/color/frame_"+str(self.frame), self.color_frame)
            
        if (self.sensor and self._kinect.has_new_body_frame()): 
            self._bodies = self._kinect.get_last_body_frame()
            self.new_body_frame = True

        if (self.sensor and self._kinect.has_new_depth_frame()) or (not self.sensor and self.frame_name in self.depth_files):
            if self.sensor:
                self.depth_frame = self._kinect.get_last_depth_frame()

            else:
                self.depth_frame = np.load(self.folder_path+"depth/"+self.frame_name)

            if self.record: np.save(self.folder_path+"/depth/frame_"+str(self.frame), self.depth_frame)
            self.depth_frame = self.depth_frame.reshape(424,512)
            self.new_depth_frame = True

        if self.frame >= 1:
            last_head_locations = copy.copy(self.head_locations)

        if self.sensor:
            if self.body_detection_kinect:
                self.get_head_location()
            else:
                self.head_locations = self.get_chest_location()
        elif self.frame_name in self.heads_files: 
            self.head_locations = [list(element) for element in np.load(self.folder_path+"heads/"+self.frame_name)]

        if self.record: np.save(self.folder_path+"/heads/frame_"+str(self.frame), np.array(self.head_locations))

        for index, head in enumerate(self.head_locations):
            if len(head) == 3:
                if self.frame >= 1:
                    d = self.get_distances(head, last_head_locations, return_zero = True)
                    # print("distance", d)
                    if len(d) > 0 and min(d) < 300:
                        last_coordinate = last_head_locations[d.index(min(d))]
                        id_to_add = last_coordinate[3]
                    else:
                        # if len(d)>0:
                        #     print(min(d))
                        self.head_id_count += 1
                        id_to_add = self.head_id_count
                else:
                    self.head_id_count += 1
                    id_to_add = self.head_id_count
                head.append(id_to_add)
                if self.topdown:
                    if id_to_add not in self.body_status.keys(): self.body_status[id_to_add] = {}
                    self.body_status[id_to_add]["last_added"] = time.time()
            if print_output:
                print("head_location frame",self.frame, self.head_locations)

        if self.topdown:
            to_delete = []
            for id in self.body_status.keys():
                if time.time() - self.body_status[id]["last_added"] > 5:
                    to_delete.append(id)
            for id in to_delete:
                del self.body_status[id]

        if self.sensor and self.topdown:
            if self.head_locations is not None:
                self.head_squares = []
                for location in self.head_locations:
                    if self.body_detection_kinect:
                        top, left, bottom, right = (200, 200, 200, 200)/location[2] * 1000 # (100, 120, 200, 120)
                    else:
                        top, left, bottom, right = (700, 300, -50, 300) / location[2] * 1000
                    width = left+right
                    height = top+bottom
                    if self.display_debug:
                        pygame.draw.rect(self.color_surface, self.blue, ([location[0]-left, location[1]-top], (width, height)), 10)
                        pygame.draw.circle(self.color_surface, self.green, [int(location[0]), int(location[1])], 20)
                    self.head_squares.append((([location[0]-left, location[1]-top], (width, height)), location[3]))

    def mask_detection_color(self, og_image, head_id, show=False):
        if show: cv2.imshow("og_image" + str(head_id), imutils.resize(og_image, width=200))
        dlib_image = og_image.copy()
        image = imutils.resize(dlib_image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        for (i, rect) in enumerate(rects):
            face_image = image[max(rect.top(), 0):max(rect.bottom(), 0), max(rect.left(), 0):max(rect.right(), 0)]
            if show: cv2.imshow("detected_image"+str(head_id), imutils.resize(face_image, width=200))

            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            shape[:, 0] -= rect.left()
            shape[:, 1] -= rect.top()

            quantize_width = 50
            color_ratio = (rect.right()-rect.left())/quantize_width
            dlib_image_small = imutils.resize(face_image, width=quantize_width)
            quantized, codes, vecs = self.quantize(dlib_image_small)

            # top bottom left right
            if show:
                face_box = shape[0][1] - abs(shape[0][1] - shape[8][1]), shape[8][1], shape[0][0], shape[16][0]
                roi = face_image[max(face_box[0], 0):max(face_box[1], 0), max(face_box[2], 0):max(face_box[3], 0)]
                cv2.imshow("roi", imutils.resize(roi, width=200))

            forehead_box = max(shape[24][1], shape[19][1]) - abs(
                max(shape[24][1], shape[19][1]) - shape[28][1]), max(shape[24][1], shape[19][1]) - 2, shape[21][0], \
                           shape[22][0]
            forehead_image = face_image[max(forehead_box[0], 0):max(forehead_box[1], 0),
                             max(forehead_box[2], 0):max(forehead_box[3], 0)]

            forehead_box_q = [int(element/color_ratio) for element in forehead_box]
            forehead_quant = quantized[max(forehead_box_q[0], 0):max(forehead_box_q[1], 0),
                             max(forehead_box_q[2], 0):max(forehead_box_q[3], 0)]
            if show and min(forehead_image.shape) > 0: cv2.imshow("forehead", imutils.resize(forehead_image, width=200))

            left_cheek_box = shape[52][1], shape[5][1], shape[36][0], shape[48][0]-5
            cheek_image = face_image[max(left_cheek_box[0], 0):max(left_cheek_box[1], 0),
                          max(left_cheek_box[2], 0):max(left_cheek_box[3], 0)]
            left_cheek_box_q = [int(element / color_ratio) for element in left_cheek_box]
            cheek_quant = quantized[max(left_cheek_box_q[0], 0):max(left_cheek_box_q[1], 0),
                          max(left_cheek_box_q[2], 0):max(left_cheek_box_q[3], 0)]
            if show and min(cheek_image.shape) > 0: cv2.imshow("left_cheek", imutils.resize(cheek_image, width=200))

            if min(abs(shape[42][0] - shape[0][0]), abs(shape[39][0] - shape[16][0])) > 20:
                forehead_average = cv2.mean(forehead_image)
                cheek_average = cv2.mean(cheek_image)
                diff = self.color_difference(forehead_average[:3], cheek_average[:3])
                forehead_q_average = codes[np.bincount(np.reshape(forehead_quant, forehead_quant.size)).argmax()] if min(forehead_quant.shape) > 0 else None
                cheek_q_average = codes[np.bincount(np.reshape(cheek_quant, cheek_quant.size)).argmax()] if min(cheek_quant.shape) > 0 else None
                if cheek_q_average is not None and forehead_average is not None and forehead_q_average is not None and cheek_q_average is not None:
                    diff_q = self.color_difference(forehead_q_average[:3], cheek_q_average[:3])
                    if show: print(diff, diff_q)

                if show:
                    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                        for (x, y) in shape[i:j]:
                            cv2.circle(face_image, (x, y), 1, (0, 0, 255), -1)

                    cv2.rectangle(face_image, (forehead_box[2], forehead_box[0]), (forehead_box[3], forehead_box[1]), self.red,
                                  thickness=1)
                    cv2.rectangle(face_image, (left_cheek_box[2], left_cheek_box[0]), (left_cheek_box[3], left_cheek_box[1]),
                                  self.red, thickness=1)
                    cv2.imshow("Image", imutils.resize(face_image, width=400))

                    shape = dlib_image_small.shape
                    ar = dlib_image_small.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
                    c = np.zeros(ar.shape, ar.dtype)
                    for i, code in enumerate(codes):
                        c[scipy.r_[scipy.where(vecs == i)], :] = code


                    c = c.reshape(*shape).astype(np.uint8)

                    cv2.rectangle(c, (forehead_box_q[2], forehead_box_q[0]), (forehead_box_q[3], forehead_box_q[1]),
                                  self.red,
                                  thickness=1)
                    cv2.rectangle(c, (left_cheek_box_q[2], left_cheek_box_q[0]),
                                  (left_cheek_box_q[3], left_cheek_box_q[1]),
                                  self.red, thickness=1)

                    cv2.imshow("quantized", imutils.resize(c, width=300))

                if diff < 100:
                    if show: print("geen mondmasker", head_id,  int(diff))
                    return "no mask"
                else:
                    if show: print("wel een mondmasker", head_id,  int(diff))
                    return "mask"
            else:
                print("turned too much", min(abs(shape[42][0] - shape[0][0]), abs(shape[39][0] - shape[16][0])))

    def detect_and_predict_mask(self, frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                     (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()
        # print(detections.shape)

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

    def mask_detection_machine(self, head_image, head_id):

        frame = imutils.resize(head_image, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = self.detect_and_predict_mask(frame, self.faceNet, self.maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            if mask > 0.8:
                return "mask"
            else:
                return "no mask"

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF

    def mask_detection(self, machine=True, color=True):
        if self.first_frame:


            self._frameRGB = self.color_frame.reshape((1080, 1920, -1)).astype(np.uint8)
            self._frameRGB = cv2.resize(self._frameRGB, (0, 0), fx=1,
                                        fy=1)
            image = cv2.cvtColor(self._frameRGB, cv2.COLOR_RGBA2BGR)

            for head_square, head_id in self.head_squares:

                head_image = image[max(int(head_square[0][1]), 0):int(head_square[0][1] + head_square[1][1]),
                             max(int(head_square[0][0]), 0):int(head_square[0][0] + head_square[1][0])]
                head_image = cv2.cvtColor(head_image, cv2.COLOR_RGB2BGR)

                if machine:
                    mask_code_machine = self.mask_detection_machine(head_image, head_id)

                if color:
                    mask_code_color = self.mask_detection_color(head_image, head_id, show=True)

                if not machine and color:
                    mask_code_machine = mask_code_color
                if not color and machine:
                    mask_code_color = mask_code_machine

                mask_code = None

                if mask_code_color is not None:
                    if (mask_code_color == mask_code_machine):
                        mask_code = mask_code_color
                else:
                    mask_code = mask_code_machine

                print(head_id, mask_code)

                if mask_code is not None:
                    if head_id not in self.body_status.keys(): self.body_status[head_id] = {}
                    if "count_mask" not in self.body_status[head_id].keys(): self.body_status[head_id]["count_mask"] = 0
                    if "count_no_mask" not in self.body_status[head_id].keys(): self.body_status[head_id]["count_no_mask"] = 0
                    minimum_count = 1
                    if mask_code == "mask":
                        self.body_status[head_id]["count_mask"] += 1
                        self.body_status[head_id]["count_no_mask"] = 0
                        if self.body_status[head_id]["count_mask"] >= minimum_count:
                            self.body_status[head_id]["mask"] = mask_code
                    else:
                        self.body_status[head_id]["count_mask"] = 0
                        self.body_status[head_id]["count_no_mask"] += 1
                        if self.body_status[head_id]["count_no_mask"] >= minimum_count:
                            self.body_status[head_id]["mask"] = mask_code






if __name__ == "__main__":
    interface = TopDownViewRuntime()
    interface.hands_too_close(500)
