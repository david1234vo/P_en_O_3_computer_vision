from pykinect2_8 import PyKinectV2
from pykinect2_8.PyKinectV2 import *
from pykinect2_8 import PyKinectRuntime
import numpy as np
from code_for_kinect.followfacekinect import *
from code_for_kinect.eigenmaskdetectionkinect.maskdetectioncolor import *
from code_for_kinect.eigenpersondetection.persondetectionslice import *
import ctypes
import _ctypes
import pygame
import sys
import math
import os
import time
import cv2

import pygame, sys
from pygame.locals import *

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies 
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                   pygame.color.THECOLORS["blue"],
                   pygame.color.THECOLORS["green"],
                   pygame.color.THECOLORS["orange"],
                   pygame.color.THECOLORS["purple"],
                   pygame.color.THECOLORS["yellow"],
                   pygame.color.THECOLORS["violet"]]

white = (255, 255, 255)
black = (0, 0, 0)
gray = (200, 200, 200)
red = (255, 0, 0)
orange = (255, 165, 0)
green = (0, 255, 0)
mask_strings = ("thanks for wearing MASK", "wear MASK properly", "please wear MASK")
face_rectangle_colors = (green, orange, red)


class TopDownViewRuntime(object):
    def __init__(self):

        # save folder locatie
        self.folder_name = "kinect_recording_1"
        self.folder_path = "C:/Users/lucas/Downloads/kinect_videos/" + self.folder_name + "/"

        # fps voor het afspelen van de savefile
        self.fps = 90

        # schaal van topdown en color camera
        self.topdown_scale = 1 / 10
        self.color_scale = 3 / 8

        # grootte van topdown surface (width, height)
        self.topdown_surface_size = (1000, 600)

        self._done = False
        self._kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_Depth)
        self._bodies = None
        self.person_positions = []
        self.frame = 0
        self.begin_time = time.time()
        self.color_files = [f for f in os.listdir(self.folder_path + "color")]
        self.depth_files = [f for f in os.listdir(self.folder_path + "depth")]
        self.heads_files = [f for f in os.listdir(self.folder_path + "heads")]
        self.last_frame = max([int(element.split("_")[1].split(".")[0]) for element in self.heads_files])
        self.topdown_position = (20, 20)
        self.new_depth_frame = False
        self.new_body_frame = False
        self.chest_depth = (0, 0)

        pygame.init()
        self._clock = pygame.time.Clock()
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((1920, 650), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)
        self.topdown_surface = pygame.Surface(self.topdown_surface_size, 0, 32)
        self.color_surface = pygame.Surface((1920, 1080), 0, 32)
        self.info_surface = pygame.Surface((400, 800), 0, 32)
        pygame.display.set_caption('Topdown view')
        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self.mask_strings = ("thanks for wearing MASK", "wear MASK properly", "please wear MASK")
        self.draw_background(self.topdown_surface)
        self._screen.fill(white)

    def draw_color_frame(self, frame, target_surface):
        # target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address

    def get_window_size(self):
        return (float(self._kinect.color_frame_desc.Width), float(self._kinect.color_frame_desc.Height))

    def convert_to_coordinates(self, locations):
        if len(locations) == 0:
            return
        lis = True
        to_return = []
        if type(locations[0]) != list and type(locations[0]) != tuple:
            locations = [locations]
            lis = False
        for location in locations:
            window_size = self.get_window_size()
            horizontal_factor = 1 / 1000
            vertical_factor = -1 / 1000
            x, y, depth = location
            width, height = window_size
            horizontal_coordinate = (x - width / 2) * depth * horizontal_factor
            vertical_coordinate = (y - height / 2) * depth * vertical_factor
            to_return.append([horizontal_coordinate, vertical_coordinate, depth])
        if not lis:
            return to_return[0]
        else:
            return to_return

    def get_distance(self, location1, location2):
        x1, y1, z1 = location1
        x2, y2, z2 = location2
        print(location1, location2)
        argument = (float(x1) - float(x2)) ** 2 + (float(y1) - float(y2)) ** 2 + (float(z1) - float(z2)) ** 2
        print(argument)
        if argument < 0:
            print("argument negative")
            return 0
        return (math.sqrt(argument))

    def get_distances(self, location, list_location):
        to_return = []
        for second_location in list_location:
            d = self.get_distance(location, second_location)
            if d != 0:
                to_return.append(d)
        return to_return

    def get_key_nearest(self, location, dict):
        if len(dict) > 0 or dict == {}:
            return None
        print(dict, len(dict), dict == {})
        nearest_value = list(dict.values())[0]
        nearest_key = list(dict.keys())[0]
        for key in list(dict.keys()):
            value = dict[key]
            if get_distance(location, dict[key]) < get_distance(location, nearest_value):
                nearest_key = key
                nearest_value = dict[key]
        return nearest_key

    def d3_to_d2(self, location):
        return [location[0], location[2]]

    def coordinate_to_pixel(self, location, extra=0):
        return (int(location[0] * self.topdown_scale + self.topdown_surface_size[0] / 2 + extra),
                int(location[2] * self.topdown_scale + extra))

    def get_middle(self, location1, location2):
        x1, y1, z1 = location1
        x2, y2, z2 = location2
        return ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)

    def draw_background(self, surface):
        surface.fill(black)
        width, height = surface.get_size()
        pygame.draw.rect(surface, black, ((0, 0), (width, height)), 3)
        grid_width = int(width / 100)
        grid_height = int(height / 100)
        margin = 5
        grid_rectangle_size = 100 - margin
        for row in range(grid_height):
            for column in range(grid_width):
                pygame.draw.rect(surface,
                                 white,
                                 [(margin + grid_rectangle_size) * column + margin / 2,
                                  (margin + grid_rectangle_size) * row + margin / 2,
                                  grid_rectangle_size,
                                  grid_rectangle_size])
        pygame.draw.line(surface, black, self.coordinate_to_pixel((0, 0, 0)),
                         self.coordinate_to_pixel((21000, 0, 27000)), 10)
        pygame.draw.line(surface, black, self.coordinate_to_pixel((0, 0, 0)),
                         self.coordinate_to_pixel((-21000, 0, 27000)), 10)

    def get_head_location(self):
        if self._bodies is not None:
            head_locations = []
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
                        head_locations.append([head_joint.x, head_joint.y, depth_value])
                    except Exception as e:
                        if "infinity" not in str(e):
                            print("error before return:", e)
            return head_locations

    def get_chest_location(self):
        if self.person_positions is not None:
            chest_locations = []
            for position in self.person_positions:
                x, y, w, h = position
                chest_location = [x + w // 2, y + h // 3]
                xd = int((chest_location[0] - 960) * 0.3673 + 256)
                yd = int((chest_location[1] - 540) * 0.3673 + 212)
                if 0 <= xd < self.depth_frame.shape[0] and 0 <= yd < self.depth_frame.shape[1]:
                    depth = self.depth_frame[xd, yd]
                    while depth == 0:
                        yd -= 1
                        depth = self.depth_frame[xd, yd]
                    self.chest_depth = self.depth_frame[xd, yd]
                chest_location.append(self.chest_depth)
                chest_locations.append(chest_location)
            #print(chest_locations)
            return chest_locations

    def draw_heads(self, head_locations):
        combos = []
        # print(head_locations)
        print(head_locations)
        head_coordinates = self.convert_to_coordinates(head_locations)
        print(head_coordinates)
        text_surfaces_to_draw = []
        too_close = 0
        for head in head_locations:
            coordinate = self.convert_to_coordinates(head)
            # print(head_coordinates, self.get_distances(coordinate, head_coordinates))
            try:
                nearest = min(self.get_distances(coordinate, head_coordinates))
            except Exception as e:
                nearest = 3000
                # print(e, head_coordinates, self.get_distances(coordinate, head_coordinates))
            if nearest > 1500:
                # print(nearest)
                circle_color = (100, 200, 100)
            else:
                circle_color = (255, 0, 0)
                too_close += 1
            radius = int(750 * self.topdown_scale)

            for second_head in head_locations:
                if second_head != head:

                    if ([head, second_head] not in combos) and ([second_head, head] not in combos):
                        second_coordinate = self.convert_to_coordinates(second_head)
                        distance = self.get_distance(coordinate, second_coordinate)

                        pygame.draw.line(self.topdown_surface, (255, 0, 0), self.coordinate_to_pixel(coordinate),
                                         self.coordinate_to_pixel(second_coordinate), 5)
                        if distance > 0:
                            textsurface = self.myfont.render(str(round(distance / 1000, 1)).replace(".", ",") + " m",
                                                             False, (0, 0, 255))
                            text_coordinate = self.coordinate_to_pixel(self.get_middle(coordinate, second_coordinate))

                            text_surfaces_to_draw.append([textsurface, text_coordinate])
                        combos.append([head, second_head])
                    # if distance < 1500:
                    #     circle_color = (255,0,0)

            pygame.draw.circle(self.topdown_surface, circle_color, self.coordinate_to_pixel(coordinate), radius, 3)
            pygame.draw.circle(self.topdown_surface, circle_color, self.coordinate_to_pixel(coordinate), 20)

        for textsurface, text_coordinate in text_surfaces_to_draw:
            pygame.draw.rect(self.topdown_surface, gray, ((text_coordinate[0] - 5, text_coordinate[1] - 5), (
            textsurface.get_size()[0] + 10, textsurface.get_size()[1] + 10)))
            pygame.draw.rect(self.topdown_surface, black, ((text_coordinate[0] - 5, text_coordinate[1] - 5), (
            textsurface.get_size()[0] + 10, textsurface.get_size()[1] + 10)), 3)
            self.topdown_surface.blit(textsurface, text_coordinate)

    def draw_foreground(self):
        self._screen.blit(self.topdown_surface, self.topdown_position)

        h_to_w = float(self.color_surface.get_height()) / self.color_surface.get_width()
        target_height = int((h_to_w * self._screen.get_width()) * self.color_scale)
        surface_to_draw = pygame.transform.scale(self.color_surface,
                                                 (int(self._screen.get_width() * self.color_scale), target_height));
        color_position = (self.topdown_position[0] + self.topdown_surface.get_size()[0] + 20, self.topdown_position[1])
        self._screen.blit(surface_to_draw, color_position)
        info_position = (color_position[0], color_position[1] + surface_to_draw.get_size()[1] + 20)
        self.info_surface = pygame.transform.scale(self.info_surface, (
        surface_to_draw.get_size()[0], self.topdown_surface.get_size()[1] - surface_to_draw.get_size()[1] - 20))
        self.info_surface.fill(white)
        pygame.draw.rect(self.info_surface, black, ((0, 0), self.info_surface.get_size()), 5)
        self._screen.blit(self.info_surface, info_position)

        pygame.display.update()
        pygame.display.flip()

    def run(self):
        while True:
            timer = cv2.getTickCount()
            frame_name = "frame_" + str(self.frame) + ".npy"
            if frame_name in self.color_files:
                self.color_frame = np.load(self.folder_path + "color/" + frame_name)
                self.draw_color_frame(self.color_frame, self.color_surface)
                view = pygame.surfarray.array3d(self.color_surface)
                view = view.transpose([1, 0, 2])
                img_BGR = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                self.person_positions = detect_persons_with_rescale(img_BGR)
                for (x_person, y_person, width_person, height_person) in self.person_positions:
                    pygame.draw.rect(self.color_surface, black, ((x_person, y_person), (width_person, height_person)),
                                     10)

                # face_positions = maskdetectioncolor(img_BGR, 50)
                # for (x_face, y_face, width_face, height_face) in face_positions:
                #   pygame.draw.rect(self.color_surface, green,
                #                     ((x_face, y_face), (width_face, height_face)), 10)
                # for (x_face, y_face, width_face, height_face, mask_code) in face_positions:
                #    pygame.draw.rect(self.color_surface, face_rectangle_colors[mask_code], ((x_face, y_face), (width_face, height_face)), 10)
                # textsurface = self.myfont.render(mask_strings[mask_code], False, (0, 0, 255))
                # text_coordinate = (x_face, y_face+height_face+10)
                # self.color_surface.blit(textsurface, text_coordinate)
                pygame.draw.rect(self.color_surface, black, ((0, 0), self.color_surface.get_size()), 80)
                fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))
                fps_text = self.myfont.render(str(fps), False, black)
                text_coordinate = (50, 50)
                self.color_surface.blit(fps_text, text_coordinate)

            if frame_name in self.depth_files:
                self.depth_frame = np.load(self.folder_path + "depth/" + frame_name)
                self.depth_frame = self.depth_frame.reshape(424, 512)
                self.new_depth_frame = True

            if frame_name in self.heads_files:
                head_locations = [list(element) for element in np.load(self.folder_path + "heads/" + frame_name)]

            head_locations = self.get_chest_location()

            if head_locations is not None:
                for head_location in head_locations:
                    pygame.draw.circle(self.color_surface, red, (head_location[0],head_location[1]), 10, 0)

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    self._screen = pygame.display.set_mode(event.dict['size'],
                                                           pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)
                    self._screen.fill(white)

            if head_locations is not None and len(head_locations) > 0:
                self.draw_background(self.topdown_surface)

                self.draw_heads(head_locations)

            self.draw_foreground()
            self.frame = int((time.time() - self.begin_time) * self.fps)
            if self.frame > self.last_frame:
                break


topDownObject = TopDownViewRuntime();
topDownObject.run()
