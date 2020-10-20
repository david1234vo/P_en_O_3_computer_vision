from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np

import ctypes
import _ctypes
import pygame
import sys
import math

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


class TopDownViewRuntime(object):
    def __init__(self):

        self._done = False
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_Depth)
        self._bodies = None
        pygame.init()
        self._clock = pygame.time.Clock()
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        self.display_size = (1000, 750)
        # self.topdown_surface = pygame.display.set_mode(self.display_size)
        self.color_factor = 0.4
        # self.color_surface = pygame.display.set_mode((int(self.display_size[0]*color_factor), int(self.display_size[1]*color_factor)))
        self.topdown_surface = pygame.Surface(self.display_size, 0, 32)
        self.color_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)
        pygame.display.set_caption('Topdown view')
        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self.scale = 1/10

        self.draw_background(self.topdown_surface)

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def get_window_size(self):
        return (float(self._kinect.color_frame_desc.Width), float(self._kinect.color_frame_desc.Height))

    def convert_to_coordinates(self, location):
        window_size = self.get_window_size()
        horizontal_factor = 1/1000
        vertical_factor = -1/1000
        x, y, depth = location
        width, height = window_size
        horizontal_coordinate = (x-width/2)*depth*horizontal_factor
        vertical_coordinate = (y-height/2)*depth*vertical_factor
        return [horizontal_coordinate, vertical_coordinate, depth]


    def get_distance(self, location1, location2):
        x1, y1, z1 = location1
        x2, y2, z2 = location2
        argument = (x1-x2)**2+(y1-y2)**2+(z1-z2)**2
        if argument < 0:
            return 0
        return(math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2))

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

    def coordinate_to_pixel(self, location, extra = 0):
        return (int(location[0]*self.scale + self.display_size[0]/2 + extra), int(location[2]*self.scale +self.display_size[1]/4 + extra))

    def get_middle(self, location1, location2):
        x1, y1, z1 = location1
        x2, y2, z2 = location2
        return ((x1+x2)/2,(y1+y2)/2,(z1+z2)/2)

    def draw_background(self, surface):
        surface.fill((0,0,0))
        pygame.draw.line(surface, (255,255, 255), (int(self.display_size[0]/2), 0), self.coordinate_to_pixel((7000, 0, 9000))) 
        pygame.draw.line(surface, (255,255, 255), (int(self.display_size[0]/2), 0), self.coordinate_to_pixel((-7000, 0, 9000))) 


    def get_head_location(self):
        # print(self._bodies, "function")
        if self._bodies is not None: 
            # print("test1")
            head_locations = []
            for i in range(0, self._kinect.max_body_count):
                body = self._bodies.bodies[i]
                if not body.is_tracked: 
                    continue 
                
                # print("test2")
                joints = body.joints 
                joint_points = self._kinect.body_joints_to_color_space(joints)
                joint_points_depth = self._kinect.body_joints_to_depth_space(joints)

                if self.new_body_frame and self.new_depth_frame:
                    # print("test3")
                    try:
                        head_joint = joint_points[PyKinectV2.JointType_Head]
                        head_joint_depth = joint_points_depth[PyKinectV2.JointType_Head]
                        depth_value = self.depth_frame[int(head_joint_depth.y), int(head_joint_depth.x)]
                        head_locations.append([head_joint.x, head_joint.y, depth_value])
                    except Exception as e:
                        print("error before return:", e)
            return head_locations

    def run(self):


        self.new_depth_frame = False
        self.new_body_frame = False

        while True:
            if self._kinect.has_new_color_frame():

                color_frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(color_frame, self.color_surface)
                

            if self._kinect.has_new_body_frame(): 
                self._bodies = self._kinect.get_last_body_frame()
                # print(self._bodies, "internal")
                self.new_body_frame = True

            if self._kinect.has_new_depth_frame():
                print("cframe")
                self.depth_frame = self._kinect.get_last_depth_frame()
                self.depth_frame = self.depth_frame.reshape(424,512)
                self.new_depth_frame = True

            head_locations = []
            # head_locations +=  [(0, 0, 0), [4000, 100, 1000]]

            head_location_to_add = self.get_head_location()
            if head_location_to_add is not None: head_locations += head_location_to_add


            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    self._screen = pygame.display.set_mode(event.dict['size'],pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

            if head_locations is not None and len(head_locations)>0: 
                self.draw_background(self.topdown_surface)

            combos = []

            # print(head_locations)

            for head in head_locations:
                coordinate = self.convert_to_coordinates(head)
                circle_color = (100, 200, 100)
                radius = int(750*self.scale)
                for second_head in head_locations:
                    if second_head != head:
                        second_coordinate = self.convert_to_coordinates(second_head)
                        distance = self.get_distance(coordinate, second_coordinate)
                        if ([head, second_head] not in combos) and ([second_head, head] not in combos):
                            
                            pygame.draw.line(self.topdown_surface, (255, 0, 0), self.coordinate_to_pixel(coordinate), self.coordinate_to_pixel(second_coordinate))
                            textsurface = self.myfont.render(str(round(distance/1000, 2)), False, (0, 0, 255))
                            self.topdown_surface.blit(textsurface, self.coordinate_to_pixel(self.get_middle(coordinate, second_coordinate)))
                            combos.append([head, second_head])
                        if distance < 1500:
                            circle_color = (255,0,0)

                pygame.draw.circle(self.topdown_surface, circle_color, self.coordinate_to_pixel(coordinate), 20)
                pygame.draw.circle(self.topdown_surface, circle_color, self.coordinate_to_pixel(coordinate), radius, 3)


            self._screen.blit(self.topdown_surface, (0,0))
            h_to_w = float(self.color_surface.get_height()) / self.color_surface.get_width()
            target_height = int((h_to_w * self._screen.get_width())*self.color_factor)
            surface_to_draw = pygame.transform.scale(self.color_surface, (int(self._screen.get_width()*self.color_factor), target_height));
            self._screen.blit(surface_to_draw, (int(self._screen.get_width() - surface_to_draw.get_width()),0))

            pygame.display.update()
            pygame.display.flip()






topDownObject = TopDownViewRuntime();
topDownObject.run()