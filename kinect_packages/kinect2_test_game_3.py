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


    def get_window_size(self):
        return (float(self._kinect.color_frame_desc.Width), float(self._kinect.color_frame_desc.Height))

    def run(self):


        new_depth_frame = False
        new_body_frame = False
        color_frame = None

        if self._kinect.has_new_color_frame():
            color_frame = self._kinect.get_last_color_frame()

        if self._kinect.has_new_depth_frame():
            depth_frame = self._kinect.get_last_depth_frame()
            depth_frame = depth_frame.reshape(424,512)
            new_depth_frame = True

        if self._kinect.has_new_body_frame(): 
            self._bodies = self._kinect.get_last_body_frame()
            new_body_frame = True

        head_locations = []

        if self._bodies is not None: 
            for i in range(0, self._kinect.max_body_count):
                body = self._bodies.bodies[i]
                if not body.is_tracked: 
                    continue 
                
                joints = body.joints 
                joint_points = self._kinect.body_joints_to_color_space(joints)
                joint_points_depth = self._kinect.body_joints_to_depth_space(joints)

                if new_body_frame and new_depth_frame:
                    head_joint = joint_points[PyKinectV2.JointType_Head]
                    head_joint_depth = joint_points_depth[PyKinectV2.JointType_Head]
                    depth_value = depth_frame[int(head_joint_depth.y), int(head_joint_depth.x)]
                    head_locations.append([head_joint.x, head_joint.y, depth_value])
        surface_to_draw = None


        return head_locations, color_frame



pygame.init()
display_size = (1000, 600)
topdown_surface = pygame.display.set_mode(display_size)
pygame.display.set_caption('Topdown view')
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 30)

__main__ = "Kinect v2 Body Game"
topDownObject = TopDownViewRuntime();
window_size = topDownObject.get_window_size()

scale = 1/10

def convert_to_coordinates(location, window_size):
    horizontal_factor = 1/1000
    vertical_factor = -1/1000
    x, y, depth = location
    width, height = window_size
    # print(x, y, depth, width, height, type(x), type(width), x-width/2, (x-width/2)*horizontal_factor)
    horizontal_coordinate = (x-width/2)*depth*horizontal_factor
    vertical_coordinate = (y-height/2)*depth*vertical_factor
    return [horizontal_coordinate, vertical_coordinate, depth]


def get_distance(location1, location2):
    x1, y1, z1 = location1
    x2, y2, z2 = location2
    argument = (x1-x2)**2+(y1-y2)**2+(z1-z2)**2
    if argument < 0:
        return 0
    return(math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2))

def get_key_nearest(location, dict):
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

def d3_to_d2(location):
    return [location[0], location[2]]

def coordinate_to_pixel(location, extra = 0):
    return (int(location[0]*scale + display_size[0]/2 + extra), int(location[2]*scale +display_size[1]/4 + extra))

def get_middle(location1, location2):
    x1, y1, z1 = location1
    x2, y2, z2 = location2
    return ((x1+x2)/2,(y1+y2)/2,(z1+z2)/2)

def draw_background(surface):
    surface.fill((0,0,0))
    pygame.draw.line(surface, (255,255, 255), (int(display_size[0]/2), 0), coordinate_to_pixel((3500, -460, 4500))) 
    pygame.draw.line(surface, (255,255, 255), (int(display_size[0]/2), 0), coordinate_to_pixel((-3500, -460, 4500))) 




draw_background(topdown_surface)


while True:
    head_locations, color_frame = topDownObject.run();
    if head_locations == "quit":
        break
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    if head_locations is not None and len(head_locations)>0:
        draw_background(topdown_surface)

    combos = []

    for head in head_locations:
        coordinate = convert_to_coordinates(head, window_size)
        print(coordinate)   
        radius = int(750*scale)
        pygame.draw.circle(topdown_surface, (100, 200, 100), coordinate_to_pixel(coordinate), 20)
        pygame.draw.circle(topdown_surface, (100, 200, 100), coordinate_to_pixel(coordinate), radius, 3)
        for second_head in head_locations:
            if second_head != head:
                if ([head, second_head] not in combos) and ([second_head, head] not in combos):
                    second_coordinate = convert_to_coordinates(second_head, window_size)
                    pygame.draw.line(topdown_surface, (255, 0, 0), coordinate_to_pixel(coordinate), coordinate_to_pixel(second_coordinate))
                    textsurface = myfont.render(str(round(get_distance(coordinate, second_coordinate)/1000, 2)), False, (0, 0, 255))
                    topdown_surface.blit(textsurface, coordinate_to_pixel(get_middle(coordinate, second_coordinate)))
                    combos.append([head, second_head])


    pygame.display.update()

