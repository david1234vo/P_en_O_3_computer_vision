from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np

import ctypes
import _ctypes
import pygame
import sys
import math

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


class BodyGameRuntime(object):
    def __init__(self):
        # pygame.init()

        # self._clock = pygame.time.Clock()
        # self._infoObject = pygame.display.Info()
        # self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               # pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        # pygame.display.set_caption("Kinect for Windows v2 Body Game")

        self._done = False
        # self._clock = pygame.time.Clock()
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_Depth)

        # self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)
        # self._depth_surface = pygame.Surface((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), 0, 32)

        self._bodies = None


    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good 
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);
    
        # Right Arm    
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # Right Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);

        # Left Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);


    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def draw_depth_frame(self, frame, target_surface):
        if frame is None:  # some usb hub do not provide the infrared image. it works with Kinect studio though
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

    def run(self):

        depth_list = []

        # -------- Main Program Loop -----------
        # while not self._done:
        # --- Main event loop
        # for event in pygame.event.get(): # User did something
        #     if event.type == pygame.QUIT: # If user clicked close
        #         self._done = True # Flag that we are done so we exit this loop
        #         self._kinect.close()
        #         pygame.quit()
        #         return "quit"

            # elif event.type == pygame.VIDEORESIZE: # window resized
            #     self._screen = pygame.display.set_mode(event.dict['size'], 
            #                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                
        # --- Game logic should go here

        # --- Getting frames and drawing  
        # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data 

        new_depth_frame = False
        new_body_frame = False
        color_frame = None

        if self._kinect.has_new_color_frame():
            color_frame = self._kinect.get_last_color_frame()
            # self.draw_color_frame(color_frame, self._frame_surface)

        if self._kinect.has_new_depth_frame():
            # print("new frame")
            depth_frame = self._kinect.get_last_depth_frame()
            # self.draw_depth_frame(depth_frame, self._depth_surface)
            depth_frame = depth_frame.reshape(424,512)
            # print(depth_frame)
            new_depth_frame = True

        # --- Cool! We have a body frame, so can get skeletons
        if self._kinect.has_new_body_frame(): 
            self._bodies = self._kinect.get_last_body_frame()
            new_body_frame = True

        head_locations = []

        # --- draw skeletons to _frame_surface
        if self._bodies is not None: 
            for i in range(0, self._kinect.max_body_count):
                body = self._bodies.bodies[i]
                if not body.is_tracked: 
                    continue 
                
                joints = body.joints 
                # print(joints)
                # convert joint coordinates to color space 
                joint_points = self._kinect.body_joints_to_color_space(joints)
                joint_points_depth = self._kinect.body_joints_to_depth_space(joints)

                # print(joint_points)
                if new_body_frame and new_depth_frame:
                    head_joint = joint_points[PyKinectV2.JointType_Head]
                    head_joint_depth = joint_points_depth[PyKinectV2.JointType_Head]
                    depth_value = depth_frame[int(head_joint_depth.y), int(head_joint_depth.x)]
                    # print(head_joint.x, head_joint.y, depth_value)
                    head_locations.append([head_joint.x, head_joint.y, depth_value])
                    depth_list.append(depth_value)
                    # print(depth_frame, type(depth_frame), depth_frame.shape, int(head_joint_depth.x), int(head_joint_depth.y))

                    # print(depth_frame[int(head_joint_depth.x), int(head_joint_depth.y)])
                    # pygame.draw.circle(self._frame_surface, (100, 200, 100), (int(head_joint.x), int(head_joint.y)), 10)
                    # self.draw_body(joints, joint_points, SKELETON_COLORS[i])

        # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
        # --- (screen size may be different from Kinect's color frame size) 
        # h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
        # target_height = int(h_to_w * self._screen.get_width())
        # surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
        # self._screen.blit(surface_to_draw, (0,0))
        # surface_to_draw = pygame.transform.scale(self._depth_surface, (self._screen.get_width(), target_height));
        # self._screen.blit(self._depth_surface, (700,0))
        surface_to_draw = None

        # pygame.display.update()

        # --- Go ahead and update the screen with what we've drawn.
        # pygame.display.flip()

        # --- Limit to 60 frames per second
        # self._clock.tick(60)

        return head_locations, color_frame

        # Close our Kinect sensor, close the window and quit.
        

        # import matplotlib.pyplot as plt
        # plt.scatter(list(range(len(depth_list))), depth_list)
        # plt.show()


pygame.init()
display_size = (1000, 600)
DISPLAYSURF = pygame.display.set_mode(display_size)
pygame.display.set_caption('Hello World!')
pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 30)

__main__ = "Kinect v2 Body Game"
game = BodyGameRuntime();
window_size = game.get_window_size()
heads_depth = []
heads_x = []
heads_y = []
head_locations = {}

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

import pygame, sys
from pygame.locals import *


draw_background(DISPLAYSURF)




while True:
    head_locations, color_frame = game.run();
    # if len(head_locations) > 0:
    #     head_locations +=  [(0, 0, 0), [4000, 100, 1000]]
    # for head in new_head_locations:
    #     key = get_key_nearest(head, head_locations)
    #     if key is None:
    #         key = 1
    #     elif get_distance(dict[key], head) > 300:
    #         key = max(dict.get_keys())+1
    #     head_locations[key] = head
    if head_locations == "quit":
        break
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    if head_locations is not None and len(head_locations)>0:
        draw_background(DISPLAYSURF)

    combos = []

    for head in head_locations:
        coordinate = convert_to_coordinates(head, window_size)
        print(coordinate)   
        heads_depth.append(coordinate[2])
        heads_x.append(coordinate[0])
        heads_y.append(coordinate[1])
        radius = int(750*scale)
        # pygame.draw.circle(DISPLAYSURF, (100, 200, 100), (int(coordinate[0]*scale + display_size[0]/2), int(coordinate[2]*scale +display_size[1]/4)), radius)
        pygame.draw.circle(DISPLAYSURF, (100, 200, 100), coordinate_to_pixel(coordinate), 20)
        pygame.draw.circle(DISPLAYSURF, (100, 200, 100), coordinate_to_pixel(coordinate), radius, 3)
        # print("drawing circle", )
        for second_head in head_locations:
            if second_head != head:
                if ([head, second_head] not in combos) and ([second_head, head] not in combos):
                    # print("drawing line", coordinate_to_pixel(head), coordinate_to_pixel(second_head))
                    second_coordinate = convert_to_coordinates(second_head, window_size)
                    pygame.draw.line(DISPLAYSURF, (255, 0, 0), coordinate_to_pixel(coordinate), coordinate_to_pixel(second_coordinate))
                    # pygame.draw.circle(DISPLAYSURF,(0, 0, 255), coordinate_to_pixel(get_middle(coordinate, second_coordinate)), 20)
                    textsurface = myfont.render(str(round(get_distance(coordinate, second_coordinate)/1000, 2)), False, (0, 0, 255))
                    DISPLAYSURF.blit(textsurface, coordinate_to_pixel(get_middle(coordinate, second_coordinate)))
                    combos.append([head, second_head])
        print("")


    pygame.display.update()





import matplotlib.pyplot as plt
print("showing")
# plt.scatter(list(range(len(heads_depth))), heads_depth)
# plt.show()
# plt.close()
plt.scatter(heads_x, heads_depth)
plt.show()
# print()
# print("heads_x", heads_x)
# print()
# print("heads_y", heads_y)
