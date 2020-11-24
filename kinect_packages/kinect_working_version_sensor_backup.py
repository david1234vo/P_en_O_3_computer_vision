from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np

import ctypes
import _ctypes
import pygame
import sys
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import traceback

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

white = (255,255,255)
black = (0,0,0)
gray = (200,200,200)
red = (255,0,0)


class TopDownViewRuntime(object):
    def __init__(self):
        #schaal van topdown en color camera
        self.topdown_scale = 1/10
        self.color_scale = 1/4

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

    def draw_infrared_frame(self, frame, target_surface):
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

    #convert x,y in frame to x, y irl
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
            horizontal_factor = 1/1000
            vertical_factor = -1/1000
            x, y, depth = location
            width, height = window_size
            horizontal_coordinate = (x-width/2)*depth*horizontal_factor
            vertical_coordinate = (y-height/2)*depth*vertical_factor
            to_return.append([horizontal_coordinate, vertical_coordinate, depth])
        if not lis:
            return to_return[0]
        else:
            return to_return

    def convert_to_coordinate(self, location, window_size = None):
        if window_size is None:
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
        # print(location1, location2)
        argument = (x1-x2)**2+(y1-y2)**2+(z1-z2)**2
        # print(argument)
        if argument < 0:
            print("argument negative")
            return 0
        return(math.sqrt(argument))

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

    def coordinate_to_pixel(self, location, extra = 0):
        return (int(location[0]*self.topdown_scale + self.topdown_surface_size[0]/2 + extra), int(location[2]*self.topdown_scale + extra))

    def get_middle(self, location1, location2):
        x1, y1, z1 = location1
        x2, y2, z2 = location2
        return ((x1+x2)/2,(y1+y2)/2,(z1+z2)/2)

    def draw_background(self, surface):
        surface.fill(black)
        width, height = surface.get_size()
        pygame.draw.rect(surface, black, ((0,0), (width, height)), 3)
        grid_width = int(width/100)
        grid_height = int(height/100)
        margin = 5
        grid_rectangle_size = 100-margin
        for row in range(grid_height):
            for column in range(grid_width):
                pygame.draw.rect(surface,
                                 white,
                                 [(margin + grid_rectangle_size) * column + margin/2,
                                  (margin + grid_rectangle_size) * row + margin/2,
                                  grid_rectangle_size,
                                  grid_rectangle_size])
        pygame.draw.line(surface, black, self.coordinate_to_pixel((0,0,0)), self.coordinate_to_pixel((21000, 0, 27000)), 10) 
        pygame.draw.line(surface, black, self.coordinate_to_pixel((0,0,0)), self.coordinate_to_pixel((-21000, 0, 27000)), 10) 

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
                        # print("depth coordinate: ",int(head_joint_depth.y), int(head_joint_depth.x), "/color coordinate:", int(head_joint.x), int(head_joint.y))
                    except Exception as e:
                        if "infinity" not in str(e):
                            print("error before return:", e)
            return head_locations

    def draw_heads(self, head_locations):
        combos = []
        # print(head_locations)
        head_coordinates = self.convert_to_coordinates(head_locations)
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
            radius = int(750*self.topdown_scale)

            for second_head in head_locations:
                if second_head != head:
                    
                    if ([head, second_head] not in combos) and ([second_head, head] not in combos):
                        second_coordinate = self.convert_to_coordinates(second_head)
                        distance = self.get_distance(coordinate, second_coordinate)
                        
                        
                        pygame.draw.line(self.topdown_surface, (255, 0, 0), self.coordinate_to_pixel(coordinate), self.coordinate_to_pixel(second_coordinate), 5)
                        if distance > 0:
                            textsurface = self.myfont.render(str(round(distance/1000, 1)).replace(".",",")+" m", False, (0, 0, 255))
                            text_coordinate = self.coordinate_to_pixel(self.get_middle(coordinate, second_coordinate))
                            
                            text_surfaces_to_draw.append([textsurface, text_coordinate])
                        combos.append([head, second_head])
                    # if distance < 1500:
                    #     circle_color = (255,0,0)

            pygame.draw.circle(self.topdown_surface, circle_color, self.coordinate_to_pixel(coordinate), radius, 3)
            pygame.draw.circle(self.topdown_surface, circle_color, self.coordinate_to_pixel(coordinate), 20)

        for textsurface, text_coordinate in text_surfaces_to_draw:
            pygame.draw.rect(self.topdown_surface, gray, ((text_coordinate[0]-5, text_coordinate[1]-5), (textsurface.get_size()[0]+10, textsurface.get_size()[1]+10)))
            pygame.draw.rect(self.topdown_surface, black, ((text_coordinate[0]-5, text_coordinate[1]-5), (textsurface.get_size()[0]+10, textsurface.get_size()[1]+10)), 3)
            self.topdown_surface.blit(textsurface, text_coordinate)

    def get_position_from_frame(self, frame_coordinate):
        frame_x, frame_y = frame_coordinate
        depth = self._kinect._mapper.MapCameraPointToDepthSpace(frame_coordinate) 
        print(frame_x, frame_y, depth)

    def translateAll(self):
        #axis is 0 voor x, 1 voor y en 2, voor z, d is dinstance om te translaten
        # print()
        for index in range(len(self.nodes)):
            self.nodes[index, 0] += self.status["offset"][0]
            self.nodes[index, 1] += self.status["offset"][1]
            # wireframe.translate(axis, d)

    def scaleAll(self, scale):
        """ Scale all wireframes by a given scale, centred on the centre of the screen. """

        centre_x = self.width/2
        centre_y = self.height/2

        for wireframe in self.wireframes.values():
            wireframe.scale((centre_x, centre_y), scale)

    def rotateAll(self, axis, theta):
        """ Rotate all wireframe about their centre, along a given axis by a given angle. """

        rotateFunction = 'rotate' + axis

        for wireframe in self.wireframes.values():
            centre = wireframe.findCentre()
            getattr(wireframe, rotateFunction)(centre, theta)

    def d3d_map(self):
        # http://archive.petercollingridge.co.uk/book/export/html/460
        pygame.init()
        factor = 2
        zoom_factor = 2
        self.screen = pygame.display.set_mode((192*factor*2,108*factor*2+192*factor), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        self.color_surface = pygame.Surface((1920, 1080), 0, 32)
        self.depth_surface = pygame.Surface((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), 0, 24)
        self.node_surface = pygame.Surface((1000*zoom_factor, 1000*zoom_factor), 0, 32)
        frame = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # self.offset = [630*zoom_factor,-440*zoom_factor]
        # self.outer_coordinates = {"max_x": self.offset[0], "max_y": self.offset[1], "min_x": self.offset[0], "min_y": self.offset[1]}
        # self.scaling_factor = 1
        # self.status["rotate"][0] = 0
        # self.status["rotate"][1] = 0
        # self.status["rotate"][2] = 0

        self.status = {"offset": [630*zoom_factor,-440*zoom_factor], "scaling_factor": 1, "rotate": [0,0,0]}

        # key_to_function = {
        # pygame.K_LEFT:   (lambda x: x.translateAll(0, -10)),
        # pygame.K_RIGHT:  (lambda x: x.translateAll(0,  10)),
        # pygame.K_DOWN:   (lambda x: x.translateAll(1,  10)),
        # pygame.K_UP:     (lambda x: x.translateAll(1, -10)),
        # pygame.K_EQUALS: (lambda x: x.scaleAll(1.25)),
        # pygame.K_MINUS:  (lambda x: x.scaleAll( 0.8)),
        # pygame.K_q:      (lambda x: x.rotateAll(0,  0.1)),
        # pygame.K_w:      (lambda x: x.rotateAll(0, -0.1)),
        # pygame.K_a:      (lambda x: x.rotateAll(1,  0.1)),
        # pygame.K_s:      (lambda x: x.rotateAll(1, -0.1)),
        # pygame.K_z:      (lambda x: x.rotateAll(2,  0.1)),
        # pygame.K_x:      (lambda x: x.rotateAll(2, -0.1))}

        

        while True:
            if self._kinect.has_new_color_frame(): 
                color_frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(color_frame, self.color_surface)
                frame += 1
                # print("frame", frame)
            if self._kinect.has_new_depth_frame():
                depth_frame = self._kinect.get_last_depth_frame()
                depth_frame = depth_frame.reshape(424,512)
                self.draw_infrared_frame(depth_frame, self.depth_surface)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:  self.status["offset"][0] += 50*zoom_factor
                    if event.key == pygame.K_RIGHT: self.status["offset"][0] -= 50*zoom_factor
                    if event.key == pygame.K_DOWN:  self.status["offset"][1] += 50*zoom_factor
                    if event.key == pygame.K_UP:    self.status["offset"][1] -= 50*zoom_factor
                    if event.key == pygame.K_EQUALS: self.status["scaling_factor"] += 0.5
                    if event.key == pygame.K_MINUS:  self.status["scaling_factor"] -= 0.5
                    if event.key == pygame.K_a:      self.status["rotate"][0] += math.pi/8
                    if event.key == pygame.K_q:      self.status["rotate"][0] -= math.pi/8
                    if event.key == pygame.K_z:      self.status["rotate"][1] += math.pi/8
                    if event.key == pygame.K_s:      self.status["rotate"][1] -= math.pi/8
                    if event.key == pygame.K_e:      self.status["rotate"][2] += math.pi/8
                    if event.key == pygame.K_d:      self.status["rotate"][2] -= math.pi/8
                    print(self.status)
            
            
            # ax.scatter(0,0,0)
            if frame >= 1:
                xyz = []
                color_list = []

                # for y in range(0,1080, 20):
                #     print(y)
                #     for x in range(0,1920, 20):
                #         try:
                #             frame_color_coordinate = (x,y)
                #             depth_value = depth_frame[int(((frame_color_coordinate[1]-540)*0.3673)+212), int(((frame_color_coordinate[0]-960)*0.3673)+256)]
                #             position = self.convert_to_coordinates([frame_color_coordinate[0], frame_color_coordinate[1], depth_value])
                #             # print(position)
                #             xyz.append([position[0], position[1], position[2]])
                #             # z.append(position[2])
                #             pixel = self.coordinate_to_pixel(position)
                #             pygame.draw.circle(self.color_surface, red, pixel, 3)
                #             # ax.plot(position[0], position[2], position[1], color=(0.5,0,0))
                #         except Exception as e:
                #             pass
                #             # print(x, "failed", e)

                step = 3
                for y in range(0,424, step):
                    # print(y)
                    for x in range(0,512, step):
                        try:
                            # frame_color_coordinate = (x,y)
                            # depth_value = depth_frame[int(((frame_color_coordinate[1]-540)*0.3673)+212), int(((frame_color_coordinate[0]-960)*0.3673)+256)]
                            depth_coordinate = [x, y, depth_frame[y, x]]
                            # color_coordinate = depth_frame[int(((frame_color_coordinate[1]-540)*0.3673)+212), int(((frame_color_coordinate[0]-960)*0.3673)+256)]
                            position = self.convert_to_coordinate(depth_coordinate, (512,424))
                            # print(position)
                            if position != [0,0,0]:
                                xyz.append([position[0], position[1], position[2]/4])
                                try:
                                    # print(a)
                                    color_list.append(self.color_surface.get_at((int(((x-256)/0.3673)+960), int(((y-212)/0.3673)+540))))
                                except Exception:
                                    color_list.append((255,0,0,255))
                            # z.append(position[2])
                            # pixel = self.coordinate_to_pixel(position)
                            # max_depth = 10000
                            # color_to_draw = (min(depth_value/max_depth, 1)*255, 0, 0)
                            # pygame.draw.circle(self.color_surface, color_to_draw, pixel, 3)
                            # ax.plot(position[0], position[2], position[1], color=(0.5,0,0))
                        except Exception as e:
                            # print("error", x, y, e)
                            # print(x, "failed", e)
                            print(traceback.format_exc())
                self.nodes = np.array(xyz)
                # print(self.nodes, self.nodes.shape)
                ones_column = np.ones((len(self.nodes), 1))
                self.nodes = np.hstack((self.nodes, ones_column))
                # print(self.nodes, self.nodes.shape)

                transformation_matrices = []

                z_factor = 1
                c = np.cos(self.status["rotate"][0])*z_factor
                s = np.sin(self.status["rotate"][0])*z_factor
                # rotateXMatrix
                transformation_matrices.append(np.array([[1, 0, 0, 0],
                                                         [0, c,-s, 0],
                                                         [0, s, c, 0],
                                                         [0, 0, 0, 1]]))

                c = np.cos(self.status["rotate"][1])
                s = np.sin(self.status["rotate"][1])
                # rotateYMatrix
                transformation_matrices.append(np.array([[ c, 0, s, 0],
                                                         [ 0, 1, 0, 0],
                                                         [-s, 0, c, 0],
                                                         [ 0, 0, 0, 1]]))

                c = np.cos(self.status["rotate"][2])
                s = np.sin(self.status["rotate"][2])
                # rotateZMatrix
                transformation_matrices.append(np.array([[c,-s, 0, 0],
                                                         [s, c, 0, 0],
                                                         [0, 0, 1, 0],
                                                         [0, 0, 0, 1]]))

                # scaling
                s = (self.status["scaling_factor"], )*3
                transformation_matrices.append(np.array([[s[0], 0,      0,      0],
                                                         [0,    s[1],   0,      0],
                                                         [0,    0,      s[2],   0],
                                                         [0,    0,      0,      1]]))

                #translation
                transformation_matrices.append(np.array([[1,                0,              0,0],
                                                         [0,                1,              0,0],
                                                         [0,                0,              1,0],
                                                         [self.status["offset"][0],   self.status["offset"][1], 0,1]]))

                for transform in transformation_matrices:
                    self.nodes = np.dot(self.nodes, transform)

                # self.outer_coordinates = {"max_x": self.status["offset"][0], "max_y": self.status["offset"][1], "min_x": self.status["offset"][0], "min_y": self.status["offset"][1]}

                # self.translateAll()
                self.node_surface.fill(black)
                for index, node in enumerate(self.nodes):
                    # max_depth = 5000
                    # color_to_draw = (max(10, min(node[2]/max_depth, 1)*255), 0, 0)
                    color_to_draw = color_list[index]
                    # node[0] = self.status["offset"][0] + self.status["scaling_factor"] * (node[0] - self.status["offset"][0])
                    # node[1] = self.status["offset"][1] + self.status["scaling_factor"] * (node[1] - self.status["offset"][1])
                    # node[0] = self.status["scaling_factor"] * node[0]
                    # node[1] = self.status["scaling_factor"] * node[1]
                    # node[2] = self.status["scaling_factor"] * node[2]
                    pygame.draw.circle(self.node_surface, color_to_draw, (int(node[0]), -int(node[1])), 5, 0)
                    # pygame.draw.circle(self.node_surface, color_to_draw, (int(node[0]+self.status["offset"][0]), -int(node[1]+self.status["offset"][1])), 5, 0)
                    # self.outer_coordinates["max_x"] = max(self.outer_coordinates["max_x"], int(node[0]+self.status["offset"][0]))
                    # self.outer_coordinates["min_x"] = min(self.outer_coordinates["min_x"], int(node[0]+self.status["offset"][0]))
                    # self.outer_coordinates["max_y"] = max(self.outer_coordinates["max_y"], -int(node[1]+self.status["offset"][1]))
                    # self.outer_coordinates["min_y"] = min(self.outer_coordinates["min_y"], -int(node[1]+self.status["offset"][1]))
                # self.status["offset"] = [int((self.outer_coordinates["max_x"]-self.outer_coordinates["min_x"])/2), int((self.outer_coordinates["max_y"]-self.outer_coordinates["min_y"])/2)]
                # print(self.outer_coordinates, "/offset:", self.status["offset"], "/average:", [int((self.outer_coordinates["max_x"]-self.outer_coordinates["min_x"])/2), int((self.outer_coordinates["max_y"]-self.outer_coordinates["min_y"])/2)], "/real average:", [np.average(self.nodes[:,0]), np.average(self.nodes[:,1])])



                # ax.scatter([x[0] for x in xyz], [x[2] for x in xyz],[x[1] for x in xyz])
                # plt.show()

                

            self.color_surface_to_draw = pygame.transform.scale(self.color_surface, (192*factor,108*factor));
            self.depth_surface_to_draw = pygame.transform.scale(self.depth_surface, (192*factor,108*factor));
            self.node_surface_to_draw = pygame.transform.scale(self.node_surface, (192*factor*2,192*factor*2));
            self.screen.blit(self.color_surface_to_draw, (0,0))
            self.screen.blit(self.depth_surface_to_draw, (192*factor,0))
            self.screen.blit(self.node_surface_to_draw, (0,108*factor))

            pygame.display.update()
            pygame.display.flip()

            # ax.clear()
            
            # plt.close()


        
        

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    self._screen = pygame.display.set_mode(event.dict['size'],pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        




    def color_and_depth_interface(self):
        pygame.init()
        factor = 4
        self.screen = pygame.display.set_mode((192*factor*2,108*factor), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        self.color_surface = pygame.Surface((1920, 1080), 0, 32)
        # self.depth_surface = pygame.Surface((424, 512), 0, 32)
        self.depth_surface = pygame.Surface((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), 0, 24)
        print("depth size", (self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), "color size", (self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height))
        while True:
            if self._kinect.has_new_color_frame():
                color_frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(color_frame, self.color_surface)

            if self._kinect.has_new_depth_frame():
                depth_frame = self._kinect.get_last_depth_frame()
                self.draw_infrared_frame(depth_frame, self.depth_surface)

            self.color_surface_to_draw = pygame.transform.scale(self.color_surface, (192*factor,108*factor));
            self.depth_surface_to_draw = pygame.transform.scale(self.depth_surface, (192*factor,108*factor));
            self.screen.blit(self.color_surface_to_draw, (0,0))
            self.screen.blit(self.depth_surface_to_draw, (192*factor,0))

            pygame.display.update()
            pygame.display.flip()

    def draw_foreground(self):
        self._screen.blit(self.topdown_surface, self.topdown_position)

        h_to_w = float(self.color_surface.get_height()) / self.color_surface.get_width()
        target_height = int((h_to_w * self._screen.get_width())*self.color_scale)
        surface_to_draw = pygame.transform.scale(self.color_surface, (int(self._screen.get_width()*self.color_scale), target_height));
        color_position = (self.topdown_position[0] + self.topdown_surface.get_size()[0] + 20, self.topdown_position[1])
        self._screen.blit(surface_to_draw, color_position)
        info_position = (color_position[0], color_position[1]+surface_to_draw.get_size()[1]+20)
        self.info_surface = pygame.transform.scale(self.info_surface, (surface_to_draw.get_size()[0], self.topdown_surface.get_size()[1]-surface_to_draw.get_size()[1]-20))
        self.info_surface.fill(white)
        pygame.draw.rect(self.info_surface, black, ((0,0), self.info_surface.get_size()), 5)
        self._screen.blit(self.info_surface, info_position)

        pygame.display.update()
        pygame.display.flip()

    def user_interface(self):
        pygame.init()
        self._clock = pygame.time.Clock()
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((1430,650), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        self.topdown_surface = pygame.Surface(self.topdown_surface_size, 0, 32)
        self.color_surface = pygame.Surface((1920, 1080), 0, 32)
        self.info_surface = pygame.Surface((400,800), 0,32)
        pygame.display.set_caption('Topdown view')
        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self.draw_background(self.topdown_surface)
        self._screen.fill(white)
        while True:
            if self._kinect.has_new_color_frame():
                color_frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(color_frame, self.color_surface)
                pygame.draw.rect(self.color_surface, black, ((0,0), self.color_surface.get_size()), 80)
                
            if self._kinect.has_new_body_frame(): 
                self._bodies = self._kinect.get_last_body_frame()
                self.new_body_frame = True

            if self._kinect.has_new_depth_frame():
                self.depth_frame = self._kinect.get_last_depth_frame()
                self.depth_frame = self.depth_frame.reshape(424,512)
                self.new_depth_frame = True

            head_locations = []
            # head_locations +=  [[0, 0, 0], [1500, 100, 4000]]

            head_location_to_add = self.get_head_location()
            if head_location_to_add is not None: head_locations += head_location_to_add

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    self._screen = pygame.display.set_mode(event.dict['size'],pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                    self._screen.fill(white)


            if head_locations is not None and len(head_locations)>0: 
                self.draw_background(self.topdown_surface)

            self.draw_heads(head_locations)

            self.draw_foreground()





topDownObject = TopDownViewRuntime();
# topDownObject.user_interface()

topDownObject.d3d_map()

# position = topDownObject.get_position_from_frame((1,1))
# print(position)
