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
import os
import copy

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
            x, y, depth, _ = location
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
        # print("1:", location1, location1[0:3], "\n","2:", location2 , location2[0:3], "\n")
        x1, y1, z1 = location1[0:3]
        x2, y2, z2 = location2[0:3]
        # print(location1, location2)
        argument = (x1-x2)**2+(y1-y2)**2+(int(z1)-int(z2))**2
        
        if argument < 0:
            # print(location1, location2, (x1-x2)**2,(y1-y2)**2,z1, z2,int(z1)-int(z2), (int(z1)-int(z2))**2, argument)
            # print("argument negative")
            return 0
        return(math.sqrt(argument))

    def get_distances(self, location, list_location, return_zero = False):
        to_return = []
        for second_location in list_location:
            d = self.get_distance(location, second_location)
            if d != 0 or return_zero:
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
                        # print("depth coordinate: ",int(head_joint_depth.y), int(head_joint_depth.x), "/color coordinate:", int(head_joint.x), int(head_joint.y))
                    except Exception as e:
                        if "infinity" not in str(e):
                            print("error before return:", e)

    def draw_heads(self):
        combos = []
        # print(self.head_locations)
        head_coordinates = self.convert_to_coordinates(self.head_locations)
        text_surfaces_to_draw = []
        too_close = 0
        for head in self.head_locations:
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

            for second_head in self.head_locations:
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

            textsurface = self.myfont.render(str(head[3]), False, (0, 0, 255))
            text_coordinate = self.coordinate_to_pixel(coordinate)            
            text_surfaces_to_draw.append([textsurface, text_coordinate])

        for textsurface, text_coordinate in text_surfaces_to_draw:
            pygame.draw.rect(self.topdown_surface, gray, ((text_coordinate[0]-5, text_coordinate[1]-5), (textsurface.get_size()[0]+10, textsurface.get_size()[1]+10)))
            pygame.draw.rect(self.topdown_surface, black, ((text_coordinate[0]-5, text_coordinate[1]-5), (textsurface.get_size()[0]+10, textsurface.get_size()[1]+10)), 3)
            self.topdown_surface.blit(textsurface, text_coordinate)

    def get_position_from_frame(self, frame_coordinate):
        frame_x, frame_y = frame_coordinate
        depth = self._kinect._mapper.MapCameraPointToDepthSpace(frame_coordinate) 
        print(frame_x, frame_y, depth)


    def d3d_map(self):
        # http://archive.petercollingridge.co.uk/book/export/html/460
        self.show_color_pixel = False

        pygame.init()
        factor = 2
        zoom_factor = 2
        self.screen = pygame.display.set_mode((192*factor*2,108*factor*2+192*factor), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        self.color_surface = pygame.Surface((1920, 1080), 0, 32)
        self.depth_surface = pygame.Surface((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), 0, 24)
        self.node_surface = pygame.Surface((1000*zoom_factor, 1000*zoom_factor), 0, 32)
        frame = 0
        got_frame = False
        begin_time = time.time()

        self.debug_time = {"mapping": 0, "transforming": 0, "displaying": 0, "new_mapping": 0}
        self.status = {"offset": [630*zoom_factor,-440*zoom_factor], "scaling_factor": 1, "rotate": [0,0,0]}

        while True:
            if got_frame:
                frame += 1
                if frame == 1:
                    begin_time = time.time()
            if self._kinect.has_new_color_frame(): 
                color_frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(color_frame, self.color_surface)
                got_frame = True
                passed_time = time.time()-begin_time
                print("frame", frame, round(passed_time, 2), round(frame/passed_time, 2))
            if self._kinect.has_new_depth_frame():
                depth_frame_og = self._kinect.get_last_depth_frame()
                depth_frame = depth_frame_og.reshape(424,512)
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
                    if event.key == pygame.K_q:      self.status["rotate"][0] += math.pi/8 #a
                    if event.key == pygame.K_a:      self.status["rotate"][0] -= math.pi/8 #q
                    if event.key == pygame.K_w:      self.status["rotate"][1] += math.pi/8 #z
                    if event.key == pygame.K_s:      self.status["rotate"][1] -= math.pi/8
                    if event.key == pygame.K_e:      self.status["rotate"][2] += math.pi/8
                    if event.key == pygame.K_d:      self.status["rotate"][2] -= math.pi/8
                    print(event.key, self.status)
            
            if frame >= 1:
                xyz = []
                color_list = []

                step = 2
                width, height = 512,424
                n_width, n_height = int(width/step), int(height/step)
                c_width, c_height = n_width*step, n_height*step

                map_time = time.time()

                y = np.repeat(np.arange(0, c_width, step), n_height)
                x = np.repeat(np.arange(0, c_width, step), n_height).reshape(n_height, n_width, order='F').ravel()
                z = depth_frame[0:c_height:step, 0:c_width:step].reshape(int(c_width*c_height/step**2), 1)
                np_depth_frame = np.c_[x, y, z]
                x = (np_depth_frame[:, 0] - width / 2) * np_depth_frame[:, 2] / 1000
                y = -(np_depth_frame[:, 1] - height / 2) * np_depth_frame[:, 2] / 1000
                xyz = np.c_[x, y, z/4]

                self.debug_time["new_mapping"] += time.time() - map_time


                # map_time = time.time()
                # for y in range(0,424, step):
                #     for x in range(0,512, step):
                #         depth_coordinate = [x, y, depth_frame[y, x]]
                #         position = self.convert_to_coordinate(depth_coordinate, (512,424))
                #         if position != [0,0,0]:
                #             xyz.append([position[0], position[1], position[2]/4])
                #             if self.show_color_pixel:
                #                 try:
                #                     color_list.append(self.color_surface.get_at((int(((x-256)/0.3673)+960), int(((y-212)/0.3673)+540))))
                #                 except Exception:
                #                     color_list.append((255,0,0,255))
                # self.debug_time["mapping"] += time.time()-map_time


                self.nodes = np.array(xyz)
                self.nodes = np.hstack((self.nodes, np.ones((len(self.nodes), 1))))

                transformation_matrices = []
                transform_time = time.time()

                # rotateXMatrix
                c = np.cos(self.status["rotate"][0])
                s = np.sin(self.status["rotate"][0])
                transformation_matrices.append(np.array([[1, 0, 0, 0],
                                                         [0, c,-s, 0],
                                                         [0, s, c, 0],
                                                         [0, 0, 0, 1]]))

                # rotateYMatrix
                c = np.cos(self.status["rotate"][1])
                s = np.sin(self.status["rotate"][1])
                transformation_matrices.append(np.array([[ c, 0, s, 0],
                                                         [ 0, 1, 0, 0],
                                                         [-s, 0, c, 0],
                                                         [ 0, 0, 0, 1]]))

                # rotateZMatrix
                c = np.cos(self.status["rotate"][2])
                s = np.sin(self.status["rotate"][2])
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

                # translation
                transformation_matrices.append(np.array([[1,                            0,                          0,0],
                                                         [0,                            1,                          0,0],
                                                         [0,                            0,                          1,0],
                                                         [self.status["offset"][0],     self.status["offset"][1],   0,1]]))

                for transform in transformation_matrices:
                    self.nodes = np.dot(self.nodes, transform)


                self.nodes[:, 1] = -self.nodes[:, 1]
                # self.nodes = self.nodes.astype(int)
                # print("len nodes:", len(self.nodes))
                # self.nodes = np.unique(self.nodes, axis = 0)
                # print("len nodes:", len(self.nodes))

                self.debug_time["transforming"] += time.time() - transform_time

                display_time = time.time()

                self.node_surface.fill(black)
                color_to_draw = (white)

                for index, node in enumerate(self.nodes):
                    if self.show_color_pixel: color_to_draw = color_list[index]
                    pygame.draw.circle(self.node_surface, color_to_draw, (int(node[0]), int(node[1])), 2, 0)

                # self.nodes[:, 0] += abs(max(min(self.nodes[:, 0]), 0))
                # self.nodes[:, 1] += abs(max(min(self.nodes[:, 1]), 0))
                # dim = int(max(max(self.nodes[:, 0]), max(self.nodes[:, 1])))+1
                # np_nodes = np.zeros((dim,dim,3))
                # for node in self.nodes:
                #     np_nodes[int(node[0]), int(node[1])] = np.array(white)
                #
                # self.node_surface = pygame.surfarray.make_surface(np_nodes)


                # ax.scatter(self.nodes[:, 0], self.nodes[:, 1])
                # plt.pause(0.01)




                self.debug_time["displaying"] += time.time()-display_time

                
            self.color_surface_to_draw = pygame.transform.scale(self.color_surface, (192*factor,108*factor));
            self.depth_surface_to_draw = pygame.transform.scale(self.depth_surface, (192*factor,108*factor));
            self.node_surface_to_draw = pygame.transform.scale(self.node_surface, (192*factor*2,192*factor*2));
            self.screen.blit(self.color_surface_to_draw, (0,0))
            self.screen.blit(self.depth_surface_to_draw, (192*factor,0))
            self.screen.blit(self.node_surface_to_draw, (0,108*factor))

            pygame.display.update()
            pygame.display.flip()

            if frame > 200:
                print(dict([(key, self.debug_time[key] / (time.time()-begin_time)) for key in self.debug_time.keys()]))
                break



        

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


    def retrieve_data(self):
        if (self.sensor and self._kinect.has_new_color_frame()) or (not self.sensor and self.frame_name in self.color_files):
            if self.sensor:
                self.color_frame = self._kinect.get_last_color_frame()
            else:
                self.color_frame = np.load(self.folder_path+"color/"+self.frame_name)

            self.draw_color_frame(self.color_frame, self.color_surface)
            pygame.draw.rect(self.color_surface, black, ((0,0), self.color_surface.get_size()), 80)

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
        # self.head_locations = []

        if self.sensor: 
            head_location_to_add = self.get_head_location() 
            if head_location_to_add is not None: self.head_locations = head_location_to_add
        elif self.frame_name in self.heads_files: 
            self.head_locations = [list(element) for element in np.load(self.folder_path+"heads/"+self.frame_name)]

        if self.record: np.save(self.folder_path+"/heads/frame_"+str(self.frame), np.array(self.head_locations))


        for index, head in enumerate(self.head_locations):
            # print(index, head)
            
            if len(head) == 3:
                # print(head, self.head_locations)
                if self.frame >= 1:
                    d = self.get_distances(head, last_head_locations, return_zero = True)
                
                if True:#len(d) > 0:
                    
                    if len(d) > 0 and min(d) < 150:
                        last_coordinate = last_head_locations[d.index(min(d))]
                        id_to_add = last_coordinate[3]
                        # print("self:", head, "/head_locations:", self.head_locations, "/distances", d, "/index:", d.index(min(d)), "/id:", id_to_add)
                    else:
                        if len(d)>0:
                            print(min(d))
                        self.head_id_count += 1
                        id_to_add = self.head_id_count
                    head.append(id_to_add)
        print("head_location frame",self.frame, self.head_locations)
            

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
        self.head_id_count = 0
        self.head_locations = []
        self.begin_time = time.time()
        self.fps = 90

        self.sensor = True
        self.record = False

        if not self.sensor:
            self.folder_name = "kinect_recording_1605639256"
            self.folder_path = "C:/Users/david/Documenten/peno/P_en_O_3_computer_vision/kinect_data/"+self.folder_name+"/"
            self.color_files = [f for f in os.listdir(self.folder_path+"color")]
            self.depth_files = [f for f in os.listdir(self.folder_path+"depth")]
            self.heads_files = [f for f in os.listdir(self.folder_path+"heads")]
            self.last_frame = max([int(element.split("_")[1].split(".")[0]) for element in self.heads_files])

        if self.record:
            self.folder_name = str(int(time.time()))
            self.folder_path = "C:/Users/david/Documenten/peno/P_en_O_3_computer_vision/kinect_data/kinect_recording_"+self.folder_name
            os.mkdir(self.folder_path)
            os.mkdir(self.folder_path+"/color")
            os.mkdir(self.folder_path+"/depth")
            os.mkdir(self.folder_path+"/heads")
            print("made directory", self.folder_path)

        while True:

            if not self.sensor: self.frame_name = "frame_"+str(self.frame)+".npy"

            self.retrieve_data()

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    self._screen = pygame.display.set_mode(event.dict['size'],pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                    self._screen.fill(white)

            if self.head_locations is not None and len(self.head_locations)>0: 
                self.draw_background(self.topdown_surface)

            self.draw_heads()

            self.draw_foreground()

            
            self.frame = int((time.time()-self.begin_time)*self.fps)
            if not self.sensor and self.frame > self.last_frame: break





topDownObject = TopDownViewRuntime();
# topDownObject.user_interface()

topDownObject.d3d_map()

# position = topDownObject.get_position_from_frame((1,1))
# print(position)
