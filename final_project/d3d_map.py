import ctypes
import math
import time
import pygame
import numpy as np


from project_functions import *

class d3d_map_class(TopDownViewRuntime):
    def __init__(self):
        self.init()

    def d3d_map(self):
        # http://archive.petercollingridge.co.uk/book/export/html/460
        display = True
        timing = False

        zoom_factor = 2
        if display:
            self.show_color_pixel = False
            pygame.init()
            factor = 2

            self.screen = pygame.display.set_mode((192 * factor * 2, 108 * factor * 2 + 192 * factor),
                                                  pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)
            self.color_surface = pygame.Surface((1920, 1080), 0, 32)
            self.depth_surface = pygame.Surface((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height),
                                                0, 24)
            self.node_surface = pygame.Surface((1000 * zoom_factor, 1000 * zoom_factor), 0, 32)
        frame = 0
        got_frame = False
        begin_time = time.time()


        self.status = {"offset": [630 * zoom_factor, -440 * zoom_factor], "scaling_factor": 1, "rotate": [0, 0, 0]}


        step = 2
        width, height = 512, 424
        n_width, n_height = int(width / step), int(height / step)
        c_width, c_height = n_width * step, n_height * step
        x_coordinates = np.repeat(np.arange(0, c_width, step), n_height).reshape(n_height, n_width, order='F').ravel()
        y_coordinates = np.repeat(np.arange(0, c_width, step), n_height)

        total_alg_time = 0

        while True:
            if got_frame:
                frame += 1
                if frame == 1:
                    begin_time = time.time()
            if self._kinect.has_new_color_frame() and display:
                color_frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(color_frame, self.color_surface)

            if self._kinect.has_new_depth_frame():
                depth_frame_og = self._kinect.get_last_depth_frame()
                depth_frame = depth_frame_og.reshape(424, 512)
                if display: self.draw_infrared_frame(depth_frame, self.depth_surface)
                got_frame = True
                passed_time = time.time() - begin_time
                if passed_time != 0: print("frame", frame, round(passed_time, 2), round(frame / passed_time, 2))

            if display:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:  self.status["offset"][0] += 50 * zoom_factor
                        if event.key == pygame.K_RIGHT: self.status["offset"][0] -= 50 * zoom_factor
                        if event.key == pygame.K_DOWN:  self.status["offset"][1] += 50 * zoom_factor
                        if event.key == pygame.K_UP:    self.status["offset"][1] -= 50 * zoom_factor
                        if event.key == pygame.K_EQUALS: self.status["scaling_factor"] += 0.5
                        if event.key == pygame.K_MINUS:  self.status["scaling_factor"] -= 0.5
                        if event.key == pygame.K_q:      self.status["rotate"][0] += math.pi / 8  # a
                        if event.key == pygame.K_a:      self.status["rotate"][0] -= math.pi / 8  # q
                        if event.key == pygame.K_w:      self.status["rotate"][1] += math.pi / 8  # z
                        if event.key == pygame.K_s:      self.status["rotate"][1] -= math.pi / 8
                        if event.key == pygame.K_e:      self.status["rotate"][2] += math.pi / 8
                        if event.key == pygame.K_d:      self.status["rotate"][2] -= math.pi / 8
                        print(event.key, self.status)

            if frame >= 1:
                alg_begin_time = time.time()

                color_list = []

                z = depth_frame[0:c_height:step, 0:c_width:step].flatten()
                x_transformed = (x_coordinates - width / 2) * z / 1000
                y_transformed = -(y_coordinates - height / 2) * z / 1000
                xyz = np.c_[x_transformed, y_transformed, z / 4]

                self.nodes = np.array(xyz)
                self.nodes = np.hstack((self.nodes, np.ones((len(self.nodes), 1))))

                transformation_matrices = []

                # rotateXMatrix
                c = np.cos(self.status["rotate"][0])
                s = np.sin(self.status["rotate"][0])
                transformation_matrices.append(np.array([[1, 0, 0, 0],
                                                         [0, c, -s, 0],
                                                         [0, s, c, 0],
                                                         [0, 0, 0, 1]]))

                # rotateYMatrix
                c = np.cos(self.status["rotate"][1])
                s = np.sin(self.status["rotate"][1])
                transformation_matrices.append(np.array([[c, 0, s, 0],
                                                         [0, 1, 0, 0],
                                                         [-s, 0, c, 0],
                                                         [0, 0, 0, 1]]))

                # rotateZMatrix
                c = np.cos(self.status["rotate"][2])
                s = np.sin(self.status["rotate"][2])
                transformation_matrices.append(np.array([[c, -s, 0, 0],
                                                         [s, c, 0, 0],
                                                         [0, 0, 1, 0],
                                                         [0, 0, 0, 1]]))

                # scaling
                s = (self.status["scaling_factor"],) * 3
                transformation_matrices.append(np.array([[s[0], 0, 0, 0],
                                                         [0, s[1], 0, 0],
                                                         [0, 0, s[2], 0],
                                                         [0, 0, 0, 1]]))

                # translation
                transformation_matrices.append(np.array([[1, 0, 0, 0],
                                                         [0, 1, 0, 0],
                                                         [0, 0, 1, 0],
                                                         [self.status["offset"][0], self.status["offset"][1], 0, 1]]))

                for transform in transformation_matrices:
                    self.nodes = np.dot(self.nodes, transform)

                total_alg_time += time.time()-alg_begin_time

                if display:
                    self.node_surface.fill(self.black)
                    color_to_draw = (self.white)

                    for index, node in enumerate(self.nodes):
                        if self.show_color_pixel: color_to_draw = color_list[index]
                        pygame.draw.circle(self.node_surface, color_to_draw, (int(node[0]), -int(node[1])), 2, 0)

            if display:
                self.color_surface_to_draw = pygame.transform.scale(self.color_surface, (192 * factor, 108 * factor));
                self.depth_surface_to_draw = pygame.transform.scale(self.depth_surface, (192 * factor, 108 * factor));
                self.node_surface_to_draw = pygame.transform.scale(self.node_surface, (192 * factor * 2, 192 * factor * 2));
                self.screen.blit(self.color_surface_to_draw, (0, 0))
                self.screen.blit(self.depth_surface_to_draw, (192 * factor, 0))
                self.screen.blit(self.node_surface_to_draw, (0, 108 * factor))

                pygame.display.update()
                pygame.display.flip()

            if frame > 150 and timing:
                total_time = time.time()-begin_time
                # print("total", int(total_time), int(total_alg_time), round(total_alg_time/total_time, 3))
                print("fps:", round(frame/total_time, 3))
                break


if __name__ == "__main__":
    topdown = d3d_map_class()
    topdown.d3d_map()
