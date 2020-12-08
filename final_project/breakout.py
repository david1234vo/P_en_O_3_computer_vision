import math
import time
import random
import sys
import pygame
from pygame.locals import *
import numpy as np

from project_functions import *

class breakout_class(TopDownViewRuntime):
    def __init__(self):
        self.init()

    def draw_grid(self, begin_pos, width, height):
        for x in range(len(self.grid)):
            for y in range(len(self.grid[0])):
                if self.grid[x][y] == 1:
                    pygame.draw.rect(self.breakout_surface, self.red, ((begin_pos[0]+width*x, begin_pos[1]+height*y), (width, height)), 3)


    def get_ball_pos(self, direction):
        if direction == "left" or direction == 0:
            return self.ball_location[0] - self.ball_radius - abs(self.ball_speed[0] * 2)
        elif direction == "right" or direction == 2:
                return self.ball_location[0] + self.ball_radius + abs(self.ball_speed[0] * 2)
        elif direction == "bottom" or direction == 3:
            return self.ball_location[1] + self.ball_radius + abs(self.ball_speed[1] * 2)
        elif direction == "top" or direction == 1:
            return self.ball_location[1] - self.ball_radius - abs(self.ball_speed[1] * 2)

    def get_ball_pos(self, fraction, fractions=4):
        # fractions = 8
        angle = math.pi*2/fractions
        # fraction = 1
        return [self.ball_location[0] + math.cos(angle * fraction) * (self.ball_radius + abs(self.ball_speed[0])),
                self.ball_location[1] + math.sin(angle * fraction) * (self.ball_radius + abs(self.ball_speed[1]))]

    def get_collide_vector(self, fraction, vector, fractions=4):
        angle = (math.pi * 2 / fractions) * fraction
        c = math.cos(angle)
        s = math.sin(angle)
        rotate_matrix = np.array([[c, s, 0],
                                  [-s, c, 0],
                                  [0, 0, 1]])

        # print(vector+[0])
        rotated_vector = np.dot(vector + [0], rotate_matrix)
        reversed_vector = [-rotated_vector[0], rotated_vector[1], rotated_vector[2]]
        final_vector = np.dot(reversed_vector, np.transpose(rotate_matrix))
        return list(final_vector[0:2])

    def breakout(self):
        pygame.init()
        self._clock = pygame.time.Clock()
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((1000, 900), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)
        self.breakout_surface = pygame.Surface((1000,900), 0, 32)
        pygame.display.set_caption('Breakout')
        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)

        # self.draw_background(self.topdown_surface)
        self._screen.fill(self.white)
        self.head_id_count = 0
        self.head_locations = []
        self.begin_time = time.time()
        self.fps = 90

        self.sensor = True
        self.record = False
        self.topdown = False
        self.body_detection_kinect = True

        self.grid = [[1 for y in range(8)] for x in range(6)]

        self.ball_location = [0,0]
        self.ball_speed = [0,0]
        self.ball_radius = 20
        self.bat_coordinate = [0,0]
        self.started = False

        grid_begin_pos = (100, 80)
        grid_width, grid_height = 130, 40
        bat_width, bat_height = 100, 20

        standard_ball_speed = 2


        surface_width, surface_height = self.breakout_surface.get_size()

        while True:

            self.breakout_surface.fill(self.black)
            self.retrieve_data(draw=False)

            if len(self.head_locations) > 0:
                person_coordinate = self.convert_to_coordinates(self.head_locations[0])[0]

                self.bat_coordinate = [int(person_coordinate+surface_width/2), surface_height-100]

                pygame.draw.rect(self.breakout_surface, self.red, ((self.bat_coordinate[0]-bat_width/2, self.bat_coordinate[1]-bat_height/2), (bat_width, bat_height)))
                if self.ball_location == [0,0]:
                    self.ball_location = [self.bat_coordinate[0], self.bat_coordinate[1]-60]
                    self.started = True

                    self.ball_speed = [standard_ball_speed*random.choice([-1,1]),-standard_ball_speed]


            if self.started:

                if 0 > self.get_ball_pos(2)[0] or self.get_ball_pos(0)[0] > surface_width:
                    self.ball_speed[0] = -self.ball_speed[0]

                    if self.get_ball_pos(2)[0] < 0: #-abs(self.ball_speed[0])*2:
                        self.ball_location[0] = abs(self.ball_speed[0])+self.ball_radius

                    if self.get_ball_pos(0)[0] > surface_width: #+abs(self.ball_speed[0])*2:
                        self.ball_location[0] = surface_width-abs(self.ball_speed[0])-self.ball_radius

                if 0 > self.get_ball_pos(3)[1] or self.get_ball_pos(1)[1] > surface_height:
                    self.ball_speed[1] = -self.ball_speed[1]

                    if self.get_ball_pos(3)[1] < 0: #-abs(self.ball_speed[1])*2:
                        self.ball_location[1] = abs(self.ball_speed[1])+self.ball_radius

                    if self.get_ball_pos(1)[1] > surface_height: #+abs(self.ball_speed[1])*2:
                        self.ball_location[1] = surface_height-abs(self.ball_speed[1])-self.ball_radius

                # print(self.ball_location[1]+self.ball_radius+abs(self.ball_speed[1]*2) - self.bat_coordinate[1], self.ball_location[0]-self.bat_coordinate[0])
                collide_resolution = 8
                for fraction in range(8):
                    edge_pos = self.get_ball_pos(fraction, fractions = collide_resolution)

                    if (self.bat_coordinate[0] - bat_width / 2 < edge_pos[0] < self.bat_coordinate[0] + bat_width / 2) and \
                            (self.bat_coordinate[1] - bat_height / 2 < edge_pos[1] < self.bat_coordinate[1] + bat_height / 2):
                        self.ball_speed = self.get_collide_vector(fraction, self.ball_speed, fractions=collide_resolution)

                    for x in range(len(self.grid)):
                        for y in range(len(self.grid[0])):
                            if self.grid[x][y] == 1:
                                top, left = grid_begin_pos[1] + grid_height * y, grid_begin_pos[0] + grid_width * x
                                bottom, right = top + grid_height, left + grid_width
                                if left <= edge_pos[0] <= right and top <= edge_pos[1] <= bottom:
                                    self.ball_speed = self.get_collide_vector(fraction, self.ball_speed,
                                                                              fractions=collide_resolution)
                                    self.grid[x][y] = 0



            self.ball_location = [int(self.ball_location[0] + self.ball_speed[0]),
                                  int(self.ball_location[1] + self.ball_speed[1])]

            if self.ball_location != [0,0]:
                pygame.draw.circle(self.breakout_surface, self.red, self.ball_location, self.ball_radius)
            self.draw_grid(grid_begin_pos, grid_width, grid_height)


            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    self._screen = pygame.display.set_mode(event.dict['size'],
                                                           pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE, 32)
                    self._screen.fill(self.white)


            self._screen.blit(self.breakout_surface, (0,0))
            pygame.display.update()
            pygame.display.flip()

            self.frame = int((time.time() - self.begin_time) * self.fps)
            # if not self.sensor and self.frame > self.last_frame: break




if __name__ == "__main__":
    breakout = breakout_class()
    breakout.breakout()