import time
import os
import sys
import cv2
import pygame
from pygame.locals import *
import dlib


from project_functions import *

class user_interface_class(TopDownViewRuntime):
    def __init__(self):
        self.init()

        self.sensor = True
        self.record = False

        self.body_detection_kinect = True

        self.enable_mask_detection = True
        self.mask_detection_by_color = True
        self.mask_detection_by_machine = False

        self.display_debug = True

        self.draw_id = False
        self.draw_fps = False

        self.draw_hands = True

    def draw_heads(self):
        combos = []
        head_coordinates = self.convert_to_coordinates(self.head_locations)
        text_surfaces_to_draw = []
        too_close = 0
        for head in self.head_locations:
            coordinate = self.convert_to_coordinates(head)
            try:
                nearest = min(self.get_distances(coordinate, head_coordinates))
            except Exception as e:
                nearest = 3000
            if nearest > 1500:
                outer_circle_color = (100, 200, 100)
            else:
                outer_circle_color = (255, 0, 0)
                too_close += 1
            radius = int(750 * self.topdown_scale)

            if head[3] in self.body_status.keys() and "mask" in self.body_status[head[3]].keys():
                if self.body_status[head[3]]["mask"] == "mask":
                    inner_circle_color = (100, 200, 100)
                elif self.body_status[head[3]]["mask"] == 1:
                    inner_circle_color = (255, 106, 0)
                else:
                    inner_circle_color = (255, 0, 0)
            else:
                inner_circle_color = (255, 0, 0)

            for second_head in self.head_locations:
                if second_head != head:

                    if ([head, second_head] not in combos) and ([second_head, head] not in combos):
                        second_coordinate = self.convert_to_coordinates(second_head)
                        distance = self.get_distance(coordinate, second_coordinate)

                        pygame.draw.line(self.topdown_surface, (255, 0, 0), self.coordinate_to_pixel(coordinate),
                                         self.coordinate_to_pixel(second_coordinate), 5)
                        if distance > 0:
                            textsurface = self.myfont.render(str(round(distance / 1000, 3)).replace(".", ",") + " m",
                                                             False, (0, 0, 255))
                            text_coordinate = self.coordinate_to_pixel(self.get_middle(coordinate, second_coordinate))

                            text_surfaces_to_draw.append([textsurface, text_coordinate])
                        combos.append([head, second_head])

            pygame.draw.circle(self.topdown_surface, outer_circle_color, self.coordinate_to_pixel(coordinate), radius,
                               3)
            pygame.draw.circle(self.topdown_surface, inner_circle_color, self.coordinate_to_pixel(coordinate), 20)

            if self.display_debug and self.draw_id:
                textsurface = self.myfont.render(str(head[3]), False, (0, 0, 255))
                text_coordinate = self.coordinate_to_pixel(coordinate)
                text_surfaces_to_draw.append([textsurface, text_coordinate])

        for textsurface, text_coordinate in text_surfaces_to_draw:
            pygame.draw.rect(self.topdown_surface, self.gray, ((text_coordinate[0] - 5, text_coordinate[1] - 5), (
            textsurface.get_size()[0] + 10, textsurface.get_size()[1] + 10)))
            pygame.draw.rect(self.topdown_surface, self.black, ((text_coordinate[0] - 5, text_coordinate[1] - 5), (
            textsurface.get_size()[0] + 10, textsurface.get_size()[1] + 10)), 3)
            self.topdown_surface.blit(textsurface, text_coordinate)

        if not self.body_detection_kinect and self.display_debug:
            if self.head_locations is not None:
                for head_location in self.head_locations:
                    pygame.draw.circle(self.color_surface, self.red, (head_location[0], head_location[1]), 10, 0)

    def draw_background(self, surface):
        surface.fill(self.black)
        width, height = surface.get_size()
        pygame.draw.rect(surface, self.black, ((0,0), (width, height)), 3)
        grid_width = int(width/100)
        grid_height = int(height/100)
        margin = 5
        grid_rectangle_size = 100-margin
        for row in range(grid_height):
            for column in range(grid_width):
                pygame.draw.rect(surface,
                                 self.white,
                                 [(margin + grid_rectangle_size) * column + margin/2,
                                  (margin + grid_rectangle_size) * row + margin/2,
                                  grid_rectangle_size,
                                  grid_rectangle_size])
        pygame.draw.line(surface, self.black, self.coordinate_to_pixel((0,0,0)), self.coordinate_to_pixel((21000, 0, 27000)), 10)
        pygame.draw.line(surface, self.black, self.coordinate_to_pixel((0,0,0)), self.coordinate_to_pixel((-21000, 0, 27000)), 10)

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
        self.info_surface.fill(self.white)
        pygame.draw.rect(self.info_surface, self.black, ((0, 0), self.info_surface.get_size()), 5)
        self._screen.blit(self.info_surface, info_position)

        pygame.display.update()
        pygame.display.flip()

    def user_interface(self):
        pygame.init()
        self._clock = pygame.time.Clock()
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((2050,650), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        self.topdown_surface = pygame.Surface(self.topdown_surface_size, 0, 32)
        self.color_surface = pygame.Surface((1920, 1080), 0, 32)
        self.info_surface = pygame.Surface((400,800), 0,32)
        pygame.display.set_caption('Topdown view')
        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self.bigfont = pygame.font.SysFont('Comic Sans MS', 200)
        self.draw_background(self.topdown_surface)
        self._screen.fill(self.white)
        self.head_id_count = 0
        self.head_locations = []
        self.begin_time = time.time()
        self.fps = 90
        self.first_frame = False
        self.body_status = {}

        self.topdown = True

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./code_for_kinect/shape_predictor_68_face_landmarks.dat")

        prototxtPath = r"code_for_kinect\face_detector\deploy.prototxt"
        weightsPath = r"code_for_kinect\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        self.maskNet = load_model("./code_for_kinect/face_detector/mask_detector.model")

        self.person_positions = []
        self.chest_depth = (0, 0)


        if not self.sensor:
            self.folder_name = "kinect_recording_mondmasker"
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

            timer = cv2.getTickCount()

            if not self.sensor: self.frame_name = "frame_"+str(self.frame)+".npy"

            self.retrieve_data()
            if self.enable_mask_detection: self.mask_detection(machine=self.mask_detection_by_machine, color=self.mask_detection_by_color)
            self.hands_too_close(150)


            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    self._screen = pygame.display.set_mode(event.dict['size'],pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                    self._screen.fill(self.white)

            if self.head_locations is not None and len(self.head_locations) > 0:
                self.draw_background(self.topdown_surface)

            self.draw_heads()

            if (self.sensor and self._kinect.has_new_color_frame()) and self.display_debug and self.draw_fps:
                fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - timer))
                fps_text = self.bigfont.render(str(fps), False, self.black)
                text_coordinate = (50, 50)
                self.color_surface.blit(fps_text, text_coordinate)

            self.draw_foreground()

            self.frame = int((time.time()-self.begin_time)*self.fps)
            if not self.sensor and self.frame > self.last_frame: break



if __name__ == "__main__":
    interface = user_interface_class()
    interface.user_interface()