import pygame

from project_functions import *


class c_and_d_interface_class(TopDownViewRuntime):
    def __init__(self):
        self.init()

    def color_and_depth_interface(self):
        pygame.init()
        factor = 4
        self.screen = pygame.display.set_mode((192*factor*2,108*factor), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
        self.color_surface = pygame.Surface((1920, 1080), 0, 32)
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

if __name__ == "__main__":
    cad_interface = c_and_d_interface_class()
    cad_interface.color_and_depth_interface()