import pygame
#from kinect_packages import kinect2_test_game_2 as X


pygame.init()

BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)

size = ( 760, 550)
WIDTH= 140
HEIGHT = 140
MARGIN = 10
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Grid")

done = False

clock = pygame.time.Clock()

for row in range(1,4):
    for column in range(5):
        color = WHITE
        pygame.draw.rect(screen,
                         color,
                         [(MARGIN + WIDTH) * column + MARGIN,
                          (MARGIN + HEIGHT) * row + MARGIN - 100,
                          WIDTH,
                          HEIGHT])

font = pygame.font.Font('freesansbold.ttf', 32)

text1 = font.render('Social distancing', True, (102,204,0))
text_rect1 = text1.get_rect()
text_rect1.center = ( 380, 25)

text2 = font.render('Amount of people to close to each other: #!', True, (102,204,0))
text_rect2 = text2.get_rect()
text_rect2.center = ( 360, 525)

pygame.display.update()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    screen.blit(text1, text_rect1)
    screen.blit(text2, text_rect2)
    pygame.display.update()

#    for head in X.head_locations:
 #       coordinate = X.convert_to_coordinates(head, X.window_size)
  #      print(coordinate)
   #     X.heads_depth.append(coordinate[2])
    #    X.heads_x.append(coordinate[0])
     #   X.heads_y.append(coordinate[1])
      #  pygame.draw.circle(screen, ( 0, 0, 255), (int(coordinate[0]/10 + 400), int(coordinate[2]/10 + 200)), 10)

