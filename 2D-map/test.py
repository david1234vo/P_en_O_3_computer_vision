import pygame

pygame.init()

BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)

size = ( 610, 610)
WIDTH= 50
HEIGHT = 50
MARGIN = 5
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Grid")

done = False

clock = pygame.time.Clock()

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    for row in range(11):
        for column in range(11):
            color = WHITE
            pygame.draw.rect(screen,
                             color,
                             [(MARGIN + WIDTH) * column + MARGIN,
                              (MARGIN + HEIGHT) * row + MARGIN,
                              WIDTH,
                              HEIGHT])

    pygame.display.flip()
    clock.tick(60)