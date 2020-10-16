import pygame
import time
import math
import pyjs
import winsound


def people_to_close(locations):
    total = 0
    for t in range(len(locations) - 1):
        for k in range(t + 1, len(locations)):
            if to_close(locations[t], locations[k]):
                total += 1
    return total



def to_close(human1, human2):
    x1 = human1[0]
    x2 = human2[0]
    z1 = human1[2]
    z2 = human2[2]
    check = 150 <= math.sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)
    if not check:
        return True
    else:
        return False

def all_to_close(locations):
    x = 0
    close = []
    not_close = list(locations)
    while x < len(not_close) - 1:
        face = not_close[x]
        for other_face in not_close[x+1:]:
            if to_close(face, other_face):
                close.append(face)
                close.append(other_face)
                not_close.remove(face)
                not_close.remove(other_face)
        if x < len(not_close) and face == not_close[x]:
            x += 1
    return close, not_close

frequency = 2500
duration = 1000

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

size = (760, 700)
WIDTH = 140
HEIGHT = 140
MARGIN = 10
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Grid")

for row in range(1, 5):
    for column in range(5):
        color = WHITE
        pygame.draw.rect(screen,
                         color,
                         [(MARGIN + WIDTH) * column + MARGIN,
                          (MARGIN + HEIGHT) * row + MARGIN - 100,
                          WIDTH,
                          HEIGHT])

font1 = pygame.font.SysFont('Calibri', 30, True)
font2 = pygame.font.SysFont('Calibri', 35, True)
font3 = pygame.font.SysFont('Calibri', 30)

text1 = font1.render('SOCIAL DISTANCING', True, WHITE)
text_rect1 = text1.get_rect()
text_rect1.center = (380, 25)

face_locations = [[100, 50, 100], [300, 200, 300],[600, 0, 550]]

if people_to_close(face_locations) == 0:
    text2 = font3.render('Amount of people to close to each other: 0 :) ', True, WHITE)
else:
    text2 = font2.render(f'Amount of people to close to each other: {people_to_close(face_locations)} !', True, (255, 0, 0))
text_rect2 = text2.get_rect()
text_rect2.center = (360, 675)



while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    screen.blit(text1, text_rect1)
    screen.blit(text2, text_rect2)

    for i in range(10):
        screen.fill((0, 0, 0))
        screen.blit(text1, text_rect1)
        for row in range(1, 5):
            for column in range(5):
                color = WHITE
                pygame.draw.rect(screen,
                                 color,
                                 [(MARGIN + WIDTH) * column + MARGIN,
                                  (MARGIN + HEIGHT) * row + MARGIN - 100,
                                  WIDTH,
                                  HEIGHT])
        face1 = face_locations[0]
        face2 = face_locations[1]
        face3 = face_locations[2]
        face1[0] = face1[0] + i * 10
        face2[2] = face2[2] - i * 10
        face3[2] = face3[2] - i * 10
        people_close, people_safe = all_to_close(face_locations)
        for face in people_safe:
            pygame.draw.circle(screen, (0, 0, 255), (face[0], face[2]), 74, 6)
            pygame.draw.circle(screen, (0, 0, 255), (face[0], face[2]), 12)
        for face in people_close:
            pygame.draw.circle(screen, (255, 0, 0), (face[0], face[2]), 74, 6)
            pygame.draw.circle(screen, (255, 0, 0), (face[0], face[2]), 12)
        if people_to_close(face_locations) == 0:
            text2 = font3.render('Amount of people to close to each other: 0 :) ', True, WHITE)
        else:
            #winsound.Beep(frequency, duration)
            text2 = font2.render(f'Amount of people to close to each other: {people_to_close(face_locations)} !', True,
                                (255, 0, 0))
        text_rect2 = text2.get_rect()
        text_rect2.center = (360, 675)
        screen.blit(text2, text_rect2)
        pygame.display.update()
        print(people_to_close(face_locations))
        time.sleep(0.4)

