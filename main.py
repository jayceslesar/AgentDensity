import os

import Room
import pygame, sys
from pygame.locals import *
import time
from pygame.rect import *


SCREEN, CLOCK = None, None

def draw(grid):
    for x, i in enumerate(range(room.num_rows)):
        for y, j in enumerate(range(room.num_cols)):
            rect = pygame.Rect(x*height_per_block, y*height_per_block,
                               height_per_block, height_per_block)

            global SCREEN
            pygame.draw.rect(SCREEN, grid[i][j].get_color(), rect)
            # print("The color of cell " + str(i) + str(j) + " is " + str(grid[i][j].get_color()))

            if grid[i][j].agent is not None:
                pygame.draw.rect(SCREEN, grid[i][j].agent.get_color(), rect, 4)


def viz(room):
    path = input("What folder do you want to save your screenshots into? Please specify the path \n")
    skip = int(input("How many steps between screenshots? \n"))
    if not os.path.exists(path):
        os.makedirs(path)

    pygame.init()
    global SCREEN, CLOCK
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BLACK)

    while room.steps_taken < room.iterations:
        room._step()
        draw(room.grid)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()
        if room.steps_taken%skip == 0:
            screenshot(SCREEN, path, room.steps_taken)
        time.sleep(.02)
        # pygame.quit()


def screenshot(screen, path, step):
    title = "step" + str(step)
    file_save_as = path + "/" + title + ".png"
    pygame.image.save(screen, file_save_as)
    print(f"step {step} has been screenshotted")


BLACK = (0, 0, 0)
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
rows_people = 7
cols_people = 7
HAVE_TEACHER = True
room = Room.Room(rows_people, cols_people, 1000, 42, HAVE_TEACHER)

height_per_block = WINDOW_HEIGHT // room.num_rows
width_per_block = WINDOW_WIDTH // room.num_cols

if __name__ == '__main__':
    viz(room)
