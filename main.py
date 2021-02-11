import Room
import pygame, sys
from pygame.locals import *
import time

rows_of_agents = 3
cols_of_agents = 3

room = Room.Room(3, 3)

BLACK = (0, 0, 0)
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
height_per_block = WINDOW_HEIGHT // len(room.grid)
width_per_block = WINDOW_WIDTH // len(room.grid[0])

global SCREEN, CLOCK

def draw(grid):
    for x, i in enumerate(range(len(room.grid))):
        for y, j in enumerate(range(len(room.grid[0]))):
            rect = pygame.Rect(x*height_per_block, y*height_per_block,
                               height_per_block, height_per_block)
            global SCREEN
            pygame.draw.rect(SCREEN, grid[i][j].get_color(), rect)



def viz():
    pygame.init()
    global SCREEN, CLOCK
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BLACK)

    draw(room.grid)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    time.sleep(2)
    pygame.quit()


if __name__ == '__main__':
    viz()