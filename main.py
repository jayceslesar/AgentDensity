import Room
import pygame, sys
from pygame.locals import *
import time
from pygame.rect import *



global SCREEN, CLOCK

def draw(grid):
    for x, i in enumerate(range(len(room.grid))):
        for y, j in enumerate(range(len(room.grid[0]))):
            rect = pygame.Rect(x*height_per_block, y*height_per_block,
                               height_per_block, height_per_block)

            global SCREEN
            pygame.draw.rect(SCREEN, grid[i][j].get_color(), rect)

            if grid[i][j].agent is not None:
                pygame.draw.rect(SCREEN, grid[i][j].agent.get_color(), rect, 4)





def viz(room):
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
        time.sleep(1)
        # pygame.quit()


BLACK = (0, 0, 0)
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800

room = Room.Room(5, 5, 50, 42)

height_per_block = WINDOW_HEIGHT // len(room.grid)
width_per_block = WINDOW_WIDTH // len(room.grid[0])

if __name__ == '__main__':
    viz(room)
