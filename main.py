import os

import Room
import pygame, sys
from pygame.locals import *
import time
from pygame.rect import *


SCREEN, CLOCK = None, None
FAN_CYCLES = 12

def draw(room, step):
    for x, i in enumerate(range(room.num_rows)):
        for y, j in enumerate(range(room.num_cols)):
            rect = pygame.Rect(y*height_per_block, x*height_per_block,
                               height_per_block, height_per_block)

            global SCREEN
            pygame.draw.rect(SCREEN, room.grid[i][j].get_color(), rect)
            # print("The color of cell " + str(i) + str(j) + " is " + str(grid[i][j].get_color()))

            if room.grid[i][j].agent is not None:
                pygame.draw.rect(SCREEN, room.grid[i][j].agent.get_color(), rect, 4)
                color_string = room.grid[i][j].agent.get_color_string()
                file_name = color_string + '_agent.png'
                agent_img = pygame.image.load(file_name)
                agent_img = pygame.transform.scale(agent_img, (height_per_block-2, height_per_block-2))
                SCREEN.blit(agent_img, rect)
            if room.grid[i][j].advec_vec is not None:
                factor = FAN_CYCLES/4
                if step%FAN_CYCLES <= factor:
                    fan_img = pygame.image.load('fan1.png')
                elif step%FAN_CYCLES <= factor * 2:
                    fan_img = pygame.image.load('fan2.png')
                elif step%FAN_CYCLES <= factor * 3:
                    fan_img = pygame.image.load('fan3.png')
                else:
                    fan_img = pygame.image.load('fan4.png')
                fan_img = pygame.transform.scale(fan_img, (height_per_block, height_per_block))
                SCREEN.blit(fan_img, rect)


def viz(room):
    choice = input('do you want screenshots?\n')
    if choice == 'y':
        path = input("What folder do you want to save your screenshots into? Please specify the path \n")
        skip = int(input("How many steps between screenshots? \n"))
        if not os.path.exists(path):
            os.makedirs(path)


    pygame.init()
    global SCREEN, CLOCK
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BLACK)

    steps_taken = 0
    while room.steps_taken < room.iterations:
        room._step()
        draw(room, steps_taken)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()
        if choice == 'y' and room.steps_taken%skip == 0:
            screenshot(SCREEN, path, room.steps_taken)

        steps_taken += 1
        # time.sleep(0.05)
        # pygame.quit()


def screenshot(screen, path, step):
    title = "step" + str(step)
    file_save_as = path + "/" + title + ".png"
    pygame.image.save(screen, file_save_as)
    print(f"step {step} has been screenshotted")


BLACK = (0, 0, 0)
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
rows_people = 5
cols_people = 5
HAVE_TEACHER = False
room = Room.Room(rows_people, cols_people, 100000, 42, HAVE_TEACHER)

height_per_block = WINDOW_HEIGHT // room.num_rows
width_per_block = WINDOW_WIDTH // room.num_cols

if __name__ == '__main__':
    viz(room)
