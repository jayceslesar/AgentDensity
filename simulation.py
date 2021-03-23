import os
import glob
from PIL import Image
import Room
import pygame, sys
from pygame.locals import *
import time
from pygame.rect import *
import csv
import copy

SCREEN, CLOCK = None, None
FAN_CYCLES = 4

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
                file_name = os.path.join('images', color_string + '_agent.png')
                agent_img = pygame.image.load(file_name)
                agent_img = pygame.transform.scale(agent_img, (height_per_block-2, height_per_block-2))
                SCREEN.blit(agent_img, rect)
            if room.grid[i][j].advec_vec is not None:
                factor = FAN_CYCLES/4
                if step%FAN_CYCLES < factor:
                    fan_img = pygame.image.load(os.path.join('images', 'fan1.png'))
                elif step%FAN_CYCLES < factor * 2:
                    fan_img = pygame.image.load(os.path.join('images', 'fan2.png'))
                elif step%FAN_CYCLES < factor * 3:
                    fan_img = pygame.image.load(os.path.join('images', 'fan3.png'))
                else:
                    fan_img = pygame.image.load(os.path.join('images', 'fan4.png'))
                fan_img = pygame.transform.scale(fan_img, (height_per_block, height_per_block))
                SCREEN.blit(fan_img, rect)


def viz(room):

    # pygame.init()
    # global SCREEN, CLOCK
    # SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    # CLOCK = pygame.time.Clock()
    # SCREEN.fill(BLACK)

    steps_taken = 0
    while room.steps_taken < room.iterations:
        room._step()
        # draw(room, steps_taken)
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         sys.exit()

        # pygame.display.update()

        steps_taken += 1
        # pygame.quit()

    with open(room.filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(room.fields)

        # writing the data rows
        csvwriter.writerows(room.rows)
    # if choice == 'y':
    #     img, *imgs = [Image.open(f) for f in stills]
    #     img.save(fp=os.path.join(path, 'sim.gif'), format='GIF', append_images=imgs, save_all=True, duration=20, loop=0)
    #     for im in stills:
    #         os.remove(im)


def screenshot(screen, path, step):
    title = "step" + str(step)
    file_save_as = os.path.join(path, title + ".png")
    pygame.image.save(screen, file_save_as)
    print(f"step {step} has been screenshotted")


BLACK = (0, 0, 0)
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
rows_people = 5
cols_people = 5
HAVE_TEACHER = True
MOVING_AGENT = False
room = Room.Room(rows_people, cols_people, 100, 42, HAVE_TEACHER, MOVING_AGENT)

room1 = copy.deepcopy(room)
room1.change_diff(0.8)
room1.filename = "room1.csv"

room12 = copy.deepcopy(room1)
room12.time_length = 1.5
room12.filename = "room12.csv"

room13 = copy.deepcopy(room1)
room13.time_length = .75
room13.filename = "room13.csv"

# room2 = copy.deepcopy(room)
# room2.change_diff(0.6)
# room2.filename = "room2.csv"

# room22 = copy.deepcopy(room2)
# room22.falloff_rate_mean = 0.0005
# room22.filename = "room22.csv"

# room3 = copy.deepcopy(room)
# room3.change_diff(0.4)
# room3.filename = "room3.csv"

# room32 = copy.deepcopy(room3)
# room32.falloff_rate_mean = 0.0005
# room32.filename = "room32.csv"

# room4 = copy.deepcopy(room)
# room4.change_diff(1.3)
# room4.filename = "room4.csv"

# room42 = copy.deepcopy(room4)
# room42.falloff_rate_mean = 0.0005
# room42.filename = "room42.csv"

# room5 = copy.deepcopy(room)
# room5.change_diff(1)

# room52 = copy.deepcopy(room5)
# room52.falloff_rate_mean = 0.0005
# room52.filename = "room52.csv"



height_per_block = WINDOW_HEIGHT // room.num_rows
width_per_block = WINDOW_WIDTH // room.num_cols

if __name__ == '__main__':
    viz(room1)
    viz(room12)
    viz(room13)
