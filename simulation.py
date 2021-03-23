import Room
import csv
import copy

SCREEN, CLOCK = None, None
FAN_CYCLES = 4

def run(room):

    steps_taken = 0
    p25 = round(room.iterations/4, 0)
    p50 = p25*2
    p75 = p25*3

    while room.steps_taken < room.iterations:
        room._step()
        steps_taken += 1
        if room.steps_taken == p25:
            print("25% done")
        elif room.steps_taken == p50:
            print("50% done")
        elif room.steps_taken == p75:
            print("75% done")

    with open(room.filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(room.fields)

        # writing the data rows
        csvwriter.writerows(room.rows)

    print("{} IS DONE".format(room.filename))


def simulate(room_array):
    for room in room_array:
        run(room)


rows_people = 5
cols_people = 5
HAVE_TEACHER = True
MOVING_AGENT = False
iterations = 100
room = Room.Room(rows_people, cols_people, iterations, 42, HAVE_TEACHER, MOVING_AGENT)

simulations = []
new_diff = 0.8

for i in range(0,8):

    copy_room = copy.deepcopy(room)
    copy_room.change_diff(new_diff)
    filename = "room_d_" + str(round(new_diff,1)) + "_t_" + str(copy_room.time_length) + ".csv"
    copy_room.filename = filename
    simulations.append(copy_room)
    # print(filename)

    for j in range(1,6):
        new_copy_room = copy.deepcopy(copy_room)
        new_copy_room.time_length = copy_room.time_length - (j * 0.5)
        filename = "room_d_" + str(round(new_diff,1)) + "_t_" + str(new_copy_room.time_length) + ".csv"
        new_copy_room.filename = filename
        simulations.append(new_copy_room)
        # print(filename)
    new_diff -= 0.1

if __name__ == '__main__':
    simulate(simulations)
