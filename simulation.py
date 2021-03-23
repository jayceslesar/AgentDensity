import Room
import csv
import copy
import itertools
import pandas as pd
import os

SCREEN, CLOCK = None, None
FAN_CYCLES = 4

def run(room, diff, group):
    run_dict = {}
    for field in room.fields:
        run_dict[field] = []
    run_dict["diffusivity"] = []
    run_dict["time_step"] = []
    run_dict["group"] = []
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

    for i in range(len(room.rows)):
        for j in range(len(room.fields)):
            run_dict[room.fields[j]].append(room.rows[i][j])
        run_dict["diffusivity"].append(diff)
        run_dict["time_step"].append(room.time_length)
        run_dict["group"].append(group)


    print("DONE WITH GROUP", group)
    return run_dict


def simulate(room_array):
    grp = 1
    sim_dicts = []
    final_dict = {}
    for room, diff in room_array:
        sim_dicts.append(run(room, diff, grp))
        grp+=1

    for key in sim_dicts[0].keys():
        final_dict[key] = list(itertools.chain.from_iterable([sim_dict[key] for sim_dict in sim_dicts]))

    final_df = pd.DataFrame(final_dict)
    final_df.to_csv(os.path.join("data","analysis_data.csv"))

if __name__ == '__main__':
    rows_people = 5
    cols_people = 5
    HAVE_TEACHER = True
    MOVING_AGENT = False
    iterations = 2500
    room = Room.Room(rows_people, cols_people, iterations, 42, HAVE_TEACHER, MOVING_AGENT)

    simulations = []
    new_diff = 0.8

    for i in range(0,6):

        copy_room = copy.deepcopy(room)
        copy_room.change_diff(new_diff)
        filename = "room_d_" + str(new_diff) + "_t_" + str(copy_room.time_length) + ".csv"
        copy_room.filename = filename
        simulations.append((copy_room, new_diff))

        for j in range(1,6):
            new_copy_room = copy.deepcopy(copy_room)
            new_copy_room.time_length = copy_room.time_length - (j * 0.5)
            filename = "room_d_" + str(new_diff) + "_t_" + str(new_copy_room.time_length) + ".csv"
            new_copy_room.filename = filename
            simulations.append((new_copy_room, new_diff))

        new_diff = new_diff/2

    simulate(simulations)
