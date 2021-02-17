"""
Authors:
---
    Jayce Slesar
    Brandon Lee
    Carter Ward

Date:
---
    12/29/2020
"""

import Agent
import numpy as np
import Cell
import copy


def ficks_law(diffusivity, concentration1, concentration2, area, length):
    numerator = float((concentration1 - concentration2) * area * diffusivity)
    return (float(numerator))/(float(length))


class Room:
    def __init__(self, num_rows_people: int, num_cols_people: int, num_steps, seed: int):
        np.random.seed(seed)
        """Initialize the instance of this Room

        Args:
            num_rows_people (int): number of actual rows
            num_cols_people (int): number of actual cols
            num_steps (int): number of steps in simulation
            seed (int): the seed to use
        """
        self.initial_infected = np.random.randint(0,num_rows_people*num_cols_people)
        self.num_rows = num_rows_people*2 + 1
        self.num_cols_people = num_cols_people
        self.iterations = num_steps
        self.seed = seed
        self.steps_taken = 0
        self.time_length = 1
        self.grid = []

        n = 0

        # border row for top and bottom rows
        for i in range(self.num_rows_people):
            row = []
            # add an empty space
            row.append(Cell.Cell(i, 0))
            for j in range(self.num_cols_people):
                a = Agent.Agent(n, i, j, self.seed)
                if n == self.initial_infected:
                    a.infected = True
                    self.initial_agent = a
                row.append(Cell.Cell(i,j, a))
                # add an empty space
                row.append(Cell.Cell(i,j))
                n += 1
            self.grid.append(row)
            self.grid.append(blank_row)
        self.width = self.grid[0][0].width

    def __str__(self):
        out = ""
        for row in self.grid:
            for cell in row:
                out += str(cell) + " "
            out += "\n"
        return out

    def _step(self):
        # to be changed or randomized
        INFECTED_CUTOFF = 0.6

        # iterate through rows and columns of cells
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if (i == 0 and j == 0) or (i == 3 and j == 2):
                    print("Cell " + str(i) + str(j) + " has a concentration of " + str(self.grid[i][j].concentration))

                # check if agent is not in cell
                if self.grid[i][j].agent is None:
                    continue

                # update total exposure, += concentration
                self.grid[i][j].agent.total_exposure += self.grid[i][j].concentration

                # update untouched, exposed
                if self.grid[i][j].concentration > 0:
                    self.grid[i][j].agent.untouched = False
                    self.grid[i][j].agent.exposed = True

                if self.grid[i][j].agent.exposed:
                    # update steps exposed
                    self.grid[i][j].agent.steps_exposed += 1

                # update infected
                # if self.grid[i][j].agent.total_exposure > INFECTED_CUTOFF:
                #     self.grid[i][j].agent.infected = True

                if self.grid[i][j].agent.infected:
                    # update steps infected
                    self.grid[i][j].agent.steps_infected += 1
                    # TODO: @Brandon, this is what changes the cell concentration, please update with formula
                    self.grid[i][j].concentration += (self.grid[i][j].production_rate) * self.time_length
        # Here is where I will call spread
        # self.simple_spread()
        print(self.grid.__str__())
        self.steps_taken += 1
        print(str(self.steps_taken) + " steps taken")

    def take_second(self, element):
        return element[2].concentration

    def get_concentration_array(self):
        sorted_array = []
        for i in range(len(self.grid[0])):
            for j in range(len(self.grid)):
                current = self.grid[i][j]
                curr_tuple = (i, j, current)
                sorted_array.append(curr_tuple)
        return sorted_array

    def sort_concentration_array(self):
        unsorted_array = self.get_concentration_array()
        sorted_array = sorted(unsorted_array, key = self.take_second, reverse = True)
        return sorted_array

    def update_surrounding_cells(self, sorted_array, copy_grid):
        current_cell = sorted_array.pop(0)
        i = current_cell[0]
        j = current_cell[1]
        concentration = current_cell[2].concentration
        surrounding = []
        for curr_i in range(len(self.grid)):
            for curr_j in range(len(self.grid[0])):
                if curr_j == j and curr_i == i:
                    continue
                elif concentration > self.grid[curr_i][curr_j].concentration:
                    diffusivity = self.grid[i][j].diffusivity
                    concentration1 = concentration
                    concentration2 = self.grid[curr_i][curr_j].concentration
                    width_factor = self.grid[curr_i][curr_j].width
                    height_factor = self.grid[curr_i][curr_j].height
                    squared_length = (abs(curr_i - i) ** 2) + (abs(curr_j - j) ** 2)
                    length = np.math.sqrt(squared_length) * width_factor
                    area = width_factor * height_factor
                    rate = ficks_law(diffusivity, concentration1, concentration2, area, length)
                    record = (curr_i, curr_j, rate)
                    surrounding.append(record)
                    # if curr_i == 3 and curr_j == 2:
                    #     print("This is cell 32 in copy grid that is being updated by cell " + str(i) + str(j) + ": ")
                    #     print("Rate: " + str(rate) + ", Concentration 1: " + str(concentration1) + ", Concentration 2: " + str(concentration2) + ", length: " + str(length) + ", area: " + str(area) + ", diffusivity: " + str(diffusivity))
                    # if curr_i == 0 and curr_j == 0:
                    #     print("This is cell 00 in copy grid that is being updated by cell " + str(i) + str(j) + ": ")
                    #     print("Rate: " + str(rate) + ", Concentration 1: " + str(
                    #         concentration1) + ", Concentration 2: " + str(concentration2) + ", length: " + str(
                    #         length) + ", area: " + str(area) + ", diffusivity: " + str(diffusivity))
                    # if curr_i == 0 and curr_j == 0:
                    #     print("This is cell 00 from the original grid:")
                    #     print(self.grid[curr_i][curr_j].concentration)
                # elif curr_i == 3 and curr_j == 2:
                #     print("Cell 32 has been skipped updating by cell " + str(i) + str(j) + "!")
                #     print("Cell 32 has a concentration of " + str(self.grid[2][3].concentration) + ", while Cell " + str(i) + str(j) + " has a concentration of " + str(self.grid[i][j].concentration))


        # After finding concentrations in surrounding cells, update the concentrations
        # pair_done_stuff = []
        for entry in surrounding:
            if entry[2] is not None:
                i = entry[0]
                j = entry[1]
                # print(str(entry[0]) + str(entry[1]) + ": Concentration rate " + str(entry[2]))
                volume = (float(self.width) ** 2) * float(self.grid[entry[0]][entry[1]].height)
                additional_concentration = (entry[2] * self.time_length) / (volume)
                # print(copy_grid[entry[0]][entry[1]].concentration)
                copy_grid[1][0].concentration += additional_concentration

                # pair_done = (entry[0],entry[1])
                # pair_done_stuff.append(pair_done)

                # print("Cell " + str(0) + str(0) + " in the copy_grid now has a concentration of " + str(
                #     copy_grid[0][0].concentration))

            # if (entry[0] == 0 and entry[1] == 0) or (entry[0] == 3 and entry[1] == 2):
            #
            #     print("Cell " + str(entry[0]) + str(entry[1]) + " has had an addition to its concentration of " + str(additional_concentration))
            #
            #     print("Cell " + str(0) + str(0) + " in the copy_grid now has a concentration of " + str(
            #         copy_grid[0][0].concentration))
            #     print("Cell " + str(3) + str(2) + " in the copy_grid now has a concentration of " + str(
            #         copy_grid[3][2].concentration))

        return copy_grid


    def simple_spread(self):
        # NOTE: This implementation uses a rudimentary approach that involves Fick's Law
        sorted_array = self.sort_concentration_array()
        # print(sorted_array)
        # curr_copy_grid = copy.deepcopy(self.grid)
        iterations = len(sorted_array)
        copy_grid = self.grid
        for i in range(0, iterations):
            copy_grid = self.update_surrounding_cells(sorted_array, copy_grid)
        self.grid = copy_grid









