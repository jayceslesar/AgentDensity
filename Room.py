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
from scipy.stats import invgamma

# Scaling number used for age calculation (that is based on length)
AGE_SCALE = 1

# Mean and Standard Deviation for the lifetime of a virus
MEAN_LIFETIME = 3
STD_LIFETIME = 1


def ficks_law(diffusivity, concentration1, concentration2, area, length):
    numerator = float((concentration1 - concentration2) * area * diffusivity)
    return (float(numerator))/(float(length))


class Room:
    def __init__(self, num_rows_people: int, num_cols_people: int, num_steps, seed: int):
        # np.random.seed(seed)
        INTAKE_LBOUND = 2
        INTAKE_UBOUND = 8

        EXPOSURE_LBOUND = 100
        EXPOSURE_UBOUND = 200
        """Initialize the instance of this Room

        Args:
            num_rows_people (int): number of actual rows
            num_cols_people (int): number of actual cols
            num_steps (int): number of steps in simulation
            seed (int): the seed to use
        """
        self.num_rows = num_rows_people*2 + 1
        self.num_cols = num_cols_people*2 + 1
        # get center of grid to place initial infectious agent
        self.initial_infectious_row = [i for i in range(self.num_rows) if int(i) % 2 != 0]
        self.initial_infectious_row = self.initial_infectious_row[int((len(self.initial_infectious_row) - 1)/2)]
        self.initial_infectious_col = [i for i in range(self.num_cols) if int(i) % 2 != 0]
        self.initial_infectious_col = self.initial_infectious_col[int((len(self.initial_infectious_col) - 1)/2)]
        # other initializers
        self.iterations = num_steps
        self.seed = seed
        self.steps_taken = 0
        self.time_length = 1
        self.grid = []
        self.max_length = np.math.sqrt((num_rows_people ** 2) + (num_cols_people ** 2))

        a = 2.4
        self.production_rates = invgamma.rvs(a, size=num_cols_people*num_rows_people, loc=7, scale=5)
        np.random.shuffle(self.production_rates)

        n = 0

        # border row for top and bottom rows
        for i in range(self.num_rows):
            row = []
            for j in range(self.num_cols):
                if i % 2 == 0:
                    row.append(Cell.Cell(i, j))
                elif j % 2 != 0:
                    a = Agent.Agent(n, i, j, self.seed)
                    a.production_rate = self.production_rates[n]
                    a.intake_per_step = np.random.randint(INTAKE_LBOUND, INTAKE_UBOUND)
                    a.exposure_boundary = np.random.randint(EXPOSURE_LBOUND, EXPOSURE_UBOUND)
                    # print(a.exposure_boundary)
                    if i == self.initial_infectious_row and j == self.initial_infectious_col:
                        a.infectious = True
                        self.initial_agent = a
                    row.append(Cell.Cell(i, j, a))
                    n += 1
                else:
                    row.append(Cell.Cell(i, j))
            self.grid.append(row)
        self.width = self.grid[0][0].width

    def __str__(self):
        out = ""
        for row in self.grid:
            for cell in row:
                out += str(cell) + " "
            out += "\n"
        return out

    def _step(self):

        # iterate through rows and columns of cells
        for i in range(self.num_rows):
            for j in range(self.num_cols):

                # check if agent is not in cell
                if self.grid[i][j].agent is None:
                    continue

                # update total exposure, += concentration
                self.grid[i][j].agent.total_exposure += self.grid[i][j].concentration * self.grid[i][j].agent.intake_per_step

                # update untouched, exposed
                if self.grid[i][j].agent.total_exposure > self.grid[i][j].agent.exposure_boundary:
                    self.grid[i][j].agent.untouched = False
                    self.grid[i][j].agent.exposed = True

                if self.grid[i][j].agent.exposed:
                    # update steps exposed
                    self.grid[i][j].agent.steps_exposed += 1

                if self.grid[i][j].agent.infectious:
                    # update steps infectious
                    self.grid[i][j].agent.steps_infectious += 1
                    # TODO: @Brandon, this is what changes the cell concentration, please update with formula
                    self.grid[i][j].total_virus_number += (self.grid[i][j].production_rate) * self.time_length

        self.simple_spread()

        self.steps_taken += 1


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

        for entry in surrounding:
            if entry[2] is not None:
                i = entry[0]
                j = entry[1]
                volume = (float(self.width) ** 2) * float(self.grid[entry[0]][entry[1]].height)
                additional_concentration = (entry[2] * self.time_length) / (volume)

                # keep track of the concentration history
                # calculate age first
                width_factor = self.grid[i][j].width
                squared_length = (abs(current_cell[0] - i) ** 2) + (abs(current_cell[1] - j) ** 2)
                length = np.math.sqrt(squared_length) * width_factor
                age = length/(AGE_SCALE * self.max_length) * self.time_length
                # # come up with a lifetime based on a random distribution
                # lifetime = np.random.normal(MEAN_LIFETIME, STD_LIFETIME, 1)
                # # add age to array
                # copy_grid[i][j].concetration_history = (age, lifetime)

                # add additional concetration to the concentration
                copy_grid[i][j].concentration += additional_concentration

                # # subtract the additional concentration from the source in order to satisfy conservation of mass
                # copy_grid[current_cell[0]][current_cell[1]].concentration -= additional_concentration

        return copy_grid


    def simple_spread(self):
        # NOTE: This implementation uses a rudimentary approach that involves Fick's Law
        sorted_array = self.sort_concentration_array()
        iterations = len(sorted_array)
        copy_grid = self.grid
        for i in range(0, iterations):
            copy_grid = self.update_surrounding_cells(sorted_array, copy_grid)
        self.grid = copy_grid
