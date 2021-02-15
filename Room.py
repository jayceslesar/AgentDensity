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
        self.num_rows_people = num_rows_people
        self.num_cols_people = num_cols_people
        self.iterations = num_steps
        self.seed = seed
        self.steps_taken = 0
        self.time_length = 1
        self.grid = []

        n = 0

        # border row for top and bottom rows
        blank_row = [Cell.Cell(0,i) for i in range(num_cols_people*2 + 1)]
        self.grid.append(blank_row)
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
        # Here is where I will call spread

        # iterate through rows and columns of cells
        for i in range(len(self.grid[0])):
            for j in range(len(self.grid)):

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
                if self.grid[i][j].agent.total_exposure > INFECTED_CUTOFF:
                    self.grid[i][j].agent.infected = True

                if self.grid[i][j].agent.infected:
                    # update steps infected
                    self.grid[i][j].agent.steps_infected += 1

        self.steps_taken += 1

    def simple_spread(self):
        # NOTE: This implementation uses a rudimentary approach that involves Fick's Law
        for i in range(len(self.grid[0])):
            for j in range(len(self.grid)):

                # make an array to store surrounding cells rates
                surrounding = []
                # check to see if surrounding cells exist
                # upper left check (i-1,j-1)
                if (i - 1) > 0 and (j - 1) > 0:
                    # check concentrations. If greater, than do calculation
                    if self.grid[i][j].concentration > self.grid[i - 1][j - 1].concentration:
                        diffusivity = self.grid[i][j].diffusivity
                        concentration1 = self.grid[i][j].concentration
                        concentration2 = self.grid[i-1][j-1].concentration
                        area = np.math.sqrt(2) * self.grid[i - 1][j - 1].width * self.grid[i - 1][j - 1].height
                        length = np.math.sqrt(2) * self.grid[i - 1][j - 1].width
                        rate = ficks_law(diffusivity, concentration1, concentration2, area, length)
                        record = (i - 1, j - 1, rate)
                        surrounding.append(record)
                    else:
                        record = (i - 1, j - 1, None)
                        surrounding.append(record)

                # upper middle check (i-1,j)
                if (i - 1) >= 0:
                    # check concentrations. If greater, than do calculation
                    if self.grid[i][j].concentration > self.grid[i - 1][j].concentration:
                        diffusivity = self.grid[i][j].diffusivity
                        concentration1 = self.grid[i][j].concentration
                        concentration2 = self.grid[i-1][j].concentration
                        area = self.grid[i - 1][j].width * self.grid[i - 1][j].height
                        length = self.grid[i - 1][j].width
                        rate = ficks_law(diffusivity, concentration1, concentration2, area, length)
                        record = (i - 1, j, rate)
                        surrounding.append(record)
                    else:
                        record = (i - 1, j, None)
                        surrounding.append(record)

                # upper right check (i-1,j+1)
                if (i - 1) >= 0 and (j + 1) <= len(self.grid):
                    # check concentrations. If greater, than do calculation
                    if self.grid[i][j].concentration > self.grid[i - 1][j + 1].concentration:
                        diffusivity = self.grid[i][j].diffusivity
                        concentration1 = self.grid[i][j].concentration
                        concentration2 = self.grid[i-1][j+1].concentration
                        area = self.grid[i - 1][j + 1].width * self.grid[i - 1][j + 1].height
                        length = self.grid[i - 1][j + 1].width
                        rate = ficks_law(diffusivity, concentration1, concentration2, area, length)
                        record = (i - 1, j + 1, rate)
                        surrounding.append(record)
                    else:
                        record = (i - 1, j + 1, None)
                        surrounding.append(record)

                # middle left check (i,j-1)
                if (j - 1) >= 0:
                    # check concentrations. If greater, than do calculation
                    if self.grid[i][j].concentration > self.grid[i][j - 1].concentration:
                        diffusivity = self.grid[i][j].diffusivity
                        concentration1 = self.grid[i][j].concentration
                        concentration2 = self.grid[i][j-1].concentration
                        area = self.grid[i][j - 1].width * self.grid[i][j - 1].height
                        length = self.grid[i][j - 1].width
                        rate = ficks_law(diffusivity, concentration1, concentration2, area, length)
                        record = (i, j - 1, rate)
                        surrounding.append(record)
                    else:
                        record = (i, j - 1, None)
                        surrounding.append(record)

                # middle right check (i,j+1)
                if (j + 1) <= len(self.grid):
                    # check concentrations. If greater, than do calculation
                    if self.grid[i][j].concentration > self.grid[i][j + 1].concentration:
                        diffusivity = self.grid[i][j].diffusivity
                        concentration1 = self.grid[i][j].concentration
                        concentration2 = self.grid[i][j+1].concentration
                        area = self.grid[i][j+1].width * self.grid[i][j+1].height
                        length = self.grid[i][j+1].width
                        rate = ficks_law(diffusivity, concentration1, concentration2, area, length)
                        record = (i, j + 1, rate)
                        surrounding.append(record)
                    else:
                        record = (i, j + 1, None)
                        surrounding.append(record)

                # bottom left check (i+1,j-1)
                if (i + 1) <= len(self.grid[0]) and (j - 1) >= 0:
                    # check concentrations. If greater, than do calculation
                    if self.grid[i][j].concentration > self.grid[i+1][j-1].concentration:
                        diffusivity = self.grid[i][j].diffusivity
                        concentration1 = self.grid[i][j].concentration
                        concentration2 = self.grid[i+1][j-1].concentration
                        area = self.grid[i+1][j-1].width * self.grid[i+1][j-1].height
                        length = self.grid[i+1][j-1].width
                        rate = ficks_law(diffusivity, concentration1, concentration2, area, length)
                        record = (i + 1, j - 1, rate)
                        surrounding.append(record)
                    else:
                        record = (i + 1, j - 1, None)
                        surrounding.append(record)

                # bottom middle check (i+1,j)
                if (i + 1) <= len(self.grid[0]):
                    # check concentrations. If greater, than do calculation
                    if self.grid[i][j].concentration > self.grid[i+1][j].concentration:
                        diffusivity = self.grid[i][j].diffusivity
                        concentration1 = self.grid[i][j].concentration
                        concentration2 = self.grid[i+1][j].concentration
                        area = self.grid[i+1][j].width * self.grid[i+1][j].height
                        length = self.grid[i+1][j].width
                        rate = ficks_law(diffusivity, concentration1, concentration2, area, length)
                        record = (i + 1, j, rate)
                        surrounding.append(record)
                    else:
                        record = (i + 1, j, None)
                        surrounding.append(record)

                # bottom right check (i+1,j+1)
                if (i + 1) <= len(self.grid[0]) and (j + 1) <= len(self.grid):
                    # check concentrations. If greater, than do calculation
                    if self.grid[i][j].concentration > self.grid[i+1][j+1].concentration:
                        diffusivity = self.grid[i][j].diffusivity
                        concentration1 = self.grid[i][j].concentration
                        concentration2 = self.grid[i+1][j+1].concentration
                        area = self.grid[i+1][j+1].width * self.grid[i+1][j+1].height
                        length = self.grid[i+1][j+1].width
                        rate = ficks_law(diffusivity, concentration1, concentration2, area, length)
                        record = (i + 1, j + 1, rate)
                        surrounding.append(record)
                    else:
                        record = (i + 1, j + 1, None)
                        surrounding.append(record)




