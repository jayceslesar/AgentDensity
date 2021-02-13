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
import random as rd
import Cell


class Room:
    def __init__(self, num_rows_people: int, num_cols_people: int, num_steps: int):
        """Initialize the instance of this Room

        Args:
            num_rows_people (int): number of actual rows
            num_cols_people (int): number of actual cols
            num_steps (int): number of steps in simulation
        """
        self.initial_infected = rd.randint(0,num_rows_people*num_cols_people)
        self.num_rows_people = num_rows_people
        self.num_cols_people = num_cols_people
        self.iterations = num_steps
        self.steps_taken = 0
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
                a = Agent.Agent(n, i, j)
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
    
    def _step_(self):
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
                self.grid[i][j].agent.total_exposure += self.grid[i][j].concentration_capacity

                # update untouched, exposed
                if self.grid[i][j].concentration_capacity > 0:
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

                    
                
room = Room(3,3,2)
room._step_()