"""
Authors:
---
    Jayce Slesar
    Brandon Lee

Date:
---
    12/29/2020
"""

import Agent
import numpy as np
import random as rd


class Room:
    def __init__(self, num_rows_people: int, num_cols_people: int):
        """Initialize the instance of this Room

        Args:
            num_rows_people (int): number of actual rows
            num_cols_people (int): number of actual cols
        """
        self.initial_infected = rd.randint(0,num_rows_people*num_cols_people)
        self.num_rows_people = num_rows_people
        self.num_cols_people = num_cols_people

        self.INCUBATION_PERIOD_DISTRIBUTION = list(np.absolute(np.around(np.random.normal(loc=3, scale=1.5, size=(num_rows_people*num_cols_people))).astype(int)))
        self.INFECTIVE_LENGTH_DISTRUBUTION = list(np.around(np.random.normal(loc=10.5, scale=3.5, size=(num_rows_people*num_cols_people))).astype(int))

        self.grid = []

        n = 0

        # border row for top and bottom rows
        blank_row = ['E' for i in range(num_cols_people*2 + 1)]
        self.grid.append(blank_row)
        for i in range(self.num_rows_people):
            row = []
            # add an empty space
            row.append('E')
            for j in range(self.num_cols_people):
                a = Agent.Agent(n, i, j)
                a.INCUBATION_PERIOD = self.INCUBATION_PERIOD_DISTRIBUTION.pop(rd.randint(0, len(self.INCUBATION_PERIOD_DISTRIBUTION) - 1))
                a.INFECTIVE_LENGTH = self.INFECTIVE_LENGTH_DISTRUBUTION.pop(rd.randint(0, len(self.INFECTIVE_LENGTH_DISTRUBUTION) - 1))
                if n == self.initial_infected:
                    a.infected = True
                    self.initial_agent = a
                row.append(a)
                # add an empty space
                row.append('E')
                n += 1
            self.grid.append(row)
            self.grid.append(blank_row)

    def __str__(self):
        out = ""
        for row in self.grid:
            for agent in row:
                out += str(agent) + " "
            out += "\n"
        return out


room = Room(3, 3)

print(room)

