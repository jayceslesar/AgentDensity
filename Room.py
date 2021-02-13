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
import Cell


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


room = Room(3, 3)

print(room)

