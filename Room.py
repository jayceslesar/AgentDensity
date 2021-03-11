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
import math
from scipy.stats import invgamma


def ficks_law(diffusivity, concentration1, concentration2, area, length):
    numerator = float((concentration1 - concentration2) * area * diffusivity)
    return (float(numerator))/(float(length))

def advection_equation(velocity, concentration, area, length):
    numerator = float((concentration) * area * velocity)
    return (float(numerator))/(float(length))


class Room:
    def __init__(self, num_rows_people: int, num_cols_people: int, num_steps, seed: int, have_teacher: bool, moving_agent: bool):
        # np.random.seed(seed)
        # 2, 8 Simple, 1, 4 efficient
        INTAKE_LBOUND = 1
        INTAKE_UBOUND = 4

        # 100, 200 for simple
        EXPOSURE_LBOUND = 3000
        EXPOSURE_UBOUND = 4000

        INHALE_MASK_FACTOR = 1.0
        EXHALE_MASK_FACTOR = 1.0
        """Initialize the instance of this Room

        Args:
            num_rows_people (int): number of actual rows
            num_cols_people (int): number of actual cols
            num_steps (int): number of steps in simulation
            seed (int): the seed to use
            have_teacher (bool): implement a teacher on a random side if true, (one row with person in the middle tacked onto top or bottomo)
            moving_agent (bool): if there is a moving agent, if there is then no initial infectious, but a moving infected agent that 'walks' the room
        """
        self.seed = seed
        self.n = 0
        self.num_rows = num_rows_people*2 + 1
        self.num_cols = num_cols_people*2 + 1
        self.expected_n = num_cols_people*num_rows_people
        self.moving_agent = moving_agent
        self.infected_production_rates = list(invgamma.rvs(a=2.4, size=self.expected_n, loc=5, scale=4))
        self.production_rates =  sorted(self.infected_production_rates)[:len(self.infected_production_rates)//2]
        np.random.shuffle(self.production_rates)

        if have_teacher:
            self.expected_n += 1
        # get center of grid to place initial infectious agent
        if not self.moving_agent:
            self.initial_infectious_row = [i for i in range(self.num_rows) if int(i) % 2 != 0]
            # center row
            self.initial_infectious_row = self.initial_infectious_row[int((len(self.initial_infectious_row) - 1)/2)]
            self.initial_infectious_col = [i for i in range(self.num_cols) if int(i) % 2 != 0]
            # center column
            self.initial_infectious_col = self.initial_infectious_col[int((len(self.initial_infectious_col) - 1)/2)]
            self.center_col = self.initial_infectious_col
        else:
            self.center_col = [i for i in range(self.num_cols) if int(i) % 2 != 0]
            # center column
            self.center_col = self.center_col[int((len(self.center_col) - 1)/2)]
            self.initial_infectious_row, self.initial_infectious_col = 0, 0
            production_rate = EXHALE_MASK_FACTOR * np.random.choice(self.production_rates)
            intake_per_step = INHALE_MASK_FACTOR * np.random.uniform(INTAKE_LBOUND, INTAKE_UBOUND)
            exposure_boundary = np.random.uniform(EXPOSURE_LBOUND, EXPOSURE_UBOUND)
            self.agent_to_move = Agent.Agent(self.n, 0, 0, self.seed, INHALE_MASK_FACTOR, EXHALE_MASK_FACTOR, production_rate, intake_per_step, exposure_boundary)
            self.agent_to_move.infectious = True
            self.n += 1
            self.expected_n += 1

        # other initializers
        self.iterations = num_steps
        self.steps_taken = 0
        # 2 for simple, 8 old
        self.time_length = 2
        self.grid = []
        self.ideal_mass = 0.0
        self.actual_mass = 0.0
        self.falloff_rate = 0.0

        for i in range(self.num_rows):
            row = []
            # columns
            for j in range(self.num_cols):
                if i % 2 == 0:
                    row.append(Cell.Cell(i, j))
                elif j % 2 != 0:
                    # agent attributes
                    production_rate = EXHALE_MASK_FACTOR * np.random.choice(self.production_rates)
                    intake_per_step = INHALE_MASK_FACTOR * np.random.uniform(INTAKE_LBOUND, INTAKE_UBOUND)
                    exposure_boundary = np.random.uniform(EXPOSURE_LBOUND, EXPOSURE_UBOUND)
                    a = Agent.Agent(self.n, i, j, self.seed, INHALE_MASK_FACTOR, EXHALE_MASK_FACTOR, production_rate, intake_per_step, exposure_boundary)
                    if i == self.initial_infectious_row and j == self.initial_infectious_col and not self.moving_agent:
                        a.infectious = True
                        a.production_rate = np.random.choice(self.infected_production_rates)
                        self.initial_agent = a
                    row.append(Cell.Cell(i, j, a))
                    self.n += 1
                else:
                    row.append(Cell.Cell(i, j))
            self.grid.append(row)

        # extra two rows for teacher/professor (they are against a wall)
        if have_teacher:
            extra_start = self.num_rows
            self.num_rows += 2
            for i in range(extra_start, self.num_rows):
                row = []
                for j in range(self.num_cols):
                    if j == self.center_col and i != self.num_rows - 1:
                        production_rate = EXHALE_MASK_FACTOR * np.random.choice(self.production_rates)
                        intake_per_step = INHALE_MASK_FACTOR * np.random.uniform(INTAKE_LBOUND, INTAKE_UBOUND)
                        exposure_boundary = np.random.uniform(EXPOSURE_LBOUND, EXPOSURE_UBOUND)
                        a = Agent.Agent(self.n, i, j, self.seed, INHALE_MASK_FACTOR, EXHALE_MASK_FACTOR, production_rate, intake_per_step, exposure_boundary)
                        row.append(Cell.Cell(i, j, a))
                    else:
                        row.append(Cell.Cell(i, j))
                self.grid.append(row)

        self.width = self.grid[0][0].width

        # self.grid[self.num_rows-1][self.center_col].advec_vec = ("u", .05)

        if self.moving_agent:
            self.grid[0][0].agent = self.agent_to_move

        # generate walkable path
        self.moving_path = []
        self.move_index = 0
        for i in range(self.num_rows):
            if i % 2 == 0:
                coordinates_in_row = []
                for j in range(self.num_cols):
                    if (i != 0 and j == 0) or j == self.num_cols:
                        coordinates_in_row.append([i - 1, j])
                        coordinates_in_row.append([i, j])
                    else:
                        coordinates_in_row.append([i, j])
                self.moving_path.append(coordinates_in_row)

        self.moving_path = self.moving_path[:-1]

        for i in range(len(self.moving_path)):
            if i % 2 != 0:
                self.moving_path[i] = list(reversed(self.moving_path[i]))

        for i in range(len(self.moving_path)):
            if i % 2 != 0:
                self.moving_path[i][-1][1] = self.num_cols - 1
                self.moving_path[i-1].append(self.moving_path[i][-1])
                self.moving_path[i] = self.moving_path[i][:-1]

        self.moving_path = [item for sublist in self.moving_path for item in sublist]
        for rev in list(reversed(self.moving_path)):
            self.moving_path.append(rev)

    def __str__(self):
        out = ""
        for row in self.grid:
            for cell in row:
                out += str(cell) + " "
            out += "\n"
        return out

    def _move(self):
        """Handles how the moving agent moves"""
        if self.move_index >= len(self.moving_path) - 1:
            self.move_index = 0

        from_row, from_col = self.moving_path[self.move_index]
        to_row, to_col = self.moving_path[self.move_index + 1]
        self.grid[from_row][from_col].agent = None

        self.grid[to_row][to_col].agent = self.agent_to_move

        self.move_index += 1


    def _step(self):
        # print([[self.grid[i][j].concentration for j in range(self.num_cols)] for i in range(self.num_rows)])
        """Represents one step in the simulation."""
        if self.moving_agent:
            # every 5 steps
            if self.steps_taken % 20 == 0:
                self._move()
        self.fallout()
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
                    self.grid[i][j].concentration += (self.grid[i][j].agent.production_rate) * self.time_length

        # Checking conservation of mass
        self.ideal_mass = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                width_factor = self.grid[i][j].width
                height_factor = self.grid[i][j].height
                self.ideal_mass += self.grid[i][j].concentration*(width_factor**2*height_factor)
        self.efficient_spread()
        self.advection()
        self.actual_mass = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                width_factor = self.grid[i][j].width
                height_factor = self.grid[i][j].height
                self.actual_mass += self.grid[i][j].concentration*(width_factor**2*height_factor)
        # if abs(self.ideal_mass - self.actual_mass) <= .5:
        #     print('mass conserved.')
        # else:
        #     print(self.ideal_mass, self.actual_mass)
        self.steps_taken += 1

    def take_second(self, element):
        """Get the element at index 2 in a make shift struct."""
        return element[2].concentration

    def get_sorted_concentration_array(self):
        """Get the list of concentrations"""
        ret = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                current = self.grid[i][j]
                curr_tuple = (i, j, current)
                ret.append(curr_tuple)
        sorted_array = sorted(ret, key = self.take_second, reverse = True)
        return ret

    def update_surrounding_cells(self, sorted_array, copy_grid):
        """Updates the surrounding cells with concentration diffusion from target cell."""
        current_cell = sorted_array.pop(0)
        i = current_cell[0]
        j = current_cell[1]
        concentration = current_cell[2].concentration
        surrounding = []
        for curr_i in range(self.num_rows):
            for curr_j in range(self.num_cols):
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
                copy_grid[i][j].concentration += additional_concentration
                copy_grid[current_cell[0]][current_cell[1]].concentration -= additional_concentration

        return copy_grid

    def simple_spread(self):
        """Runs a spread of grid"""
        # NOTE: This implementation uses a rudimentary approach that involves Fick's Law
        sorted_array = self.get_sorted_concentration_array()
        iterations = len(sorted_array)
        copy_grid = self.grid
        for i in range(0, iterations):
            copy_grid = self.update_surrounding_cells(sorted_array, copy_grid)
        self.grid = copy_grid

    def get_coordinate_list(self, i, j):
        """gets list of lower, higher, left, and right cell

        Args:
            i (int): origin row index
            j (int): origin col index

        Returns:
            list: list of tuples of coordinates for grid
        """
        if i == 0 and j == 0:
            return [(i + 1, j), (i, j + 1)]
        if i == 0 and j == self.num_cols:
            return [(i + 1, j), (i, j - 1)]
        if i == self.num_rows and j == 0:
            return [(i - 1, j), (i, j + 1)]
        if i == self.num_rows and j == self.num_cols:
            return [(i - 1, j), (i, j - 1)]
        if i == 0:
            return [(i, j - 1), (i, j + 1), (i +  1, j)]
        if i == self.num_rows:
            return [(i, j - 1), (i, j + 1), (i -  1, j)]
        if j == 0:
            return [(i - 1, j), (i + 1, j), (i, j + 1)]
        if j == self.num_cols:
            return [(i - 1, j), (i + 1, j), (i, j - 1)]


        return [(i, j-1), (i, j+1), (i+1,j), (i-1,j)]

    def efficient_spread(self):
        """ Linear algorithm for calculating flux across grid
        """
        copy_grid = copy.deepcopy(self.grid)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                total_flux = 0
                num_fluxes = 0
                concentration1 = self.grid[i][j].concentration
                diffusivity = self.grid[i][j].diffusivity
                width_factor = self.grid[i][j].width
                height_factor = self.grid[i][j].height
                length = width_factor
                area = width_factor * height_factor
                for coor in self.get_coordinate_list(i,j):
                    try:
                        concentration2 = self.grid[coor[0]][coor[1]].concentration
                        total_flux += ficks_law(diffusivity, concentration1, concentration2, area, length)
                        num_fluxes += 1
                    except IndexError:
                        pass
                copy_grid[i][j].concentration -= (total_flux/num_fluxes)*self.time_length/(width_factor**2*height_factor)
        self.grid = copy_grid

    def direct_vector(self, direction, coordinate):
        affected_vector = []
        if direction == 'u':
            for i in range(coordinate[0]-1, -1, -1):
                cur_cor = (i, coordinate[1])
                affected_vector.append(cur_cor)
        elif direction =='d':
            print('test2')
            for i in range(coordinate[0]+1, self.num_rows):
                cur_cor = (i, coordinate[1])
                affected_vector.append(cur_cor)
        elif direction == 'r':
            print('test3')
            for j in range(coordinate[1]+1, self.num_cols):
                cur_cor = (coordinate[0], j)
                affected_vector.append(cur_cor)
        elif direction == 'l':
            print('test4')
            for j in range(coordinate[1]-1, -1, -1):
                cur_cor = (coordinate[0], j)
                affected_vector.append(cur_cor)
        return affected_vector

    def advection(self):
        copy_grid = copy.deepcopy(self.grid)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                width_factor = self.grid[i][j].width
                height_factor = self.grid[i][j].height
                area = width_factor * height_factor
                # print(copy_grid[i][j].advec_vec)
                if copy_grid[i][j].advec_vec is not None:
                    affected_vector = self.direct_vector(copy_grid[i][j].advec_vec[0],(i,j))
                    for c in range(len(affected_vector)-1):
                        distance = math.sqrt((i - affected_vector[c][0])**2 + (j - affected_vector[c][1])**2)*width_factor
                        change = advection_equation(copy_grid[i][j].advec_vec[1], self.grid[affected_vector[c][0]][affected_vector[c][1]].concentration, area, distance)
                        copy_grid[affected_vector[c+1][0]][affected_vector[c+1][1]].concentration += change
                        copy_grid[affected_vector[c][0]][affected_vector[c][1]].concentration -= change
        self.grid = copy_grid

    def fallout(self):
        """Represents the fallout of particles in the air."""
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.falloff_rate = np.random.normal(0.05, 0.001, 1)[0]
                self.grid[i][j].concentration = self.grid[i][j].concentration * (1 - self.falloff_rate)
