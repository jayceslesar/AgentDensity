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


def advection_equation(velocity, concentration, area):
    return float((concentration) * area * velocity)


class Room:
    def __init__(self, sim_params: dict):
        np.random.seed(sim_params['SEED'])
        """Initialize the instance of this Room

        Args:
            sim_params (dict): dictionary of parameters that get passed in
        """
        self.sim_params = sim_params
        self.seed = self.sim_params['SEED']
        self.n = 0
        self.num_rows = self.sim_params['ROWS_PEOPLE']*2 + 1
        self.num_cols = self.sim_params['COLS_PEOPLE']*2 + 1
        self.expected_n = self.sim_params['ROWS_PEOPLE']*self.sim_params['COLS_PEOPLE']
        self.moving_agent = self.sim_params['MOVING_AGENT']
        self.infected_production_rates = list(invgamma.rvs(a=2.4, size=self.expected_n, loc=5, scale=4))
        self.production_rates = sorted(self.infected_production_rates)[:len(self.infected_production_rates)//2]
        np.random.shuffle(self.production_rates)

        if self.sim_params['HAVE_TEACHER']:
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
            production_rate = np.random.exponential(scale=400) / 0.001  # to convert to cubic
            self.agent_to_move = Agent.Agent(self.n, 0, 0, self.seed, production_rate, self.sim_params)
            self.agent_to_move.infectious = True
            self.n += 1
            self.expected_n += 1

        self.fields = ["steps_taken", "difference", "close", "far", "ratio"]
        self.rows = []
        self.filename = "concentration_graph.csv"

        # other initializers
        self.iterations = self.sim_params['ITERATIONS']
        print(self.iterations)
        self.steps_taken = 0
        # 2 for simple, 8 old
        self.time_length = 1
        self.grid = []
        self.ideal_mass = 0.0
        self.actual_mass = 0.0
        self.falloff_rate = 1.7504e-4
        self.total_volume = 0

        for i in range(self.num_rows):
            row = []
            # columns
            for j in range(self.num_cols):
                if i % 2 == 0:
                    row.append(Cell.Cell(i, j, self.sim_params))
                elif j % 2 != 0:
                    production_rate = np.random.exponential(scale=400) / 0.001

                    a = Agent.Agent(self.n, i, j, self.seed, production_rate, self.sim_params)

                    if i == self.initial_infectious_row and j == self.initial_infectious_col and not self.moving_agent:
                        a.infectious = True
                        self.initial_agent = a
                    row.append(Cell.Cell(i, j, self.sim_params, a))
                    self.n += 1
                else:
                    row.append(Cell.Cell(i, j, self.sim_params))
                self.total_volume += row[j].volume
            self.grid.append(row)

        # extra two rows for teacher/professor (they are against a wall)
        # TODO: set it so that the teacher is the infected individual
        if self.sim_params['HAVE_TEACHER']:
            extra_start = self.num_rows
            self.num_rows += 2
            for i in range(extra_start, self.num_rows):
                row = []
                for j in range(self.num_cols):
                    if j == self.center_col and i != self.num_rows - 1:
                        production_rate = np.random.exponential(scale=400) / 0.001
                        a = Agent.Agent(self.n, i, j, self.seed, production_rate, self.sim_params)
                        row.append(Cell.Cell(i, j, self.sim_params, a))
                    else:
                        row.append(Cell.Cell(i, j, self.sim_params))
                self.grid.append(row)

        self.width = self.grid[0][0].width

        # TODO: parameterize this
        # Set sink and source
        self.grid[0][0].source = True
        self.grid[0][0].acr = 0.0001861 * self.total_volume
        self.grid[self.num_rows-1][self.num_cols-1].sink = True
        self.grid[self.num_rows-1][self.num_cols-1].acr = 0.0008333 * self.total_volume
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
        # print(self.steps_taken)
        # print(self.grid[0][0].concentration)
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
                self.grid[i][j].agent.total_exposure += self.grid[i][j].concentration * self.grid[i][j].agent.intake_per_step * self.time_length

                if self.grid[i][j].agent.exposed:
                    # update steps exposed
                    self.grid[i][j].agent.steps_exposed += 1

                if self.grid[i][j].agent.infectious:
                    # update steps infectious
                    self.grid[i][j].agent.steps_infectious += 1
                    self.grid[i][j].add_concentration((self.grid[i][j].agent.production_rate) * self.time_length)

        # Checking conservation of mass
        self.ideal_mass = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                width_factor = self.grid[i][j].width
                height_factor = self.grid[i][j].height
                if self.grid[i][j].agent is None:
                    self.ideal_mass += self.grid[i][j].concentration*(width_factor**2*height_factor)
                else:
                    self.ideal_mass += self.grid[i][j].concentration*(width_factor**2*height_factor - self.grid[i][j].agent.volume)
        self.efficient_spread()
        self.advection()
        self.actual_mass = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                width_factor = self.grid[i][j].width
                height_factor = self.grid[i][j].height
                if self.grid[i][j].agent is None:
                    self.actual_mass += self.grid[i][j].concentration*(width_factor**2*height_factor)
                else:
                    self.actual_mass += self.grid[i][j].concentration*(width_factor**2*height_factor - self.grid[i][j].agent.volume)
        # if abs(self.ideal_mass - self.actual_mass) / self.ideal_mass <= .01:
        #     print('mass conserved.')
        # else:
        #     print(abs(self.ideal_mass - self.actual_mass) / self.ideal_mass)
        self.steps_taken += 1

        close = self.grid[self.initial_infectious_row + 1][self.initial_infectious_col].concentration
        far = self.grid[0][0].concentration
        diff = close - far
        ratio = far/close

    def take_second(self, element):
        """Get the element at index 2 in a make shift struct."""
        return element[2].concentration

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
            return [(i, j - 1), (i, j + 1), (i + 1, j)]
        if i == self.num_rows:
            return [(i, j - 1), (i, j + 1), (i - 1, j)]
        if j == 0:
            return [(i - 1, j), (i + 1, j), (i, j + 1)]
        if j == self.num_cols:
            return [(i - 1, j), (i + 1, j), (i, j - 1)]

        return [(i, j - 1), (i, j + 1), (i + 1, j), (i - 1, j)]

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
                for coor in self.get_coordinate_list(i, j):
                    try:
                        concentration2 = self.grid[coor[0]][coor[1]].concentration
                        total_flux += ficks_law(diffusivity, concentration1, concentration2, area, length)
                        num_fluxes += 1
                    except IndexError:
                        pass
                copy_grid[i][j].add_concentration(-1 * ((total_flux/num_fluxes)*self.time_length))
        self.grid = copy_grid

    def advection(self):
        copy_grid = copy.deepcopy(self.grid)
        source_sink_list = []
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if copy_grid[i][j].sink:
                    source_sink_list.append((i,j, True))
                elif copy_grid[i][j].source:
                    source_sink_list.append((i,j, False))

        for cell_info in source_sink_list:
            i = cell_info[0]
            j = cell_info[1]
            sink = cell_info[2]
            area = self.grid[i][j].height*self.grid[i][j].width
            for x in range(self.num_rows):
                for y in range(self.num_cols):
                    if x == i and y == j:
                        continue
                    if sink:
                        x_component = j - y
                        y_component = i - x
                    else:
                        x_component = y - i
                        y_component = x - i
                    sum_component = abs(x_component) + abs(y_component)
                    x_proportion = abs(x_component)/sum_component
                    y_proportion = abs(y_component)/sum_component

                    # TODO: Change this to the inverse square law
                    surface_area = (abs(x_component) + 1) * (abs(y_component) + 1)
                    velocity = self.grid[i][j].acr / surface_area

                    change = advection_equation(velocity, self.grid[x][y].concentration, area) * self.time_length

                    amount_to_left_right = x_proportion * change
                    amount_to_up_down = y_proportion * change

                    copy_grid[x][y].add_concentration(-1 * change)

                    skip_x = False
                    skip_y = False

                    if x_component > 0 and y+1 == self.num_cols:
                        skip_x = True
                    elif x_component < 0 and y - 1 == 0:
                        skip_x = True
                    if y_component > 0 and x + 1 == self.num_rows:
                        skip_y = True
                    elif y_component < 0 and x - 1 == 0:
                        skip_y = True

                    if x_component > 0 and not skip_x:
                        if skip_y:
                            try:
                                copy_grid[x][y - 1].add_concentration(amount_to_up_down*.5)
                                amount_to_left_right += amount_to_up_down*.5
                            except:
                                amount_to_left_right += amount_to_up_down
                        copy_grid[x][y + 1].add_concentration(amount_to_left_right)
                    elif x_component < 0 and not skip_x:
                        if skip_y:
                            try:
                                copy_grid[x][y + 1].add_concentration(amount_to_up_down*.5)
                                amount_to_left_right += amount_to_up_down*.5
                            except:
                                amount_to_left_right += amount_to_up_down
                        copy_grid[x][y - 1].add_concentration(amount_to_left_right)

                    if y_component < 0 and not skip_y:
                        if skip_x:
                            try:
                                copy_grid[x + 1][y].add_concentration(amount_to_left_right*.5)
                                amount_to_up_down += amount_to_left_right*.5
                            except:
                                amount_to_up_down += amount_to_left_right
                        copy_grid[x - 1][y].add_concentration(amount_to_up_down)
                    elif y_component > 0 and not skip_y:
                        if skip_x:
                            try:
                                copy_grid[x - 1][y].add_concentration(amount_to_left_right*.5)
                                amount_to_up_down += amount_to_left_right*.5
                            except:
                                amount_to_up_down += amount_to_left_right
                        copy_grid[x + 1][y].add_concentration(amount_to_up_down)

        self.grid = copy_grid

    def fallout(self):
        """Represents the fallout of particles in the air."""
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.grid[i][j].concentration = self.grid[i][j].concentration * (1 - self.falloff_rate)
