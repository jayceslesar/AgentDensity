import Agent


class Cell:
    def __init__(self, row: int, column: int, Agent=None):
        """Initialize a cell class

        Args:
            row (int): row of cell
            column (int): column of cell
            Agent ([Agent], optional): Some cells have an Agent. Defaults to None.
        """

        # not all cells have Agents
        if Agent is not None:
            self.agent = Agent
            self.production_rate = self.agent.production_rate
        else:
            self.agent = None
            self.production_rate = None

        self.row = row
        self.column = column
        self.color = (255, 255, 255)
        self.concentration = 0
        self.width = 2
        self.height = 10
        self.diffusivity = 0.004
        self.color_upper_limit = .75

    def get_color(self):
        """Represent the color of the cell by the concentration inside."""
        if self.concentration < (self.color_upper_limit / 2):
            green = 255
            blue  = 255
            decrease_factor = 255/(self.color_upper_limit / 2)
            red = 255 - decrease_factor * self.concentration
            color = (red, green, blue)
        else:
            red = 0
            blue = 255
            decrease_factor = 255/(self.color_upper_limit / 2)
            green = 255 - decrease_factor * (self.concentration - (self.color_upper_limit / 2))
            if green >= 0:
                color = (red, green, blue)
            else:
                color = (0, 0, 255)
        return color

    def __str__(self):
        if self.agent is not None:
            return str(self.agent)
        else:
            return 'E'
