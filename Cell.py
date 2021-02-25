import Agent


class Cell:
    def __init__(self, row: int, column: int, Agent=None):

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
        self.diffusivity = 4
        self.gradient_map = {0.0: (255, 255, 255 ), 0.1: (255, 242, 230), 0.2: (255,229,204), 0.3: (255, 217, 179), 0.4: (255, 204, 153), 0.5: (255, 191, 128),
                             0.6: (255, 178, 102), 0.7: (255, 165, 77), 0.8: (255, 153, 51), 0.9: (255, 140, 25), 1.0: (255, 127, 0)}
        self.diffusivity = 0.004
        self.color_upper_limit = .75

    def get_color(self):
        return self.scaled_color()

    def _color(self):
        try:
            return self.gradient_map[round(self.concentration, 1)]
        except:
            return self.gradient_map[1.0]

    def scaled_color(self):
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
