import Agent


class Cell:
    def __init__(self, row: int, column: int, Agent=None):

        # not all cells have Agents
        if Agent is not None:
            self.agent = Agent
            self.production_rate = self.agent.production_rate
        else:
            self.agent = None
            self.production_rate = 0

        self.color = (255, 255, 255)
        self.concentration = 0
        self.width = 2
        self.height = 10
        self.diffusivity = 0.004
        self.gradient_map = {0.0: (255, 255, 255 ), 0.2: (249, 189, 138), 0.4: (246, 135, 86), 0.6: (248, 110, 49), 0.8: (243, 95, 30), 1.0: (251, 69, 3)}
        self.color_upper_limit = 12

    def get_color(self):
        if self.agent is not None:
            return self.agent.get_color()
        else:
            return self.scaled_color()

    def _color(self):
        if self.concentration == 0.0:
            return self.gradient_map[0.0]
        if self.concentration < 0.2:
            return self.gradient_map[0.2]
        if self.concentration >= 0.2 and self.concentration < 0.4:
            return self.gradient_map[0.4]
        if self.concentration >= 0.4 and self.concentration < 0.6:
            return self.gradient_map[0.6]
        if self.concentration >= 0.6 and self.concentration < 0.8:
            return self.gradient_map[0.8]
        if self.concentration >= 0.8:
            return self.gradient_map[1.0]

    def scaled_color(self):
        if self.concentration < (self.color_upper_limit / 2):
            red = 255
            green = 255
            decrease_factor = 255/(self.color_upper_limit / 2)
            blue = 255 - decrease_factor * self.concentration
            color = (red, green, blue)
        else:
            red = 255
            blue = 0
            decrease_factor = 155/(self.color_upper_limit / 2)
            green = 255 - decrease_factor * (self.concentration - (self.color_upper_limit / 2))
            if green >= 100:
                color = (red, green, blue)
            else:
                color = (255, 100, 0)
        return color

    def __str__(self):
        if self.agent is not None:
            return str(self.agent)
        else:
            return 'E'

