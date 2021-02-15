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
        self.diffusivity = 4
        self.gradient_map = {0.0: (255, 255, 255 ), 0.2: (249, 189, 138), 0.4: (246, 135, 86), 0.6: (248, 110, 49), 0.8: (243, 95, 30), 1.0: (251, 69, 3)}

    def get_color(self):
        if self.agent is not None:
            return self.agent.get_color()
        else:
            return self._color()

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

    def __str__(self):
        if self.agent is not None:
            return str(self.agent)
        else:
            return 'E'

