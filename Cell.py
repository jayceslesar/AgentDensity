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
        self.gradient_map = {0.0: (255, 255, 255 ), 0.1: (255, 242, 230), 0.2: (255,229,204), 0.3: (255, 217, 179), 0.4: (255, 204, 153), 0.5: (255, 191, 128),
                             0.6: (255, 178, 102), 0.7: (255, 165, 77), 0.8: (255, 153, 51), 0.9: (255, 140, 25), 1.0: (255, 127, 0)}

    def get_color(self):
        if self.agent is not None:
            return self.agent.get_color()
        else:
            return self._color()

    def _color(self):
        return self.gradient_map[round(self.concentration, 1)]

    def __str__(self):
        if self.agent is not None:
            return str(self.agent)
        else:
            return 'E'

