import Agent


class Cell:
    def __init__(self, row: int, column: int, Agent=None):

        # not all cells have Agents
        if Agent is not None:
            self.agent = Agent
        else:
            self.agent = None

        self.color = None
        self.concentration_capacity = 0
        self.diffusion_rate = None
        self.gradient_map = {0.2: (249, 189, 138), 0.4: (246, 135, 86), 0.6: (248, 110, 49), 0.8: (243, 95, 30), 1.0: (251, 69, 3)}

    def get_color(self):
        if self.Agent is not None:
            return agent.get_color()
        else:
            return self._color()

    def _color(self):
        if self.concentration_capacity < 0.2:
            return self.gradient_map[0.2]
        if self.concentration_capacity >= 0.2 and self.concentration_capacity < 0.4:
            return self.gradient_map[0.4]
        if self.concentration_capacity >= 0.4 and self.concentration_capacity < 0.6:
            return self.gradient_map[0.6]
        if self.concentration_capacity >= 0.6 and self.concentration_capacity < 0.8:
            return self.gradient_map[0.8]
        if self.concentration_capacity >= 0.8:
            return self.gradient_map[1.0]

    def __str__(self):
        if self.Agent is not None:
            return str(agent)
        else:
            return 'E'

