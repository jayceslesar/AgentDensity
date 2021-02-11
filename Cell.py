import Agent


class Cell:
    def __init__(self, row: int, column: int, Agent=None):

        # not all cells have Agents
        if Agent is not None:
            self.agent = Agent
        else:
            self.agent = None

        self.color = None
        self.concentration_capacity = None
        self.diffusion_rate = None
