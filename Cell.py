class Cell:
    def __init__(self, row, column, Agent=None):

        if Agent:
            self.agent = Agent
        self.color = None
        self.concentration_capacity = None
        self.diffusion_rate = None