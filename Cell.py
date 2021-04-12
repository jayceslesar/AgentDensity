

class Cell:
    def __init__(self, row: int, column: int, sim_params: dict, Agent=None):
        """Initialize a cell class

        Args:
            row (int): row of cell
            column (int): column of cell
            Agent ([Agent], optional): Some cells have an Agent. Defaults to None.
            width (float): width of cell
            height (float): height of cell

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
        self.width = sim_params['CELL_WIDTH']  # in meters
        self.height = sim_params['CELL_HEIGHT']  # in meters
        # 0.004 for ss

        self.real_diffusivity = 2.83e-5
        self.micro_current_factor = 1000
        self.diffusivity = self.micro_current_factor*self.real_diffusivity
        # .75 for ss
        self.color_upper_limit = 0.000000000000000075
        self.advec_vec = None

        self.sink = False
        self.source = False

        self.temperature = sim_params["TEMPERATURE"]
        self.gas_const = sim_params["GAS_CONST"]
        self.mols = sim_params["MOLS"]*width**2*height
        self.molar_mass_a = sim_params["MOLAR_MASS_A"]
        self.molar_mass_w = sim_params["MOLAR_MASS_W"]

    def get_w_mol_prop(self):
        return self.concentration*width**2*height/self.molar_mass_w / self.mols

    def get_color(self):
        """Represent the color of the cell by the concentration inside."""
        if self.concentration < 0:
            print("concentration is less than 0!!!!!")
        # print(self.concentration)
        if self.concentration < (self.color_upper_limit / 2):
            green = 255
            blue = 255
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
        # print(color)
        return color

    # Could pad grid with filters to make this work, but what if filtration system is inside room?
    # For windows, there is no suction, just the diffusion pulling concentration out the window
    def add_concentration(self, term):
        self.concentration += term
        if self.sink:
            self.concentration = 0

    def __str__(self):
        if self.agent is not None:
            return str(self.agent)
        else:
            return 'E'
