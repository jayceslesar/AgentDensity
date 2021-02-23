import Agent
import Virus
import copy


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
        self.color_upper_limit = 12
        # used to keep track of the concetration ages
        self.virus_array = []
        self.total_virus_number = 0

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

    def diffuse(self, rate, time):
        amount = rate*time
        additional_virus = []
        for entry in self.virus_array:
            new_amount = amount * entry.percentage_of_cell
            new_virus = copy.deepcopy(entry)
            new_virus.virus_number = new_amount
            new_virus.percentage_of_cell = 0
            additional_virus.append(new_virus)
        return additional_virus

    def conserve_mass(self, rate, time):
        amount = rate * time
        for entry in self.virus_array:
            new_amount = amount * entry.percentage_of_cell
            entry.decrease_number(new_amount)


    def update_concentration(self):
        total_virus_number = 0
        for virus in self.virus_array:
            total_virus_number += virus.virus_number
        self.total_virus_number = total_virus_number
        volume = self.width * self.width * self.height
        self.concentration = float(self.total_virus_number)/volume

    def update_age(self, time):
        for virus in self.virus_array:
            virus.update_age(time)
            if virus.age >= virus.lifetime:
                self.virus_array.remove(virus)

    def update_percentages(self):
        for virus in self.virus_array:
            virus.update_percentage(self.concentration)

    def update_wo_age(self):
        self.update_concentration()
        self.update_percentages()
    def update(self, time):
        self.update_age(time)
        self.update_concentration()
        self.update_percentages()



    def __str__(self):
        if self.agent is not None:
            return str(self.agent)
        else:
            return 'E'
