import numpy as np

# Scaling number used for age calculation (that is based on length)
AGE_SCALE = 1
# Mean and Standard Deviation for the lifetime of a virus
MEAN_LIFETIME = 3
STD_LIFETIME = 1

class SubtractionError(Exception): pass

class Virus:
    def __init__(self):
        self.lifetime = np.random.normal(MEAN_LIFETIME, STD_LIFETIME, 1)
        self.age = 0
        self.virus_number = 0
        self.percentage_of_cell = 0

    def decrease_number(self, number):
        if self.virus_number < number:
            raise SubtractionError("ERROR: Virus number subtraction is invalid!")
        else:
            self.virus_number = self.virus_number - number

    def update_age(self,time):
        self.age += time

    def update_percentage(self, total):
        self.percentage_of_cell = float(self.virus_number)/total

    def check_inactive(self):
        return self.age > self.lifetime

    def update(self, time, total):
        self.update_age(time)
        self.update_percentage(total)
        return self.check_inactive()

