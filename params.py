
# Sim Specific...
SEED = 42
ROWS_PEOPLE = 5
COLS_PEOPLE = 5
HAVE_TEACHER = True
MOVING_AGENT = False
ITERATIONS = 2500
FAN_CYCLES = 4

# viz...
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800

# Agent Specific...
# 2, 8 Simple, 1, 4 efficient
INTAKE_LBOUND = 1
INTAKE_UBOUND = 4

# 100, 200 for simple
EXPOSURE_LBOUND = 500
EXPOSURE_UBOUND = 1500

# how effective masks are (1.0 for no mask as things get scaled by this)
INHALE_MASK_FACTOR = 1.0
EXHALE_MASK_FACTOR = 1.0

#  Cell Speficic (meters)
CELL_WIDTH = 1
CELL_HEIGHT = 1


class Params_Class():
    def __init__(self):
        # Sim Specific...
        self.SEED = 42
        self.ROWS_PEOPLE = 5
        self.COLS_PEOPLE = 5
        self.HAVE_TEACHER = True
        self.MOVING_AGENT = False
        self.ITERATIONS = 2500
        self.FAN_CYCLES = 4
        self.NUM_STEPS = 1000

        # viz...
        self.WINDOW_HEIGHT = 800
        self.WINDOW_WIDTH = 800

        # Agent Specific...
        # 2, 8 Simple, 1, 4 efficient
        self.INTAKE_LBOUND = 1
        self.INTAKE_UBOUND = 4

        # 100, 200 for simple
        self.EXPOSURE_LBOUND = 500
        self.EXPOSURE_UBOUND = 1500

        # how effective masks are (1.0 for no mask as things get scaled by this)
        self.INHALE_MASK_FACTOR = 1.0
        self.EXHALE_MASK_FACTOR = 1.0

        #  Cell Speficic (meters)
        self.CELL_WIDTH = 2
        self.CELL_HEIGHT = 3

    def to_dict(self):
        return vars(self)
