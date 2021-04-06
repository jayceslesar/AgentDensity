
# Sim Specific...
SEED = 42
ROWS_PEOPLE = 5
COLS_PEOPLE = 5
HAVE_TEACHER = True
MOVING_AGENT = False
ITERATIONS = 10000000
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
CELL_HEIGHT = 3


class Sim_Params():
    def __init__(self):
        # Sim Specific...
        SEED = 42
        ROWS_PEOPLE = 5
        COLS_PEOPLE = 5
        HAVE_TEACHER = True
        MOVING_AGENT = False
        ITERATIONS = 10000000
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