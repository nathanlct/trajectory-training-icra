import math
import numpy as np


GRAMS_PER_SEC_TO_GALS_PER_HOUR = {
    'diesel': 1.119,  # 1.119 gal/hr = 1g/s
    'gasoline': 1.268,  # 1.268 gal/hr = 1g/s
}

GRAMS_TO_JOULES = {
    'diesel': 42470,
    'gasoline': 42360,
}


class PFM2019RAV4(object):
    """Energy model not released as part of this work."""
    def __init__(self):
        pass

    def get_instantaneous_fuel_consumption(self, accel, speed, grade):
        raise ValueError('The PFM2019RAV4 energy model is private at the time of publication.')
