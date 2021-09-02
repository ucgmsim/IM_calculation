"""
Generates the konno matrices in the IM directory
"""

import os
from IM_calculation.scripts.A_KonnoMatricesComputation import createKonnoMatrices

createKonnoMatrices(os.getcwd(), num_to_gen=5)
