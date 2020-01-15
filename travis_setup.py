"""
Generates the konno matrices in the IM directory
"""

import os
from IM_calculation.scripts.A_KonnoMatricesComputation import createKonnoMatrices

createKonnoMatrices(os.path.join(os.getcwd(), "IM_calculation", "IM", "KO_matrices"), True, 5)

