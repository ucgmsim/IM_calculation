"""
Generates the konno matrices in the IM directory
"""

import os
from IM_calculation.scripts.A_KonnoMatricesComputation import createKonnoMatrices

path = os.getcwd()
print(path)
createKonnoMatrices(path, True, 5)

