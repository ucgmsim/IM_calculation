"""
Generates the konno matrices in the IM directory
"""

from pathlib import Path
from IM_calculation.scripts.A_KonnoMatricesComputation import createKonnoMatrices

createKonnoMatrices(
    Path(__file__) / "IM_calculation" / "IM" / "KO_matrices", num_to_gen=5
)
