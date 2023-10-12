#######################################################################
# EMLE-Engine: https://github.com/chemle/emle-engine
#
# Copyright: 2023
#
# Authors: Lester Hedges   <lester.hedges@gmail.com>
#          Kirill Zinovjev <kzinovjev@gmail.com>
#
# EMLE-Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# EMLE-Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EMLE-Engine If not, see <http://www.gnu.org/licenses/>.
#####################################################################

import ase
from ase.calculators.calculator import Calculator, CalculatorSetupError, all_changes
import numpy as np
import sander


class SanderCalculator(Calculator):
    kcalmol_to_eV = ase.units.kcal / ase.units.mol
    implemented_properties = ["energy", "forces"]

    def __init__(self, parm7, atoms, is_gas=True):
        super().__init__()
        if sander.is_setup():
            sander.cleanup()
        if is_gas:
            sander.setup(
                parm7, atoms.get_positions(), self._get_box(atoms), sander.gas_input()
            )
        else:
            sander.setup(
                parm7, atoms.get_positions(), self._get_box(atoms), sander.pme_input()
            )

    def calculate(
        self, atoms, properties=["energy", "forces"], system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
        sander.set_positions(atoms.get_positions())
        box = self._get_box(atoms)
        if box is not None:
            sander.set_box(*box)

        energy, forces = sander.energy_forces()
        self.results = {
            "energy": energy.tot * self.kcalmol_to_eV,
            "forces": np.array(forces).reshape((-1, 3)) * self.kcalmol_to_eV,
        }

    @staticmethod
    def _get_box(atoms):
        if not atoms.get_pbc().all():
            return None
        else:
            return atoms.get_cell().cellpar()
