#######################################################################
# EMLE-Engine: https://github.com/chemle/emle-engine
#
# Copyright: 2023-2025
#
# Authors: Lester Hedges   <lester.hedges@gmail.com>
#          Kirill Zinovjev <kzinovjev@gmail.com>
#          Joao Morado     <joaomorado@gmail.com>
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
# along with EMLE-Engine. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

"""Exchange-repulsion interaction."""

__author__ = "Joao Morado"
__email__ = "joaomorado@gmail.com"

__all__ = ["ExchangeRepulsion"]

import torch as _torch
from ._base import BaseInteraction


class ExchangeRepulsion(BaseInteraction):
    """
    Exchange-repulsion energy correction.

    Accounts for Pauli repulsion between QM and MM electron densities.
    """

    def __init__(self, emle_base, device=None, dtype=None):
        """
        Constructor.

        Parameters
        ----------

        emle_base: EMLEBase
            Reference to EMLEBase.

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for floating point tensors.
        """
        super().__init__(device=device, dtype=dtype)

        self._emle_base = emle_base

    def forward(self, A_exrep_qm, A_exrep_mm, q_val_qm, q_val_mm, S):
        """
        Calculate the exchange-repulsion energy between QM and MM valence Slater charge distributions.

        This computes the Pauli repulsion between QM and MM electron densities, which arises
        from the Pauli exclusion principle. The energy is proportional to the overlap between
        the valence charge distributions.

        Parameters
        ----------

        A_exrep_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Exchange-repulsion parameters for QM atoms in Hartree.

        A_exrep_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            Exchange-repulsion parameters for MM atoms in Hartree.

        q_val_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM valence charges in atomic units.

        q_val_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MM valence charges in atomic units.

        S: torch.Tensor (N_BATCH, N_QM_ATOMS, N_MM_ATOMS)
            Overlap integrals between QM and MM valence Slater charge distributions.

        Returns
        -------

        result: torch.Tensor (N_BATCH,)
            Exchange-repulsion energy in Hartree.
        """
        A_exrep = A_exrep_qm[:, :, None] * A_exrep_mm[:, None, :]
        q_prod = 1  # q_val_qm[:, :, None] * q_val_mm[:, None, :]
        return _torch.sum(q_prod * S * A_exrep, dim=(1, 2))
