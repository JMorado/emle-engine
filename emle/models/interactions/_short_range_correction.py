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

"""Short-range correction interaction."""

__author__ = "Joao Morado"
__email__ = "joaomorado@gmail.com"

__all__ = ["ShortRangeCorrection"]

import torch as _torch
from ._base import BaseInteraction


class ShortRangeCorrection(BaseInteraction):
    """
    Short-range correction energy.

    Corrects for short-range errors in the electrostatic embedding.
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

    def forward(self, q_val, charges_mm, mesh_data, s, idx_mm=None):
        """
        Calculate short-range correction energy.

        Parameters
        ----------

        q_val: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM valence charges.

        charges_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MM charges in atomic units.

        mesh_data: tuple
            Mesh data from EMLEBase._get_mesh_data.

        s: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM valence shell widths.

        idx_mm: torch.Tensor (N_BATCH, N_MM_ATOMS), optional
            Indices for selecting MM parameters from NAGL model.

        Returns
        -------

        E_sr_corr: torch.Tensor (N_BATCH,)
            Short-range correction energy in Hartree.
        """
        # TODO: Implement proper parameter retrieval
        # For now, return zero
        return _torch.zeros(q_val.shape[0], dtype=q_val.dtype, device=q_val.device)

    @staticmethod
    def get_sr_corr_energy(
        A_short_range_corr_qm,
        A_short_range_corr_mm,
        q_val_qm,
        q_val_mm,
        mesh_data,
        s_qm,
        s_mm,
        S=None,
    ):
        """
        Calculate the short-range correction energy between QM and MM valence Slater charge distributions.

        This corrects for short-range errors in the electrostatic embedding arising from the
        point charge approximation. The functional form is identical to the exchange-repulsion
        energy but with opposite sign and different parameters.

        Parameters
        ----------

        A_short_range_corr_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Short-range correction parameters for QM atoms in Hartree.

        A_short_range_corr_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            Short-range correction parameters for MM atoms in Hartree.

        q_val_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM valence charges in atomic units.

        q_val_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MM valence charges in atomic units.

        mesh_data: tuple of torch.Tensor
            Mesh data object from EMLEBase._get_mesh_data.
            Contains (r_inv, T0_slater, T1) where:
                r_inv: (N_BATCH, N_QM_ATOMS, N_MM_ATOMS) - inverse distances
                T0_slater: (N_BATCH, N_QM_ATOMS, N_MM_ATOMS) - T0 tensor for Slater
                T1: (N_BATCH, N_QM_ATOMS, N_MM_ATOMS, 3) - T1 tensor for dipoles

        s_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            MBIS valence shell widths for QM atoms in Bohr.

        s_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            Slater widths for MM atoms in Bohr.

        S: torch.Tensor (N_BATCH, N_QM_ATOMS, N_MM_ATOMS), optional
            Precomputed Slater overlap integrals.
            If None, they will be computed.

        Returns
        -------

        result: torch.Tensor (N_BATCH,)
            Short-range correction energy in Hartree.

        Notes
        -----

        The short-range correction energy is computed as:
        E_sr_corr = -sum_{i,j} A_{ij} * S_{ij}

        where A_{ij} = A_sr_corr_qm[i] * A_sr_corr_mm[j] and S_{ij} is the overlap integral
        between Slater functions centered on atoms i and j. The negative sign distinguishes
        this from the exchange-repulsion term.
        """
        mask_s = (s_qm > 0)[:, :, None] & (s_mm > 0)[:, None, :]
        A_short_range_corr = (
            A_short_range_corr_qm[:, :, None] * A_short_range_corr_mm[:, None, :]
        )
        r = _torch.where(mesh_data[0] > 0, 1.0 / (mesh_data[0] + 1e-16), 0.0)
        # Import to access _get_slater_overlap
        from ._exchange_repulsion import ExchangeRepulsion

        S = ExchangeRepulsion._get_slater_overlap(s_qm, s_mm, r) if S is None else S
        q_prod = q_val_qm[:, :, None] * q_val_mm[:, None, :]
        return -_torch.sum(S * A_short_range_corr * mask_s, dim=(1, 2))
