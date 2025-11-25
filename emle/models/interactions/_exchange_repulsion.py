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

    def forward(self, q_val, charges_mm, mesh_data, s, idx_mm=None):
        """
        Calculate exchange-repulsion energy.

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

        E_exrep: torch.Tensor (N_BATCH,)
            Exchange-repulsion energy in Hartree.
        """
        # TODO: Implement proper parameter retrieval
        # For now, return zero
        return _torch.zeros(q_val.shape[0], dtype=q_val.dtype, device=q_val.device)

    @staticmethod
    def get_exchange_repulsion_energy(
        A_exrep_qm,
        A_exrep_mm,
        q_val_qm,
        q_val_mm,
        mesh_data,
        s_qm,
        s_mm,
        S=None,
    ):
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
            Exchange-repulsion energy in Hartree.

        Notes
        -----

        The exchange-repulsion energy is computed as:
        E_exrep = sum_{i,j} A_{ij} * S_{ij}

        where A_{ij} = A_exrep_qm[i] * A_exrep_mm[j] and S_{ij} is the overlap integral
        between Slater functions centered on atoms i and j.
        """
        mask_s = (s_qm > 0)[:, :, None] & (s_mm > 0)[:, None, :]
        A_exrep = A_exrep_qm[:, :, None] * A_exrep_mm[:, None, :]

        r_inv, _, _ = mesh_data
        r = _torch.where(mesh_data[0] > 0, 1.0 / (mesh_data[0] + 1e-16), 0.0)
        S = ExchangeRepulsion._get_slater_overlap(s_qm, s_mm, r) if S is None else S
        q_prod = q_val_qm[:, :, None] * q_val_mm[:, None, :]
        return _torch.sum(S * A_exrep * mask_s, dim=(1, 2))

    @staticmethod
    def _get_slater_overlap(s_qm, s_mm, r):
        """
        Internal method to compute the overlap integral between two Slater-type charge distributions.

        The overlap integral measures the spatial overlap between two exponentially decaying
        charge distributions (Slater functions). This is used to compute short-range corrections
        such as exchange-repulsion and penetration effects.

        Parameters
        ----------

        s_qm: torch.Tensor (N_BATCH, N_QM_ATOMS) or (N_BATCH, N_QM_ATOMS, 1)
            MBIS valence shell widths for QM atoms in Bohr.

        s_mm: torch.Tensor (N_BATCH, N_MM_ATOMS) or (N_BATCH, 1, N_MM_ATOMS)
            Slater widths for MM atoms in Bohr.

        r: torch.Tensor (N_BATCH, N_QM_ATOMS, N_MM_ATOMS)
            Distance matrix between QM and MM atoms in Bohr.

        Returns
        -------

        result: torch.Tensor (N_BATCH, N_QM_ATOMS, N_MM_ATOMS)
            Overlap integrals S_{ij} between Slater functions i and j.

        Notes
        -----

        The Slater function for atom i has the form:
        ρ_i(r) = (1/(8πs_i³)) exp(-r/s_i)

        The overlap integral is:
        S_{ij} = ∫ ρ_i(r - R_i) ρ_j(r - R_j) dr

        Different analytical formulas are used depending on whether s_i ≈ s_j:
        - h_same: for |s_i - s_j| < 1e-2 (numerically stable)
        - h_diff: for different widths
        """

        def h_diff(si, sj, r):
            """Overlap between Slater functions with different widths."""
            s_diff = sj**2 - si**2
            term1 = 4 * si**2 * sj**2 / (s_diff**3 + 1e-16)
            term2 = (si * r) / (s_diff**2 + 1e-16)
            return (term1 + term2) * _torch.exp(-r / (si + 1e-16))

        def h_same(si, sj, r):
            """Overlap between Slater functions with the same widths."""
            s_diff = sj - si
            x = r / (si + 1e-16)
            exp_x = _torch.exp(-x)
            term1 = 1 / (192 * _torch.pi * si**3 + 1e-16) * (3 + 3 * x + x**2) * exp_x
            term2 = (
                s_diff / (384 * si**4 + 1e-16) * (-9 - 9 * x - 2 * x**2 + x**3) * exp_x
            )
            term3 = (
                s_diff**2
                / (3840 * si**5 + 1e-16)
                * (90 + 90 * x + 5 * x**2 - 25 * x**3 + 3 * x**4)
                * exp_x
            )
            return term1 + term2 + term3

        # Broadcast QM x MM
        s_qm_exp = s_qm.unsqueeze(-1).expand(-1, -1, s_mm.size(1))
        s_mm_exp = s_mm.unsqueeze(1).expand(-1, s_qm.size(1), -1)
        mask_s = (s_qm_exp > 0) & (s_mm_exp > 0)

        # Initialize result tensor
        S = _torch.zeros_like(r)

        # Masks
        equal_mask = _torch.abs(s_qm_exp - s_mm_exp) < 1e-2
        diff_mask = ~equal_mask

        # Compute only on the elements where needed
        if equal_mask.any():
            si_eq = s_qm_exp[equal_mask]
            sj_eq = s_mm_exp[equal_mask]
            r_eq = r[equal_mask]
            S[equal_mask] = h_same(si_eq, sj_eq, r_eq)

        if diff_mask.any():
            si_diff = s_qm_exp[diff_mask]
            sj_diff = s_mm_exp[diff_mask]
            r_diff = r[diff_mask]
            S[diff_mask] = (
                h_diff(si_diff, sj_diff, r_diff) + h_diff(sj_diff, si_diff, r_diff)
            ) / (8 * _torch.pi * r_diff + 1e-16)

        return S * mask_s
