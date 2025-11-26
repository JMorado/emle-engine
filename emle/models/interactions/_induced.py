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

"""Induced electrostatic (polarization) interaction."""

__author__ = "Joao Morado"
__email__ = "joaomorado@gmail.com"

__all__ = ["InducedElectrostatic"]

import torch as _torch
from ._base import BaseInteraction


class InducedElectrostatic(BaseInteraction):
    """
    Induced electrostatic interaction (polarization).

    Calculates the energy from induced dipoles in the QM region
    due to the MM electric field.
    """

    def __init__(self, emle_base, alpha_mode="species", device=None, dtype=None):
        """
        Constructor.

        Parameters
        ----------

        emle_base: EMLEBase
            Reference to EMLEBase.

        alpha_mode: str
            How atomic polarizabilities are calculated:
                "species": one volume scaling factor per species
                "reference": scaling factors from GPR using reference environments

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for floating point tensors.
        """
        super().__init__(device=device, dtype=dtype)

        self._emle_base = emle_base

        if alpha_mode is None:
            alpha_mode = "species"
        if not isinstance(alpha_mode, str):
            raise TypeError("'alpha_mode' must be of type 'str'")
        alpha_mode = alpha_mode.lower().replace(" ", "")
        if alpha_mode not in ["species", "reference"]:
            raise ValueError("'alpha_mode' must be 'species' or 'reference'")

        self._alpha_mode = alpha_mode

    def forward(self, A_thole, charges_mm, s, mesh_data, mask, gaussian_field=False):
        """
        Calculate induced electrostatic energy.

        Parameters
        ----------

        A_thole: torch.Tensor (N_BATCH, 3*N_QM_ATOMS, 3*N_QM_ATOMS)
            Thole-damped polarizability interaction matrix.

        charges_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MM point charges in atomic units.

        s: torch.Tensor (N_BATCH, N_QM_ATOMS)
            MBIS valence shell widths.

        mesh_data: tuple
            Mesh data from EMLEBase._get_mesh_data.

        mask: torch.Tensor (N_BATCH, N_QM_ATOMS, 1)
            Mask for padded coordinates.

        gaussian_field: bool, optional
            Whether to use Gaussian-smeared MM charges to compute the
            electric field (True) or point charges (False).

        Returns
        -------

        E_induced: torch.Tensor (N_BATCH,)
            Induced electrostatic energy in Hartree.
        """
        # Compute A_thole matrix
        mu_ind = InducedElectrostatic._get_mu_ind(
            A_thole, mesh_data, charges_mm, s, mask, gaussian_field
        )
        vpot_ind = InducedElectrostatic._get_vpot_mu(mu_ind, mesh_data[2])
        return _torch.sum(vpot_ind * charges_mm, dim=1) * 0.5

    @staticmethod
    def _get_mu_ind(A, mesh_data, q, s, mask, gaussian_field):
        """
        Calculate induced atomic dipoles.

        Parameters
        ----------

        A: torch.Tensor (N_BATCH, 3*N_QM_ATOMS, 3*N_QM_ATOMS)
            Thole-damped polarizability interaction matrix.

        mesh_data: tuple
            Mesh data from EMLEBase._get_mesh_data.

        q: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MM point charges in atomic units.

        s: torch.Tensor (N_BATCH, N_QM_ATOMS)
            MBIS valence shell widths.

        mask: torch.Tensor (N_BATCH, N_QM_ATOMS, 1)
            Mask for padded coordinates.

        gaussian_field: bool
            Whether to use Gaussian-smeared MM charges to compute the
            electric field (True) or point charges (False).

        Returns
        -------

        mu_ind: torch.Tensor (N_BATCH, N_QM_ATOMS, 3)
            Induced atomic dipoles in atomic units.
        """
        # r = 1.0 / mesh_data[0]
        if gaussian_field:
            fields = _torch.sum(mesh_data[3] * q[:, None, :, None], dim=2).reshape(
                len(s), -1
            )
        else:
            fields = _torch.sum(mesh_data[2] * q[:, None, :, None], dim=2).reshape(
                len(s), -1
            )
        mu_ind = _torch.linalg.solve(A, fields)
        return mu_ind.reshape((mu_ind.shape[0], -1, 3))

    @staticmethod
    def _get_vpot_mu(mu, T1):
        """
        Calculate electrostatic potential from dipoles.

        Parameters
        ----------

        mu: torch.Tensor (N_BATCH, N_QM_ATOMS, 3)
            Induced atomic dipoles in atomic units.

        T1: torch.Tensor (N_BATCH, N_QM_ATOMS, N_MM_ATOMS, 3)
            First-order interaction tensor.

        Returns
        -------

        vpot: torch.Tensor (N_BATCH, N_MM_ATOMS)
            Electrostatic potential at MM atom positions in atomic units.
        """
        return -_torch.einsum("ijkl,ijl->ik", T1, mu)
