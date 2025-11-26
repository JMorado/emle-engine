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

"""Dispersion/Lennard-Jones interaction."""

__author__ = "Joao Morado"
__email__ = "joaomorado@gmail.com"

__all__ = ["Dispersion"]

from ._base import BaseInteraction

import torch as _torch


class Dispersion(BaseInteraction):
    """
    Dispersion/Lennard-Jones energy.

    Calculates dispersion interactions using either Lennard-Jones 12-6
    potential or C6/R^6 with Tang-Toennies damping.
    """

    def __init__(self, emle_base, mode=None, device=None, dtype=None):
        """
        Constructor.

        Parameters
        ----------

        emle_base: EMLEBase
            Reference to EMLEBase.

        mode: str
            Dispersion mode:
                "lj": Lennard-Jones 12-6 potential
                "c6": C6/R^6 with Tang-Toennies damping

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for floating point tensors.
        """
        super().__init__(device=device, dtype=dtype)

        self._emle_base = emle_base

        if mode is None:
            raise ValueError("'mode' must be specified ('lj' or 'c6')")
        if not isinstance(mode, str):
            raise TypeError("'mode' must be of type 'str'")
        mode = mode.lower().replace(" ", "")
        if mode not in ["lj", "c6"]:
            raise ValueError("'mode' must be 'lj' or 'c6'")

        if mode == "lj":
            self.forward = self._forward_lj
        elif mode == "c6":
            self.forward = self._forward_c6
        else:
            raise ValueError(f"Unknown dispersion mode: {mode}")

    def _forward_lj(
        self,
        c6_qm,
        alpha_qm,
        epsilon_mm,
        sigma_mm,
        mesh_data,
        *args,
        **kwargs,
    ):
        """
        Calculate Lennard-Jones dispersion energy.

        Parameters
        ----------

        c6_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            C6 dispersion coefficients for QM atoms.

        alpha_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Isotropic polarizabilities for QM atoms.

        epsilon_mm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Lennard-Jones epsilon parameters for MM atoms.

        sigma_mm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Lennard-Jones sigma parameters for MM atoms.

        mesh_data: tuple
            Mesh data from EMLEBase._get_mesh_data.

        Returns
        -------

        E_disp: torch.Tensor (N_BATCH,)
            Lennard-Jones energy in Hartree.
        """
        c6_qm = 0.5 * c6_qm * alpha_qm
        sigma_qm, epsilon_qm = self._get_lj_parameters(c6_qm, alpha_qm)
        return self._get_lj_energy(
            sigma_qm, epsilon_qm, sigma_mm, epsilon_mm, mesh_data
        )

    @staticmethod
    def _get_lj_parameters(c6, alpha):
        """Calculate LJ sigma and epsilon from c6 and polarizabilities."""
        radius = 2.54 * alpha ** (1.0 / 7.0)
        rmin = 2 * radius
        sigma = rmin / (2 ** (1.0 / 6.0))
        epsilon = c6 / (2 * rmin**6.0)
        return sigma, epsilon

    @staticmethod
    def _get_lj_energy(sigma_qm, epsilon_qm, sigma_mm, epsilon_mm, mesh_data):
        """Calculate Lennard-Jones energy."""
        # Lorentz-Berthelot combining rules
        sigma = 0.5 * (sigma_qm[:, :, None] + sigma_mm[:, None, :])
        epsilon_product = epsilon_qm[:, :, None] * epsilon_mm[:, None, :]
        epsilon = _torch.where(
            epsilon_product > 0, _torch.sqrt(epsilon_product + 1e-16), 0.0
        )
        r_inv, _, _ = mesh_data
        sigma_r_inv_6 = (sigma * r_inv) ** 6
        sigma_r_inv_12 = sigma_r_inv_6 * sigma_r_inv_6
        lj_energy = 4 * epsilon * (sigma_r_inv_12 - sigma_r_inv_6)
        return lj_energy.sum(dim=(1, 2))

    def _forward_c6(
        self,
        c6_qm,
        alpha_qm,
        epsilon_mm,
        sigma_mm,
        mesh_data,
    ):
        """
        Calculate C6/R^6 dispersion energy with Tang-Toennies damping.

        Parameters
        ----------

        c6_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            C6 dispersion coefficients for QM atoms.

        alpha_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Isotropic polarizabilities for QM atoms.

        epsilon_mm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Lennard-Jones epsilon parameters for MM atoms.

        sigma_mm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Lennard-Jones sigma parameters for MM atoms.

        mesh_data: tuple
            Mesh data from EMLEBase._get_mesh_data.

        s_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            MBIS valence widths for QM atoms.

        s_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MBIS valence widths for MM atoms.

        Returns
        -------

        E_disp: torch.Tensor (N_BATCH,)
            Lennard-Jones energy in Hartree.
        """
        c6_qm = 0.5 * c6_qm * alpha_qm
        c6_mm = 4 * epsilon_mm * sigma_mm**6.0
        return self._get_dispersion_energy(c6_qm, c6_mm, alpha_qm, sigma_mm, mesh_data)

    @staticmethod
    def _get_dispersion_energy(c6_qm, c6_mm, s_qm, s_mm, mesh_data):
        """Calculate C6 dispersion energy with Tang-Toennies damping."""

        r_inv, _, _ = mesh_data

        # Tang-Toennies damping function of order 6
        x_damp = 1.0 / ((s_qm[:, :, None] + s_mm[:, None, :]) * r_inv * 0.5)
        f6_damp = 1 - _torch.exp(-x_damp) * (
            1
            + x_damp
            + x_damp**2 / 2
            + x_damp**3 / 6
            + x_damp**4 / 24
            + x_damp**5 / 120
            + x_damp**6 / 720
        )

        # Lorentz-Berthelot combining rules for C6
        c6_product = c6_qm[:, :, None] * c6_mm[:, None, :]
        c6 = _torch.where(c6_product > 0, _torch.sqrt(c6_product + 1e-16), 0.0)
        disp_energy = -c6 * r_inv**6 * f6_damp
        return disp_energy.sum(dim=(1, 2))
