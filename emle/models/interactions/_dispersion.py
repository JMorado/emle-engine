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

        # Assign the appropriate forward method based on mode
        if mode == "lj":
            self.forward = self._forward_lj
        else:  # mode == "c6"
            self.forward = self._forward_c6

    def _forward_lj(
        self,
        species_id,
        aev,
        q_val,
        xyz_qm,
        s,
        charges_mm,
        mesh_data,
        mask,
        idx_mm=None,
        r_data=None,
    ):
        """
        Calculate Lennard-Jones dispersion energy.

        Parameters
        ----------

        species_id: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Species IDs.

        aev: torch.Tensor (N_BATCH, N_QM_ATOMS, N_AEV)
            Atomic environment vectors.

        q_val: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM valence charges.

        xyz_qm: torch.Tensor (N_BATCH, N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        s: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM valence shell widths.

        charges_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MM charges (unused but kept for signature consistency).

        mesh_data: tuple
            Mesh data from EMLEBase._get_mesh_data.

        mask: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Mask for valid atoms.

        idx_mm: torch.Tensor (N_BATCH, N_MM_ATOMS), optional
            Indices for selecting MM parameters from NAGL model.

        r_data: r_data object, optional
            Pre-computed r_data from _get_r_data to avoid redundant computation.

        Returns
        -------

        E_disp: torch.Tensor (N_BATCH,)
            Lennard-Jones energy in Hartree.
        """
        # Compute A_thole and c6
        A_thole = self._emle_base.get_A_thole(
            xyz_qm, s, q_val, species_id, aev, mask, r_data=r_data
        )
        c6 = self._emle_base.get_c6(species_id, aev, mask)
        if c6 is None:
            return q_val.new_zeros(q_val.shape[0])

        # Prepare LJ parameters for MM atoms
        batch_size = c6.shape[0]
        if not hasattr(self._emle_base, "_lj_sigma_mm"):
            return c6.new_zeros(batch_size)

        sigma_mm = self._emle_base._lj_sigma_mm.expand(batch_size, -1)
        epsilon_mm = self._emle_base._lj_eps_mm.expand(batch_size, -1)
        if idx_mm is not None:
            sigma_mm = sigma_mm.gather(1, idx_mm)
            epsilon_mm = epsilon_mm.gather(1, idx_mm)

        alpha_qm = self._get_isotropic_polarizabilities(A_thole)
        c6_qm = c6 * alpha_qm * 0.5
        sigma_qm, epsilon_qm = self._get_lj_parameters(c6_qm, alpha_qm)
        return self._get_lj_energy(
            sigma_qm, epsilon_qm, sigma_mm, epsilon_mm, mesh_data
        )

    @staticmethod
    def _get_isotropic_polarizabilities(A_thole):
        """Compute isotropic polarizabilities from A_thole."""
        batch, dim, _ = A_thole.shape
        n_atoms = dim // 3
        Ainv = _torch.linalg.inv(A_thole)
        Ainv_blocks = Ainv.reshape(batch, n_atoms, 3, n_atoms, 3)
        block_traces = _torch.diagonal(Ainv_blocks, dim1=2, dim2=4)
        block_traces = block_traces.sum(dim=-1)
        per_atom = block_traces.sum(dim=-1) / 3.0
        return per_atom

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
        import torch as _torch

        ANGSTROM_TO_BOHR = 1.8897259886
        KJ_MOL_TO_HARTREE = 1.0 / 2625.5002

        # Fix specific MM parameters (hardcoded for now)
        mask_H = (sigma_mm - 1.8897).abs() < 1e-3
        mask_O = (sigma_mm - 5.9540).abs() < 1e-3
        epsilon_mm = epsilon_mm.clone()
        sigma_mm = sigma_mm.clone()
        epsilon_mm[mask_H] = 0.04102677991569812 * KJ_MOL_TO_HARTREE
        sigma_mm[mask_H] = 2.501348134279251 * ANGSTROM_TO_BOHR
        epsilon_mm[mask_O] = 0.4943618410643088 * KJ_MOL_TO_HARTREE
        sigma_mm[mask_O] = 2.992692406177521 * ANGSTROM_TO_BOHR

        # Lorentz-Berthelot combining rules
        sigma = 0.5 * (sigma_qm[:, :, None] + sigma_mm[:, None, :])
        epsilon_product = epsilon_qm[:, :, None] * epsilon_mm[:, None, :]
        epsilon = _torch.where(
            epsilon_product > 0, _torch.sqrt(epsilon_product + 1e-16), 0.0
        )

        # Get distances
        r_inv, _, _ = mesh_data
        sigma_r_inv_6 = (sigma * r_inv) ** 6
        sigma_r_inv_12 = sigma_r_inv_6 * sigma_r_inv_6
        lj_energy = 4 * epsilon * (sigma_r_inv_12 - sigma_r_inv_6)
        return lj_energy.sum(dim=(1, 2))

    def _forward_c6(
        self,
        species_id,
        aev,
        q_val,
        xyz_qm,
        s,
        charges_mm,
        mesh_data,
        mask,
        idx_mm=None,
        r_data=None,
    ):
        """
        Calculate C6 dispersion energy with Tang-Toennies damping.

        Parameters
        ----------

        species_id: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Species IDs.

        aev: torch.Tensor (N_BATCH, N_QM_ATOMS, N_AEV)
            Atomic environment vectors.

        q_val: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM valence charges.

        xyz_qm: torch.Tensor (N_BATCH, N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        s: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM valence shell widths.

        charges_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MM charges (unused but kept for signature consistency).

        mesh_data: tuple
            Mesh data from EMLEBase._get_mesh_data.

        mask: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Mask for valid atoms.

        idx_mm: torch.Tensor (N_BATCH, N_MM_ATOMS), optional
            Indices for selecting MM parameters from NAGL model.

        r_data: r_data object, optional
            Pre-computed r_data from _get_r_data to avoid redundant computation.

        Returns
        -------

        E_disp: torch.Tensor (N_BATCH,)
            C6 dispersion energy in Hartree.
        """
        # Compute A_thole and c6
        A_thole = self._emle_base.get_A_thole(
            xyz_qm, s, q_val, species_id, aev, mask, r_data=r_data
        )
        c6 = self._emle_base.get_c6(species_id, aev, mask)
        if c6 is None:
            return q_val.new_zeros(q_val.shape[0])

        # Prepare C6 parameters for MM atoms
        batch_size = c6.shape[0]
        if not hasattr(self._emle_base, "_lj_sigma_mm"):
            return c6.new_zeros(batch_size)

        sigma_mm = self._emle_base._lj_sigma_mm.expand(batch_size, -1)
        epsilon_mm = self._emle_base._lj_eps_mm.expand(batch_size, -1)
        s_mm = (
            self._emle_base._s_mm.expand(batch_size, -1)
            if hasattr(self._emle_base, "_s_mm")
            else None
        )

        if idx_mm is not None:
            sigma_mm = sigma_mm.gather(1, idx_mm)
            epsilon_mm = epsilon_mm.gather(1, idx_mm)
            if s_mm is not None:
                s_mm = s_mm.gather(1, idx_mm)

        alpha_qm = self._get_isotropic_polarizabilities(A_thole)
        c6_qm = c6 * alpha_qm * 0.5
        c6_mm = 4 * epsilon_mm * sigma_mm**6
        return self._get_dispersion_energy(c6_qm, c6_mm, s, s_mm, mesh_data)

    @staticmethod
    def _get_dispersion_energy(c6_qm, c6_mm, s_qm, s_mm, mesh_data):
        """Calculate C6 dispersion energy with Tang-Toennies damping."""
        import torch as _torch

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
