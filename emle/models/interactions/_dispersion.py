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
import numpy as _np


class Dispersion(BaseInteraction):
    """
    Dispersion/Lennard-Jones energy.

    Calculates dispersion interactions using either Lennard-Jones 12-6
    potential or C6/R^6 with Tang-Toennies damping.
    """

    def __init__(
        self,
        emle_base,
        mode=None,
        method="electrostatic",
        epsilon_mm_qm=None,
        sigma_mm_qm=None,
        r_switch=None,
        r_cutoff=None,
        n_particles=None,
        device=None,
        dtype=None,
    ):
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

        method: str
            The embedding method ("electrostatic", "mechanical", "nonpol", "mm").
            Determines how LJ parameters are handled.

        epsilon_mm_qm: torch.Tensor, optional
            MM Lennard-Jones epsilon parameters for the QM region (required if method="mm").

        sigma_mm_qm: torch.Tensor, optional
            MM Lennard-Jones sigma parameters for the QM region (required if method="mm").

        r_switch: float, optional
            Switching distance in Angstrom. If provided with r_cutoff,
            applies OpenMM-style switching function for r_switch < r < r_cutoff.
            Only used for "lj" mode.

        r_cutoff: float, optional
            Cutoff distance in Angstrom. Required if r_switch is provided.
            Only used for "lj" mode.

        n_particles: int, optional
            Number of particles in the system (for long-range corrections).

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

        # Validate switching parameters
        if r_switch is not None or r_cutoff is not None:
            if r_switch is None or r_cutoff is None:
                raise ValueError(
                    "Both 'r_switch' and 'r_cutoff' must be provided together"
                )
            if r_switch >= r_cutoff:
                raise ValueError("'r_switch' must be less than 'r_cutoff'")
            if mode != "lj":
                raise ValueError("Switching function is only supported for 'lj' mode")
        
        if n_particles is not None:
            if not isinstance(n_particles, int) or n_particles <= 0:
                raise ValueError("'n_particles' must be a positive integer")
            self._n_particles = n_particles
        else:
            self._n_particles = None

        self._r_switch = r_switch
        self._r_cutoff = r_cutoff
        self._method = method
        self._epsilon_mm_qm = epsilon_mm_qm
        self._sigma_mm_qm = sigma_mm_qm

        if mode == "lj":
            self._forward_impl = self._forward_lj
            self._rvdw_prefactor = _torch.nn.Parameter(
                _torch.tensor(2.54, device=self._device, dtype=self._dtype),
            )
            self._rvdw_exp = _torch.nn.Parameter(
                _torch.tensor(1.0 / 7.0, device=self._device, dtype=self._dtype),
            )
        elif mode == "c6":
            self._forward_impl = self._forward_c6
        else:
            raise ValueError(f"Unknown dispersion mode: {mode}")

    def _forward_lj(
        self,
        c6_qm,
        alpha_qm,
        epsilon_mm,
        sigma_mm,
        mesh_data,
        s_qm,
        s_mm,
        sigma_scale,
        cell,
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
        if self._method == "mm":
            batch_size = epsilon_mm.shape[0]
            sigma_qm = self._sigma_mm_qm.expand(batch_size, -1)
            epsilon_qm = self._epsilon_mm_qm.expand(batch_size, -1)
        else:
            c6_qm = 0.5 * c6_qm * alpha_qm
            sigma_qm, epsilon_qm = self._get_lj_parameters(c6_qm, alpha_qm, sigma_scale)
        return self._get_lj_energy(
            sigma_qm,
            epsilon_qm,
            sigma_mm,
            epsilon_mm,
            mesh_data,
            r_switch=self._r_switch,
            r_cutoff=self._r_cutoff,
            cell=cell,
            n_particles=self._n_particles,
        )

    def _get_lj_parameters(self, c6, alpha, sigma_scale):
        """
        Calculate Lennard-Jones sigma and epsilon parameters from C6 and polarizabilities.

        Uses empirical relationships to convert C6 dispersion coefficients and
        polarizabilities into Lennard-Jones parameters.

        Parameters
        ----------

        c6: torch.Tensor (N_BATCH, N_QM_ATOMS)
            C6 dispersion coefficients.

        alpha: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Isotropic polarizabilities.

        Returns
        -------

        sigma: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Lennard-Jones sigma parameter (distance at which potential is zero).

        epsilon: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Lennard-Jones epsilon parameter (depth of potential well).
        """
        # Get sigma scaling factors
        radius = self._rvdw_prefactor * alpha**self._rvdw_exp
        rmin = 2 * radius * sigma_scale
        sigma = rmin / (2 ** (1.0 / 6.0))
        epsilon = c6 / (2 * rmin**6.0)
        return sigma, epsilon

    @staticmethod
    def _apply_switching_function(r_inv, r_switch, r_cutoff):
        """
        Calculate OpenMM-style switching function.

        Implements the switching function:
        S = 1 - 6x^5 + 15x^4 - 10x^3
        where x = (r - r_switch) / (r_cutoff - r_switch)

        Parameters
        ----------

        r_inv: torch.Tensor (N_BATCH, N_QM_ATOMS, N_MM_ATOMS)
            Inverse distances between QM and MM atoms.

        r_switch: float
            Switching distance in Angstrom.

        r_cutoff: float
            Cutoff distance in Angstrom.

        Returns
        -------

        switch: torch.Tensor (N_BATCH, N_QM_ATOMS, N_MM_ATOMS)
            Switching function values in range [0, 1].
        """
        r = 1.0 / r_inv
        x = (r - r_switch) / (r_cutoff - r_switch)
        # Clamp x to [0, 1] range to handle r < r_switch and r > r_cutoff
        x = _torch.clamp(x, 0.0, 1.0)
        switch = 1.0 - 6.0 * x**5 + 15.0 * x**4 - 10.0 * x**3
        return switch

    @staticmethod
    def _get_lj_energy(
        sigma_qm,
        epsilon_qm,
        sigma_mm,
        epsilon_mm,
        mesh_data,
        r_switch=None,
        r_cutoff=None,
        cell=None,
        n_particles=None,
    ):
        """
        Calculate Lennard-Jones 12-6 energy between QM and MM atoms.

        Uses the standard Lennard-Jones potential:
        E_LJ = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]

        Lorentz-Berthelot combining rules are applied to mix QM and MM parameters:
        - sigma_mix = (sigma_i + sigma_j) / 2
        - epsilon_mix = sqrt(epsilon_i * epsilon_j)

        Optionally applies a switching function to smoothly reduce the energy to zero
        at the cutoff distance (OpenMM-style switching).

        Parameters
        ----------

        sigma_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Lennard-Jones sigma parameters for QM atoms.

        epsilon_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            Lennard-Jones epsilon parameters for QM atoms.

        sigma_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            Lennard-Jones sigma parameters for MM atoms.

        epsilon_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            Lennard-Jones epsilon parameters for MM atoms.

        mesh_data: tuple
            Mesh data tuple (r_inv, T0, T1) from EMLEBase._get_mesh_data.

        r_switch: float, optional
            Switching distance. If provided with r_cutoff, applies switching function.

        r_cutoff: float, optional
            Cutoff distance. Required if r_switch is provided.

        cell: torch.Tensor, optional
            Simulation cell tensor for LJ long-range correction.

        n_particles: int, optional
            Number of particles in the system (for long-range corrections).

        Returns
        -------

        E_lj: torch.Tensor (N_BATCH,)
            Total Lennard-Jones energy in Hartree.
        """
        # Lorentz-Berthelot combining rules
        sigma = 0.5 * (sigma_qm[:, :, None] + sigma_mm[:, None, :])
        epsilon_product = epsilon_qm[:, :, None] * epsilon_mm[:, None, :]
        epsilon = _torch.where(
            epsilon_product > 0, _torch.sqrt(epsilon_product + 1e-16), 0.0
        )
        r_inv, *_ = mesh_data
        sigma_r_inv_6 = (sigma * r_inv) ** 6
        sigma_r_inv_12 = sigma_r_inv_6 * sigma_r_inv_6
        lj_energy = 4 * epsilon * (sigma_r_inv_12 - sigma_r_inv_6)

        #if r_switch is not None and r_cutoff is not None:
        #    switch = Dispersion._apply_switching_function(r_inv, r_switch, r_cutoff)
        #    lj_energy = lj_energy * switch

        if cell is not None and r_cutoff is not None and n_particles is not None:
            lr_corr = Dispersion._lj_long_range_correction(
                epsilon,
                sigma,
                r_cutoff * 1.889726124993589, 
                cell,
                n_particles=n_particles,
            ) 

        print("LR_CORR:", lr_corr * 2625.5)
        return lj_energy.sum(dim=(1, 2))

    @staticmethod
    def _lj_long_range_correction(
        epsilon,
        sigma,
        r_cutoff,
        cell,
        n_particles=None,
    ):
        """
        Calculate long-range correction to Lennard-Jones energy.

        Parameters
        ----------
        epsilon: torch.Tensor (N_BATCH, N_QM_ATOMS, N_MM_ATOMS)
            Lennard-Jones epsilon parameters for QM-MM atom pairs.
        sigma: torch.Tensor (N_BATCH, N_QM_ATOMS, N_MM_ATOMS)
            Lennard-Jones sigma parameters for QM-MM atom pairs.
        r_cutoff: float
            Cutoff distance in Bohr.
        volume: torch.Tensor (N_BATCH,)
            Volume of the simulation box in Bohr^3.

        Returns
        -------
        E_lj_lrc: torch.Tensor (N_BATCH,)
            Long-range correction to Lennard-Jones energy in Hartree.
        """
        sigma6 = sigma**6
        sigma12 = sigma6 * sigma6

        # 8 * pi * (N_QM * N_MM) / V
        cell = cell * 1.889726124993589
        volume = _torch.det(cell).abs()
        _, n_qm, _ = epsilon.shape
        n_mm = n_particles - n_qm
        rho_mm = n_mm / volume
        pre_factor = 8 * _np.pi * n_qm * rho_mm * 2.

        # LJ long-range correction
        lj_lrc = pre_factor * (
            (_torch.mean(epsilon * sigma12, dim=(1, 2)) / (9 * r_cutoff**9))
            - (_torch.mean(epsilon * sigma6, dim=(1, 2)) / (3 * r_cutoff**3))
        )
        return lj_lrc

    def _forward_c6(
        self,
        c6_qm,
        alpha_qm,
        epsilon_mm,
        sigma_mm,
        mesh_data,
        s_qm,
        s_mm,
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
        return self._get_dispersion_energy(c6_qm, c6_mm, s_qm, s_mm, mesh_data)

    @staticmethod
    def _get_dispersion_energy(c6_qm, c6_mm, s_qm, s_mm, mesh_data):
        """
        Calculate C6/R^6 dispersion energy with Tang-Toennies damping.

        Implements the dispersion interaction:
        E_disp = -C6 / R^6 * f_damp(R)

        where f_damp is the Tang-Toennies damping function of order 6:
        f_6(x) = 1 - exp(-x) * sum_{k=0}^{6} (x^k / k!)

        The damping prevents unphysical behavior at short range by using
        MBIS valence widths to define the damping length scale.

        Parameters
        ----------

        c6_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            C6 dispersion coefficients for QM atoms.

        c6_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            C6 dispersion coefficients for MM atoms.

        s_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            MBIS valence shell widths for QM atoms (used for damping).

        s_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MBIS valence shell widths for MM atoms (used for damping).

        mesh_data: tuple
            Mesh data tuple (r_inv, T0, T1) from EMLEBase._get_mesh_data.

        Returns
        -------

        E_disp: torch.Tensor (N_BATCH,)
            Total C6 dispersion energy in Hartree.
        """
        r_inv, *_ = mesh_data

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
