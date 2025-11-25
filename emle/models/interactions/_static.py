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

"""Static electrostatic interaction."""

__author__ = "Joao Morado"
__email__ = "joaomorado@gmail.com"

__all__ = ["StaticElectrostatic"]

import torch as _torch
from ._base import BaseInteraction
from .._emle_base import EMLEBase


class StaticElectrostatic(BaseInteraction):
    """
    Static electrostatic interaction between QM and MM regions.

    Calculates the electrostatic energy from core and valence charges,
    with optional charge penetration correction.
    """

    def __init__(
        self,
        emle_base,
        cp_mode=None,
        method="electrostatic",
        q_core_mm=None,
        device=None,
        dtype=None,
    ):
        """
        Constructor.

        Parameters
        ----------

        emle_base: EMLEBase
            Reference to EMLEBase.

        cp_mode: str, optional
            Charge penetration mode:
                "gaussian": Use Gaussian charge distributions for net total charges (core + valence). Only available when method is "electrostatic" or "nonpol".
                "slater": Use Slater valence shells + point core charges. Only available when method is "electrostatic" or "nonpol".
                None: No charge penetration correction.

        method: str
            The embedding method ("electrostatic", "mechanical", "nonpol", "mm").
            Determines how charges are handled.

        q_core_mm: torch.Tensor, optional
            MM charges for the QM region (required if method="mm").

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for floating point tensors.
        """
        super().__init__(device=device, dtype=dtype)

        self._emle_base = emle_base

        if cp_mode is not None:
            if not isinstance(cp_mode, str):
                raise TypeError("'cp_mode' must be of type 'str'")
            cp_mode = cp_mode.lower().replace(" ", "")
            if cp_mode not in ["gaussian", "slater"]:
                raise ValueError("'cp_mode' must be 'gaussian' or 'slater'")

        self._cp_mode = cp_mode
        self._method = method
        self._q_core_mm = q_core_mm

        if cp_mode is None or self._method in ["mm", "mechanical"]:
            self._forward_impl = self._forward_emle
        elif cp_mode == "slater":
            self._forward_impl = self._forward_slater
        elif cp_mode == "gaussian":
            if self._emle_base.a_Gauss is None or self._emle_base.a_Gauss.numel() == 0:
                raise ValueError(
                    "Gaussian CP correction requires 'a_Gauss' parameter. Make sure it is provided in the model."
                )
            if self._emle_base._s_mm is None or self._emle_base._s_mm.numel() == 0:
                raise ValueError(
                    "Gaussian CP correction requires MM valence shell widths 's_mm'. Make sure they have been supplied in a NAGL model."
                )
            self._s_mm = self._emle_base._s_mm
            self._forward_impl = self._forward_gaussian
        else:
            raise NotImplementedError(f"CP mode '{cp_mode}' not implemented.")

    def forward(self, *args, **kwargs):
        """
        Calculate static electrostatic energy.

        This method dispatches to the appropriate implementation based on cp_mode.
        See _forward_emle, _forward_slater, and _forward_gaussian for specific signatures.
        """
        return self._forward_impl(*args, **kwargs)

    def _forward_emle(
        self, q_core, q_val, charges_mm, mesh_data, s_qm=None, idx_mm=None
    ):
        """
        Calculate static energy with point charges for MM (no charge penetration).

        Parameters
        ----------

        q_core: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM core charges.

        q_val: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM valence charges.

        charges_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MM charges in atomic units.

        mesh_data: tuple
            Mesh data from EMLEBase._get_mesh_data.

        Returns
        -------

        E_static: torch.Tensor (N_BATCH,)
            Static electrostatic energy in Hartree.
        """
        if self._method == "mm":
            batch_size = q_core.shape[0]
            q_core = self._q_core_mm.expand(batch_size, -1)
            q_val = _torch.zeros_like(
                q_core, dtype=charges_mm.dtype, device=self._device
            )
        elif self._method == "mechanical":
            q_core = q_core + q_val
            q_val = _torch.zeros_like(
                q_core, dtype=charges_mm.dtype, device=self._device
            )

        # Calculate electrostatic potential due to core and valence charges.
        vpot_q_core = StaticElectrostatic._get_vpot_q(q_core, mesh_data[0])
        vpot_q_val = StaticElectrostatic._get_vpot_q(q_val, mesh_data[1])
        vpot_static = vpot_q_core + vpot_q_val

        return _torch.sum(vpot_static * charges_mm, dim=1)

    def _forward_gaussian(
        self,
        q_core,
        q_val,
        charges_mm,
        mesh_data,
        s_qm,
        idx_mm,
    ):
        """
        Calculate static energy with Gaussian charge penetration correction.

        Parameters
        ----------

        q_core: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM core charges.

        q_val: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM valence charges.

        charges_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MM charges in atomic units.

        mesh_data: tuple
            Mesh data from EMLEBase._get_mesh_data.

        s_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            MBIS valence shell widths for QM.

        idx_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            Indices for selecting MM parameters from NAGL model.

        Returns
        -------

        E_static: torch.Tensor (N_BATCH,)
            Static electrostatic energy with Gaussian CP correction in Hartree.
        """
        # Convert MBIS widths to Gaussian widths.
        sigma_qm = s_qm * self._emle_base.a_Gauss
        sigma_mm = self._emle_base._s_mm.gather(1, idx_mm) * self._emle_base.a_Gauss
        s_mat = _torch.sqrt(sigma_qm[:, :, None] ** 2 + sigma_mm[:, None, :] ** 2)

        # Get gaussian T0 tensor.
        r = 1.0 / mesh_data[0]
        q_qm = q_core + q_val
        T0_cp = EMLEBase._get_T0_gaussian(1.0, r, s_mat)

        # Calculate electrostatic potential due to QM charges with Gaussian CP.
        vpot_static = _torch.sum(T0_cp * q_qm[:, :, None], dim=1)
        return _torch.sum(vpot_static * charges_mm, dim=1)

    def _forward_slater(
        self, emle_base, q_core, q_val, charges_mm, mesh_data, idx_mm=None
    ):
        """
        Calculate static energy with Slater charge penetration correction.

        Parameters
        ----------

        emle_base: EMLEBase
            Reference to EMLEBase for accessing helper methods.

        q_core: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM core charges.

        q_val: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM valence charges.

        charges_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MM charges in atomic units.

        mesh_data: tuple
            Mesh data from EMLEBase._get_mesh_data.

        idx_mm: torch.Tensor (N_BATCH, N_MM_ATOMS), optional
            Indices for selecting MM parameters from NAGL model.

        Returns
        -------

        E_static: torch.Tensor (N_BATCH,)
            Static electrostatic energy with Slater CP correction in Hartree.
        """
        pass

    @staticmethod
    def _get_static_energy_slater(
        q_core_qm, q_val_qm, q_core_mm, q_val_mm, mesh_data, s_qm, s_mm
    ):
        """
        Internal method to calculate the static electrostatic energy using a charge model
        with point core charges and Slater valence charge distributions.

        This implements the charge penetration correction by using diffuse Slater-type
        valence distributions instead of point charges, which accounts for the finite
        size of electron clouds at short range.

        Parameters
        ----------

        q_core_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM core charges in atomic units (point charges).

        q_val_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM valence charges in atomic units (Slater distributions).

        q_core_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MM core charges in atomic units (point charges).

        q_val_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MM valence charges in atomic units (Slater distributions).

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

        Returns
        -------

        result: torch.Tensor (N_BATCH,)
            Static electrostatic energy in Hartree.

        Notes
        -----

        The total energy is decomposed into four terms:
        - E_core_core: core-core point charge interactions
        - E_val_core: valence Slater - core point interactions
        - E_core_val: core point - valence Slater interactions
        - E_val_val: valence Slater - valence Slater interactions
        """
        r_inv, T0_slater_qm_mm = mesh_data[0], mesh_data[1]
        r = _torch.where(r_inv > 0, 1.0 / (r_inv + 1e-16), 0.0)
        T0_slater_mm_qm = EMLEBase._get_T0_slater(r.permute(0, 2, 1), s_mm[:, :, None])

        # Mask out interactions involving padded atoms
        mask_s = (s_qm != 0)[:, :, None] & (s_mm != 0)[:, None, :]
        T0_slater_qm_mm = T0_slater_qm_mm * mask_s
        T0_slater_mm_qm = T0_slater_mm_qm * mask_s.permute(0, 2, 1)
        r_inv = r_inv * mask_s

        # Calculate electrostatic energy components
        E_core_core = _torch.sum(
            StaticElectrostatic._get_vpot_q(q_core_qm, r_inv) * q_core_mm, dim=1
        )
        E_val_core = _torch.sum(
            StaticElectrostatic._get_vpot_q(q_val_qm, T0_slater_qm_mm) * q_core_mm,
            dim=1,
        )
        E_core_val = _torch.sum(
            StaticElectrostatic._get_vpot_q(q_val_mm, T0_slater_mm_qm) * q_core_qm,
            dim=1,
        )
        E_val_val = StaticElectrostatic._get_valence_repulsion(
            q_val_qm, q_val_mm, r, s_qm, s_mm
        )
        total = E_val_core + E_core_val + E_val_val + E_core_core
        return total

    @staticmethod
    def _get_valence_repulsion(q_val_qm, q_val_mm, r, s_qm, s_mm):
        """
        Internal method to calculate the valence-valence repulsion energy for Slater charge distributions.

        This computes the Coulomb interaction between two Slater-type charge distributions,
        accounting for charge penetration effects. The interaction differs from point charges
        at short range due to the finite extent of the Slater distributions.

        Parameters
        ----------

        q_val_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            QM valence charges in atomic units.

        q_val_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            MM valence charges in atomic units.

        r: torch.Tensor (N_BATCH, N_QM_ATOMS, N_MM_ATOMS)
            Distance matrix between QM and MM atoms in Bohr.

        s_qm: torch.Tensor (N_BATCH, N_QM_ATOMS)
            MBIS valence shell widths for QM atoms in Bohr.

        s_mm: torch.Tensor (N_BATCH, N_MM_ATOMS)
            Slater widths for MM atoms in Bohr.

        Returns
        -------

        result: torch.Tensor (N_BATCH,)
            Valence-valence repulsion energy in Hartree.
        """

        def f_diff(si, sj, r):
            """Interaction between Slater functions with different widths."""
            s_diff = si**2 - sj**2
            f = (
                (si**4 / (s_diff**2 + 1e-16))
                * (1 + r / (2 * si + 1e-16) - 2 * sj**2 / (s_diff + 1e-16))
                * _torch.exp(-r / (si + 1e-16))
            )
            return f

        def f_same(si, sj, r):
            """Interaction between Slater functions with the same widths."""
            x = r / (si + 1e-16)
            delta = sj - si
            exp_x = _torch.exp(-x)
            term1 = (1 + 11 / 16.0 * x + 3 / 16.0 * x**2 + 1 / 48.0 * x**3) * exp_x
            term2 = (
                (delta / (96.0 * si**2 + 1e-16))
                * (15.0 + 15 * x + 6.0 * x**2 + x**3)
                * exp_x
            )
            term3 = (
                (delta**2 / (320.0 * si**3 + 1e-16))
                * (20.0 + 20.0 * x + 5.0 * x**2 - 5.0 / 3 * x**3 - x**4)
                * exp_x
            )
            f = (1 - term1) / (r + 1e-16) - term2 - term3
            return f

        s_mask = (s_qm > 0)[:, :, None] & (s_mm > 0)[:, None, :]
        s_qm = s_qm[:, :, None]
        s_mm = s_mm[:, None, :]
        cp_corr_diff = (1 - f_diff(s_qm, s_mm, r) - f_diff(s_mm, s_qm, r)) / (r + 1e-16)
        cp_corr_same = f_same(s_qm, s_mm, r)
        cp_corr = _torch.where(
            _torch.abs(s_qm - s_mm) < 1e-2, cp_corr_same, cp_corr_diff
        )
        return _torch.sum(
            q_val_qm[:, :, None] * q_val_mm[:, None, :] * cp_corr * s_mask, dim=(1, 2)
        )

    @staticmethod
    def _get_vpot_q(q, T0):
        """
        Internal method to calculate the electrostatic potential.

        Parameters
        ----------

        q: torch.Tensor (N_BATCH, MAX_QM_ATOMS,)
            QM charges (q_core or q_val).

        T0: torch.Tensor (N_BATCH, MAX_QM_ATOMS, MAX_MM_ATOMS)
            T0 tensor for QM atoms over MM atom positions.

        Returns
        -------

        result: torch.Tensor (N_BATCH, MAX_MM_ATOMS)
            Electrostatic potential over MM atoms.
        """
        return _torch.sum(T0 * q[:, :, None], dim=1)
