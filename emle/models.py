#######################################################################
# EMLE-Engine: https://github.com/chemle/emle-engine
#
# Copyright: 2023-2024
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
# along with EMLE-Engine. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

"""EMLE model implementation."""

__author__ = "Lester Hedges"
__email__ = "lester.hedges@gmail.com"

__all__ = ["EMLE", "ANI2xEMLE"]

import ase as _ase
import numpy as _np
import os as _os
import scipy.io as _scipy_io
import torch as _torch
import torchani as _torchani

from torch import Tensor
from typing import Optional, Tuple

from . import _torchani_patches

# Monkey-patch the TorchANI BuiltInModel and BuiltinEnsemble classes so that
# they call self.aev_computer using args only to allow forward hooks to work
# with TorchScript.
_torchani.models.BuiltinModel = _torchani_patches.BuiltinModel
_torchani.models.BuiltinEnsemble = _torchani_patches.BuiltinEnsemble

try:
    import NNPOps as _NNPOps

    _NNPOps.OptimizedTorchANI = _torchani_patches.OptimizedTorchANI

    _has_nnpops = True
except:
    _has_nnpops = False
    pass


class EMLE(_torch.nn.Module):
    """
    Predicts EMLE energies and gradients allowing QM/MM with ML electrostatic
    embedding.
    """

    def __init__(
        self, alpha_mode="species", device=None, dtype=None, create_aev_calculator=True
    ):
        """
        Constructor

        Parameters
        ----------

        alpha_mode: str
            How atomic polarizabilities are calculated.
                "species":
                    one volume scaling factor is used for each species
                "reference":
                    scaling factors are obtained with GPR using the values learned
                    for each reference environment

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for the models floating point tensors.

        create_aev_calculator: bool
            Whether to create an AEV calculator instance.
        """

        # Call the base class constructor.
        super().__init__()

        from ._utils import _fetch_resources

        # Fetch or update the resources.
        _fetch_resources()

        # Store the expected path to the resources directory.
        resource_dir = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), "resources"
        )

        if not isinstance(alpha_mode, str):
            raise TypeError("'alpha_mode' must be of type 'str'")
        if alpha_mode not in ["species", "reference"]:
            raise ValueError("'alpha_mode' must be 'species' or 'reference'")
        self._alpha_mode = alpha_mode

        # Choose the model based on the alpha_mode.
        if alpha_mode == "species":
            model = _os.path.join(resource_dir, "emle_qm7_aev.mat")
        else:
            model = _os.path.join(resource_dir, "emle_qm7_aev_alphagpr.mat")

        if device is not None:
            if not isinstance(device, _torch.device):
                raise TypeError("'device' must be of type 'torch.device'")
        else:
            device = _torch.get_default_device()

        if dtype is not None:
            if not isinstance(dtype, _torch.dtype):
                raise TypeError("'dtype' must be of type 'torch.dtype'")
        else:
            dtype = _torch.get_default_dtype()

        if not isinstance(create_aev_calculator, bool):
            raise TypeError("'create_aev_calculator' must be of type 'bool'")

        # Create an AEV calculator to perform the feature calculations.
        if create_aev_calculator:
            ani2x = _torchani.models.ANI2x(periodic_table_index=True).to(device)
            self._aev_computer = ani2x.aev_computer
        else:
            self._aev_computer = None

        # Load the model parameters.
        try:
            params = _scipy_io.loadmat(model, squeeze_me=True)
        except:
            raise IOError(f"Unable to load model parameters from: '{model}'")

        # Set the supported species.
        species = [1, 6, 7, 8, 16]

        # Create a map between species and their indices.
        species_map = _np.full(max(species) + 1, fill_value=-1, dtype=_np.int64)
        for i, s in enumerate(species):
            species_map[s] = i

        # Convert to a tensor.
        species_map = _torch.tensor(species_map, dtype=_torch.int64, device=device)

        # Store model parameters as tensors.
        aev_mask = _torch.tensor(params["aev_mask"], dtype=_torch.bool, device=device)
        q_core = _torch.tensor(params["q_core"], dtype=dtype, device=device)
        a_QEq = _torch.tensor(params["a_QEq"], dtype=dtype, device=device)
        a_Thole = _torch.tensor(params["a_Thole"], dtype=dtype, device=device)
        if self._alpha_mode == "species":
            k = _torch.tensor(params["k_Z"], dtype=dtype, device=device)
        else:
            k = _torch.tensor(params["sqrtk_ref"], dtype=dtype, device=device)
        q_total = _torch.tensor(
            params.get("total_charge", 0), dtype=dtype, device=device
        )

        # Extract the reference features.
        ref_features = _torch.tensor(params["ref_aev"], dtype=dtype, device=device)

        # Extract the reference values for the MBIS valence shell widths.
        ref_values_s = _torch.tensor(params["s_ref"], dtype=dtype, device=device)

        # Compute the inverse of the K matrix.
        Kinv = self._get_Kinv(ref_features, 1e-3)

        # Store additional attributes for the MBIS GPR model.
        n_ref = _torch.tensor(params["n_ref"], dtype=_torch.int64, device=device)
        ref_mean_s = _torch.sum(ref_values_s, dim=1) / n_ref
        ref_shifted = ref_values_s - ref_mean_s[:, None]
        c_s = (Kinv @ ref_shifted[:, :, None]).squeeze()

        # Extract the reference values for the electronegativities.
        ref_values_chi = _torch.tensor(params["chi_ref"], dtype=dtype, device=device)

        # Store additional attributes for the electronegativity GPR model.
        ref_mean_chi = _torch.sum(ref_values_chi, dim=1) / n_ref
        ref_shifted = ref_values_chi - ref_mean_chi[:, None]
        c_chi = (Kinv @ ref_shifted[:, :, None]).squeeze()

        # Extract the reference values for the polarizabilities.
        if self._alpha_mode == "reference":
            ref_mean_k = _torch.sum(k, dim=1) / n_ref
            ref_shifted = k - ref_mean_k[:, None]
            c_k = (Kinv @ ref_shifted[:, :, None]).squeeze()
        else:
            ref_mean_k = _torch.empty(0, dtype=dtype, device=device)
            c_k = _torch.empty(0, dtype=dtype, device=device)

        # Store the current device.
        self._device = device

        # Register constants as buffers.
        self.register_buffer("_species_map", species_map)
        self.register_buffer("_aev_mask", aev_mask)
        self.register_buffer("_q_core", q_core)
        self.register_buffer("_a_QEq", a_QEq)
        self.register_buffer("_a_Thole", a_Thole)
        self.register_buffer("_k", k)
        self.register_buffer("_q_total", q_total)
        self.register_buffer("_ref_features", ref_features)
        self.register_buffer("_n_ref", n_ref)
        self.register_buffer("_ref_values_s", ref_values_s)
        self.register_buffer("_ref_values_chi", ref_values_chi)
        self.register_buffer("_ref_mean_s", ref_mean_s)
        self.register_buffer("_ref_mean_chi", ref_mean_chi)
        self.register_buffer("_c_s", c_s)
        self.register_buffer("_c_chi", c_chi)
        self.register_buffer("_ref_mean_k", ref_mean_k)
        self.register_buffer("_c_k", c_k)

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion on the model.
        """
        if self._aev_computer is not None:
            self._aev_computer = self._aev_computer.to(*args, **kwargs)
        self._species_map = self._species_map.to(*args, **kwargs)
        self._aev_mask = self._aev_mask.to(*args, **kwargs)
        self._q_core = self._q_core.to(*args, **kwargs)
        self._a_QEq = self._a_QEq.to(*args, **kwargs)
        self._a_Thole = self._a_Thole.to(*args, **kwargs)
        self._k = self._k.to(*args, **kwargs)
        self._q_total = self._q_total.to(*args, **kwargs)
        self._ref_features = self._ref_features.to(*args, **kwargs)
        self._n_ref = self._n_ref.to(*args, **kwargs)
        self._ref_values_s = self._ref_values_s.to(*args, **kwargs)
        self._ref_values_chi = self._ref_values_chi.to(*args, **kwargs)
        self._ref_mean_s = self._ref_mean_s.to(*args, **kwargs)
        self._ref_mean_chi = self._ref_mean_chi.to(*args, **kwargs)
        self._c_s = self._c_s.to(*args, **kwargs)
        self._c_chi = self._c_chi.to(*args, **kwargs)
        self._ref_mean_k = self._ref_mean_k.to(*args, **kwargs)
        self._c_k = self._c_k.to(*args, **kwargs)

        # Check for a device type in args and update the device attribute.
        for arg in args:
            if isinstance(arg, _torch.device):
                self._device = arg
                break

        return self

    def cuda(self, **kwargs):
        """
        Returns a copy of this model in CUDA memory.
        """
        if self._aev_computer is not None:
            self._aev_computer = self._aev_computer.cuda(**kwargs)
        self._species_map = self._species_map.cuda(**kwargs)
        self._aev_mask = self._aev_mask.cuda(**kwargs)
        self._q_core = self._q_core.cuda(**kwargs)
        self._a_QEq = self._a_QEq.cuda(**kwargs)
        self._a_Thole = self._a_Thole.cuda(**kwargs)
        self._k = self._k.cuda(**kwargs)
        self._q_total = self._q_total.cuda(**kwargs)
        self._ref_features = self._ref_features.cuda(**kwargs)
        self._n_ref = self._n_ref.cuda(**kwargs)
        self._ref_values_s = self._ref_values_s.cuda(**kwargs)
        self._ref_values_chi = self._ref_values_chi.cuda(**kwargs)
        self._ref_mean_s = self._ref_mean_s.cuda(**kwargs)
        self._ref_mean_chi = self._ref_mean_chi.cuda(**kwargs)
        self._c_s = self._c_s.cuda(**kwargs)
        self._c_chi = self._c_chi.cuda(**kwargs)
        self._ref_mean_k = self._ref_mean_k.cuda(**kwargs)
        self._c_k = self._c_k.cuda(**kwargs)

        # Update the device attribute.
        self._device = self._species_map.device

        return self

    def cpu(self, **kwargs):
        """
        Returns a copy of this model in CPU memory.
        """
        if self._aev_computer is not None:
            self._aev_computer = self._aev_computer.cpu(**kwargs)
        self._species_map = self._species_map.cpu(**kwargs)
        self._aev_mask = self._aev_mask.cpu(**kwargs)
        self._q_core = self._q_core.cpu(**kwargs)
        self._a_QEq = self._a_QEq.cpu(**kwargs)
        self._a_Thole = self._a_Thole.cpu(**kwargs)
        self._k = self._k.cpu(**kwargs)
        self._q_total = self._q_total.cpu(**kwargs)
        self._ref_features = self._ref_features.cpu(**kwargs)
        self._n_ref = self._n_ref.cpu(**kwargs)
        self._ref_values_s = self._ref_values_s.cpu(**kwargs)
        self._ref_values_chi = self._ref_values_chi.cpu(**kwargs)
        self._ref_mean_s = self._ref_mean_s.cpu(**kwargs)
        self._ref_mean_chi = self._ref_mean_chi.cpu(**kwargs)
        self._c_s = self._c_s.cpu(**kwargs)
        self._c_chi = self._c_chi.cpu(**kwargs)
        self._ref_mean_k = self._ref_mean_k.cpu(**kwargs)
        self._c_k = self._c_k.cpu(**kwargs)

        # Update the device attribute.
        self._device = self._species_map.device

        return self

    def double(self):
        """
        Returns a copy of this model in float64 precision.
        """
        if self._aev_computer is not None:
            self._aev_computer = self._aev_computer.double()
        self._q_core = self._q_core.double()
        self._a_QEq = self._a_QEq.double()
        self._a_Thole = self._a_Thole.double()
        self._k = self._k.double()
        self._q_total = self._q_total.double()
        self._ref_features = self._ref_features.double()
        self._ref_values_s = self._ref_values_s.double()
        self._ref_values_chi = self._ref_values_chi.double()
        self._ref_mean_s = self._ref_mean_s.double()
        self._ref_mean_chi = self._ref_mean_chi.double()
        self._c_s = self._c_s.double()
        self._c_chi = self._c_chi.double()
        self._ref_mean_k = self._ref_mean_k.double()
        self._c_k = self._c_k.double()
        return self

    def float(self):
        """
        Returns a copy of this model in float32 precision.
        """
        if self._aev_computer is not None:
            self._aev_computer = self._aev_computer.float()
        self._q_core = self._q_core.float()
        self._a_QEq = self._a_QEq.float()
        self._a_Thole = self._a_Thole.float()
        self._k = self._k.float()
        self._q_total = self._q_total.float()
        self._ref_features = self._ref_features.float()
        self._ref_values_s = self._ref_values_s.float()
        self._ref_values_chi = self._ref_values_chi.float()
        self._ref_mean_s = self._ref_mean_s.float()
        self._ref_mean_chi = self._ref_mean_chi.float()
        self._c_s = self._c_s.float()
        self._c_chi = self._c_chi.float()
        self._ref_mean_k = self._ref_mean_k.float()
        self._c_k = self._c_k.float()
        return self

    def forward(self, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
        """
        Computes the static and induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_QM_ATOMS,)
            Atomic numbers of QM atoms.

        charges_mm: torch.Tensor (max_mm_atoms,)
            MM point charges in atomic units.

        xyz_qm: torch.Tensor (N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        xyz_mm: torch.Tensor (N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.

        Returns
        -------

        result: torch.Tensor (2,)
            The static and induced EMLE energy components in Hartree.
        """

        # If there are no point charges, return zeros.
        if len(xyz_mm) == 0:
            return _torch.zeros(2, dtype=xyz_qm.dtype, device=xyz_qm.device)

        # Convert the atomic numbers to species IDs.
        species_id = self._species_map[atomic_numbers]

        # Reshape the IDs.
        zid = species_id.unsqueeze(0)

        # Reshape the atomic positions.
        xyz = xyz_qm.unsqueeze(0)

        # Compute the AEVs.
        aev = self._aev_computer((zid, xyz))[1][0][:, self._aev_mask]
        aev = aev / _torch.linalg.norm(aev, ord=2, dim=1, keepdim=True)

        # Compute the MBIS valence shell widths.
        s = self._gpr(aev, self._ref_mean_s, self._c_s, species_id)

        # Compute the electronegativities.
        chi = self._gpr(aev, self._ref_mean_chi, self._c_chi, species_id)

        # Convert coordinates to Bohr.
        ANGSTROM_TO_BOHR = 1.8897261258369282
        xyz_qm_bohr = xyz_qm * ANGSTROM_TO_BOHR
        xyz_mm_bohr = xyz_mm * ANGSTROM_TO_BOHR

        # Compute the static energy.
        q_core = self._q_core[species_id]
        if self._alpha_mode == "species":
            k = self._k[species_id]
        else:
            k = self._gpr(aev, self._ref_mean_k, self._c_k, species_id)
        r_data = self._get_r_data(xyz_qm_bohr)
        mesh_data = self._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s)
        q = self._get_q(r_data, s, chi)
        q_val = q - q_core
        mu_ind = self._get_mu_ind(r_data, mesh_data, charges_mm, s, q_val, k)
        vpot_q_core = self._get_vpot_q(q_core, mesh_data[0])
        vpot_q_val = self._get_vpot_q(q_val, mesh_data[1])
        vpot_static = vpot_q_core + vpot_q_val
        E_static = _torch.sum(vpot_static @ charges_mm)

        # Compute the induced energy.
        vpot_ind = self._get_vpot_mu(mu_ind, mesh_data[2])
        E_ind = _torch.sum(vpot_ind @ charges_mm) * 0.5

        return _torch.stack([E_static, E_ind])

    @classmethod
    def _get_Kinv(cls, ref_features, sigma):
        """
        Internal function to compute the inverse of the K matrix for GPR.

        Parameters
        ----------

        ref_features: torch.Tensor (N_Z, MAX_N_REF, N_FEAT)
            The basis feature vectors for each species.

        sigma: float
            The uncertainty of the observations (regularizer).

        Returns
        -------

        result: torch.Tensor (MAX_N_REF, MAX_N_REF)
            The inverse of the K matrix.
        """
        n = ref_features.shape[1]
        K = (ref_features @ ref_features.swapaxes(1, 2)) ** 2
        return _torch.linalg.inv(
            K + sigma**2 * _torch.eye(n, dtype=ref_features.dtype, device=K.device)
        )

    def _gpr(self, mol_features, ref_mean, c, zid):
        """
        Internal method to predict a property using Gaussian Process Regression.

        Parameters
        ----------

        mol_features: torch.Tensor (N_ATOMS, N_FEAT)
            The feature vectors for each atom.

        ref_mean: torch.Tensor (N_Z,)
            The mean of the reference values for each species.

        c: torch.Tensor (N_Z, MAX_N_REF)
            The coefficients of the GPR model.

        zid: torch.Tensor (N_ATOMS,)
            The species identity value of each atom.

        Returns
        -------

        result: torch.Tensor (N_ATOMS)
            The values of the predicted property for each atom.
        """

        result = _torch.zeros(
            len(zid), dtype=mol_features.dtype, device=mol_features.device
        )
        for i in range(len(self._n_ref)):
            n_ref = self._n_ref[i]
            ref_features_z = self._ref_features[i, :n_ref]
            mol_features_z = mol_features[zid == i, :, None]

            K_mol_ref2 = (ref_features_z @ mol_features_z) ** 2
            K_mol_ref2 = K_mol_ref2.reshape(K_mol_ref2.shape[:-1])
            result[zid == i] = K_mol_ref2 @ c[i, :n_ref] + ref_mean[i]

        return result

    def _get_q(self, r_data: Tuple[Tensor, Tensor, Tensor, Tensor], s, chi):
        """
        Internal method that predicts MBIS charges
        (Eq. 16 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        r_data: r_data object (output of self._get_r_data)

        s: torch.Tensor (N_ATOMS,)
            MBIS valence shell widths.

        chi: torch.Tensor (N_ATOMS,)
            Electronegativities.

        Returns
        -------

        result: torch.Tensor (N_ATOMS,)
            Predicted MBIS charges.
        """
        A = self._get_A_QEq(r_data, s)
        b = _torch.hstack([-chi, self._q_total])
        return _torch.linalg.solve(A, b)[:-1]

    def _get_A_QEq(self, r_data: Tuple[Tensor, Tensor, Tensor, Tensor], s):
        """
        Internal method, generates A matrix for charge prediction
        (Eq. 16 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        r_data: r_data object (output of self._get_r_data)

        s: torch.Tensor (N_ATOMS,)
            MBIS valence shell widths.

        Returns
        -------

        result: torch.Tensor (N_ATOMS + 1, N_ATOMS + 1)
        """
        s_gauss = s * self._a_QEq
        s2 = s_gauss**2
        s_mat = _torch.sqrt(s2[:, None] + s2[None, :])

        device = r_data[0].device
        dtype = r_data[0].dtype

        A = self._get_T0_gaussian(r_data[1], r_data[0], s_mat)

        new_diag = _torch.ones_like(A.diagonal(), dtype=dtype, device=device) * (
            1.0
            / (
                s_gauss
                * _torch.sqrt(_torch.tensor([_torch.pi], dtype=dtype, device=device))
            )
        )
        mask = _torch.diag(_torch.ones_like(new_diag, dtype=dtype, device=device))
        A = mask * _torch.diag(new_diag) + (1.0 - mask) * A

        # Store the dimensions of A.
        x, y = A.shape

        # Create an tensor of ones with one more row and column than A.
        B = _torch.ones(x + 1, y + 1, dtype=dtype, device=device)

        # Copy A into B.
        B[:x, :y] = A

        # Set the final entry on the diagonal to zero.
        B[-1, -1] = 0.0

        return B

    def _get_mu_ind(
        self,
        r_data: Tuple[Tensor, Tensor, Tensor, Tensor],
        mesh_data: Tuple[Tensor, Tensor, Tensor],
        q,
        s,
        q_val,
        k,
    ):
        """
        Internal method, calculates induced atomic dipoles
        (Eq. 20 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        r_data: r_data object (output of self._get_r_data)

        mesh_data: mesh_data object (output of self._get_mesh_data)

        q: torch.Tensor (N_MM_ATOMS,)
            MM point charges.

        s: torch.Tensor (N_QM_ATOMS,)
            MBIS valence shell widths.

        q_val: torch.Tensor (N_QM_ATOMS,)
            MBIS valence charges.

        k: torch.Tensor (N_Z)
            Scaling factors for polarizabilities.

        Returns
        -------

        result: torch.Tensor (N_ATOMS, 3)
            Array of induced dipoles
        """
        A = self._get_A_thole(r_data, s, q_val, k)

        r = 1.0 / mesh_data[0]
        f1 = self._get_f1_slater(r, s[:, None] * 2.0)
        fields = _torch.sum(mesh_data[2] * f1[:, :, None] * q[:, None], dim=1).flatten()

        mu_ind = _torch.linalg.solve(A, fields)
        return mu_ind.reshape((-1, 3))

    def _get_A_thole(self, r_data: Tuple[Tensor, Tensor, Tensor, Tensor], s, q_val, k):
        """
        Internal method, generates A matrix for induced dipoles prediction
        (Eq. 20 in 10.1021/acs.jctc.2c00914)

        Parameters
        ----------

        r_data: r_data object (output of self._get_r_data)

        s: torch.Tensor (N_ATOMS,)
            MBIS valence shell widths.

        q_val: torch.Tensor (N_ATOMS,)
            MBIS charges.

        k: torch.Tensor (N_Z)
            Scaling factors for polarizabilities.

        Returns
        -------

        result: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
            The A matrix for induced dipoles prediction.
        """
        v = -60 * q_val * s**3
        alpha = v * k

        alphap = alpha * self._a_Thole
        alphap_mat = alphap[:, None] * alphap[None, :]

        au3 = r_data[0] ** 3 / _torch.sqrt(alphap_mat)
        au31 = au3.repeat_interleave(3, dim=1)
        au32 = au31.repeat_interleave(3, dim=0)

        A = -self._get_T2_thole(r_data[2], r_data[3], au32)

        new_diag = 1.0 / alpha.repeat_interleave(3)
        mask = _torch.diag(_torch.ones_like(new_diag, dtype=A.dtype, device=A.device))
        A = mask * _torch.diag(new_diag) + (1.0 - mask) * A

        return A

    @staticmethod
    def _get_vpot_q(q, T0):
        """
        Internal method to calculate the electrostatic potential.

        Parameters
        ----------

        q: torch.Tensor (N_MM_ATOMS,)
            MM point charges.

        T0: torch.Tensor (N_QM_ATOMS, max_mm_atoms)
            T0 tensor for QM atoms over MM atom positions.

        Returns
        -------

        result: torch.Tensor (max_mm_atoms)
            Electrostatic potential over MM atoms.
        """
        return _torch.sum(T0 * q[:, None], dim=0)

    @staticmethod
    def _get_vpot_mu(mu, T1):
        """
        Internal method to calculate the electrostatic potential generated
        by atomic dipoles.

        Parameters
        ----------

        mu: torch.Tensor (N_ATOMS, 3)
            Atomic dipoles.

        T1: torch.Tensor (N_ATOMS, max_mm_atoms, 3)
            T1 tensor for QM atoms over MM atom positions.

        Returns
        -------

        result: torch.Tensor (max_mm_atoms)
            Electrostatic potential over MM atoms.
        """
        return -_torch.tensordot(T1, mu, ((0, 2), (0, 1)))

    @classmethod
    def _get_r_data(cls, xyz):
        """
        Internal method to calculate r_data object.

        Parameters
        ----------

        xyz: torch.Tensor (N_ATOMS, 3)
            Atomic positions.

        Returns
        -------

        result: r_data object
        """
        n_atoms = len(xyz)

        rr_mat = xyz[:, None, :] - xyz[None, :, :]
        r_mat = _torch.cdist(xyz, xyz)
        r_inv = _torch.where(r_mat == 0.0, 0.0, 1.0 / r_mat)

        r_inv1 = r_inv.repeat_interleave(3, dim=1)
        r_inv2 = r_inv1.repeat_interleave(3, dim=0)

        # Get a stacked matrix of outer products over the rr_mat tensors.
        outer = _torch.einsum("bik,bij->bjik", rr_mat, rr_mat).reshape(
            (n_atoms * 3, n_atoms * 3)
        )

        id2 = _torch.tile(
            _torch.tile(
                _torch.eye(3, dtype=xyz.dtype, device=xyz.device).T, (1, n_atoms)
            ).T,
            (1, n_atoms),
        )

        t01 = r_inv
        t21 = -id2 * r_inv2**3
        t22 = 3 * outer * r_inv2**5

        return (r_mat, t01, t21, t22)

    @classmethod
    def _get_mesh_data(cls, xyz, xyz_mesh, s):
        """
        Internal method, calculates mesh_data object.

        Parameters
        ----------

        xyz: torch.Tensor (N_ATOMS, 3)
            Atomic positions.

        xyz_mesh: torch.Tensor (max_mm_atoms, 3)
            MM positions.

        s: torch.Tensor (N_ATOMS,)
            MBIS valence widths.
        """
        rr = xyz_mesh[None, :, :] - xyz[:, None, :]
        r = _torch.linalg.norm(rr, ord=2, dim=2)

        return (1.0 / r, cls._get_T0_slater(r, s[:, None]), -rr / r[:, :, None] ** 3)

    @classmethod
    def _get_f1_slater(cls, r, s):
        """
        Internal method, calculates damping factors for Slater densities.

        Parameters
        ----------

        r: torch.Tensor (N_ATOMS, max_mm_atoms)
            Distances from QM to MM atoms.

        s: torch.Tensor (N_ATOMS,)
            MBIS valence widths.

        Returns
        -------

        result: torch.Tensor (N_ATOMS, max_mm_atoms)
        """
        return (
            cls._get_T0_slater(r, s) * r
            - _torch.exp(-r / s) / s * (0.5 + r / (s * 2)) * r
        )

    @staticmethod
    def _get_T0_slater(r, s):
        """
        Internal method, calculates T0 tensor for Slater densities.

        Parameters
        ----------

        r: torch.Tensor (N_ATOMS, max_mm_atoms)
            Distances from QM to MM atoms.

        s: torch.Tensor (N_ATOMS,)
            MBIS valence widths.

        Returns
        -------

        results: torch.Tensor (N_ATOMS, max_mm_atoms)
        """
        return (1 - (1 + r / (s * 2)) * _torch.exp(-r / s)) / r

    @staticmethod
    def _get_T0_gaussian(t01, r, s_mat):
        """
        Internal method, calculates T0 tensor for Gaussian densities (for QEq).

        Parameters
        ----------

        t01: torch.Tensor (N_ATOMS, N_ATOMS)
            T0 tensor for QM atoms.

        r: torch.Tensor (N_ATOMS, N_ATOMS)
            Distance matrix for QM atoms.

        s_mat: torch.Tensor (N_ATOMS, N_ATOMS)
            Matrix of Gaussian sigmas for QM atoms.

        Returns
        -------

        results: torch.Tensor (N_ATOMS, N_ATOMS)
        """
        return t01 * _torch.erf(
            r
            / (
                s_mat
                * _torch.sqrt(_torch.tensor([2.0], dtype=r.dtype, device=r.device))
            )
        )

    @classmethod
    def _get_T2_thole(cls, tr21, tr22, au3):
        """
        Internal method, calculates T2 tensor with Thole damping.

        Parameters
        ----------

        tr21: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
            r_data[2]

        tr21: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
            r_data[3]

        au3: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
            Scaled distance matrix (see _get_A_thole).

        Returns
        -------

        result: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
        """
        return cls._lambda3(au3) * tr21 + cls._lambda5(au3) * tr22

    @staticmethod
    def _lambda3(au3):
        """
        Internal method, calculates r^3 component of T2 tensor with Thole
        damping.

        Parameters
        ----------

        au3: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
            Scaled distance matrix (see _get_A_thole).

        Returns
        -------

        result: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
        """
        return 1 - _torch.exp(-au3)

    @staticmethod
    def _lambda5(au3):
        """
        Internal method, calculates r^5 component of T2 tensor with Thole
        damping.

        Parameters
        ----------

        au3: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
            Scaled distance matrix (see _get_A_thole).

        Returns
        -------

        result: torch.Tensor (N_ATOMS * 3, N_ATOMS * 3)
        """
        return 1 - (1 + au3) * _torch.exp(-au3)


class ANI2xEMLE(EMLE):
    def __init__(
        self,
        alpha_mode="species",
        model_index=None,
        ani2x_model=None,
        atomic_numbers=None,
        device=None,
        dtype=None,
    ):
        """
        Constructor

        Parameters
        ----------

        alpha_mode: str
            How atomic polarizabilities are calculated.
                "species":
                    one volume scaling factor is used for each species
                "reference":
                    scaling factors are obtained with GPR using the values learned
                    for each reference environment

        model_index: int
            The index of the model to use. If None, then the full 8 model
            ensemble will be used.

        ani2x_model: torchani.models.ANI2x, NNPOPS.OptimizedTorchANI
            An existing ANI2x model to use. If None, a new ANI2x model will be
            created. If using an OptimizedTorchANI model, please ensure that
            the ANI2x model from which it derived was created using
            periodic_table_index=True.

        atomic_numbers: torch.Tensor (N_ATOMS,)
            List of atomic numbers to use in the ANI2x model. If specified,
            and NNPOps is available, then an optimised version of ANI2x will
            be used.

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for the models floating point tensors.
        """
        if model_index is not None:
            if not isinstance(model_index, int):
                raise TypeError("'model_index' must be of type 'int'")
            if model_index < 0 or model_index > 7:
                raise ValueError("'model_index' must be in the range [0, 7]")
        self._model_index = model_index

        if device is not None:
            if not isinstance(device, _torch.device):
                raise TypeError("'device' must be of type 'torch.device'")
        else:
            device = _torch.get_default_device()

        if dtype is not None:
            if not isinstance(dtype, _torch.dtype):
                raise TypeError("'dtype' must be of type 'torch.dtype'")
        else:
            dtype = _torch.get_default_dtype()

        if atomic_numbers is not None:
            if not isinstance(atomic_numbers, _torch.Tensor):
                raise TypeError("'atomic_numbers' must be of type 'torch.Tensor'")
            # Check that they are integers.
            if atomic_numbers.dtype != _torch.int64:
                raise ValueError("'atomic_numbers' must be of dtype 'torch.int64'")
            self._atomic_numbers = atomic_numbers.to(device)
        else:
            self._atomic_numbers = None

        # Call the base class constructor.
        super().__init__(
            alpha_mode=alpha_mode,
            device=device,
            dtype=dtype,
            create_aev_calculator=False,
        )

        if ani2x_model is not None:
            # Add the base ANI2x model and ensemble.
            allowed_types = [
                _torchani.models.BuiltinModel,
                _torchani.models.BuiltinEnsemble,
            ]

            # Add the optimised model if NNPOps is available.
            try:
                allowed_types.append(_NNPOps.OptimizedTorchANI)
            except:
                pass

            if not isinstance(ani2x_model, tuple(allowed_types)):
                raise TypeError(f"'ani2x_model' must be of type {allowed_types}")

            if (
                isinstance(
                    ani2x_model,
                    (_torchani.models.BuiltinModel, _torchani.models.BuiltinEnsemble),
                )
                and not ani2x_model.periodic_table_index
            ):
                raise ValueError(
                    "The ANI2x model must be created with 'periodic_table_index=True'"
                )

            self._ani2x = ani2x_model.to(device)
            if dtype == _torch.float64:
                self._ani2x = self._ani2x.double()
        else:
            # Create the ANI2x model.
            self._ani2x = _torchani.models.ANI2x(
                periodic_table_index=True, model_index=model_index
            ).to(device)
            if dtype == _torch.float64:
                self._ani2x = self._ani2x.double()

            # Optimise the ANI2x model if atomic_numbers are specified.
            if atomic_numbers is not None:
                try:
                    species = atomic_numbers.reshape(1, *atomic_numbers.shape)
                    self._ani2x = _NNPOps.OptimizedTorchANI(self._ani2x, species).to(
                        device
                    )
                except:
                    pass

        # Assign a tensor attribute that can be used for assigning the AEVs.
        self._ani2x.aev_computer._aev = _torch.empty(0, device=device)

        # Hook the forward pass of the ANI2x model to get the AEV features.
        # Note that this currently requires a patched versions of TorchANI and NNPOps.
        if _has_nnpops and isinstance(self._ani2x, _NNPOps.OptimizedTorchANI):

            def hook(
                module,
                input: Tuple[Tuple[Tensor, Tensor], Optional[Tensor], Optional[Tensor]],
                output: Tuple[Tensor, Tensor],
            ):
                module._aev = output[1][0]

        else:

            def hook(
                module,
                input: Tuple[Tuple[Tensor, Tensor], Optional[Tensor], Optional[Tensor]],
                output: _torchani.aev.SpeciesAEV,
            ):
                module._aev = output[1][0]

        # Register the hook.
        self._aev_hook = self._ani2x.aev_computer.register_forward_hook(hook)

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion on the model.
        """
        module = super(ANI2xEMLE, self).to(*args, **kwargs)
        module._ani2x = module._ani2x.to(*args, **kwargs)
        return module

    def cpu(self, **kwargs):
        """
        Returns a copy of this model in CPU memory.
        """
        module = super(ANI2xEMLE, self).cpu(**kwargs)
        module._ani2x = module._ani2x.cpu(**kwargs)
        if self._atomic_numbers is not None:
            module._atomic_numbers = module._atomic_numbers.cpu(**kwargs)
        return module

    def cuda(self, **kwargs):
        """
        Returns a copy of this model in CUDA memory.
        """
        module = super(ANI2xEMLE, self).cuda(**kwargs)
        module._ani2x = module._ani2x.cuda(**kwargs)
        if self._atomic_numbers is not None:
            module._atomic_numbers = module._atomic_numbers.cuda(**kwargs)
        return module

    def double(self):
        """
        Returns a copy of this model in float64 precision.
        """
        module = super(ANI2xEMLE, self).double()
        module._ani2x = module._ani2x.double()
        return module

    def float(self):
        """
        Returns a copy of this model in float32 precision.
        """
        module = super(ANI2xEMLE, self).float()
        # Using .float() or .to(torch.float32) is broken for ANI2x models.
        module._ani2x = _torchani.models.ANI2x(
            periodic_table_index=True, model_index=self._model_index
        ).to(self._device)
        # Optimise the ANI2x model if atomic_numbers were specified.
        if self._atomic_numbers is not None:
            try:
                from NNPOps import OptimizedTorchANI as _OptimizedTorchANI

                species = self._atomic_numbers.reshape(1, *atomic_numbers.shape)
                self._ani2x = _OptimizedTorchANI(self._ani2x, species).to(self._device)
            except:
                pass

        return module

    def forward(self, atomic_numbers, charges_mm, xyz_qm, xyz_mm):
        """
        Computes the static and induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_QM_ATOMS,)
            Atomic numbers of QM atoms.

        charges_mm: torch.Tensor (max_mm_atoms,)
            MM point charges in atomic units.

        xyz_qm: torch.Tensor (N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        xyz_mm: torch.Tensor (N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.

        Returns
        -------

        result: torch.Tensor (3,)
            The ANI2x and static and induced EMLE energy components in Hartree.
        """

        # Convert the atomic numbers to species IDs.
        species_id = self._species_map[atomic_numbers]

        # Reshape the IDs.
        zid = species_id.unsqueeze(0)

        # Reshape the atomic numbers.
        atomic_numbers = atomic_numbers.unsqueeze(0)

        # Reshape the coordinates,
        xyz = xyz_qm.unsqueeze(0)

        # Get the in vacuo energy.
        E_vac = self._ani2x((atomic_numbers, xyz)).energies[0]

        # If there are no point charges, return the in vacuo energy and zeros
        # for the static and induced terms.
        if len(xyz_mm) == 0:
            zero = _torch.tensor(0.0, dtype=xyz_qm.dtype, device=xyz_qm.device)
            return _torch.stack([E_vac, zero, zero])

        # Get the AEVs computer by the forward hook and normalise.
        aev = self._ani2x.aev_computer._aev[:, self._aev_mask]
        aev = aev / _torch.linalg.norm(aev, ord=2, dim=1, keepdim=True)

        # Compute the MBIS valence shell widths.
        s = self._gpr(aev, self._ref_mean_s, self._c_s, species_id)

        # Compute the electronegativities.
        chi = self._gpr(aev, self._ref_mean_chi, self._c_chi, species_id)

        # Convert coordinates to Bohr.
        ANGSTROM_TO_BOHR = 1.8897261258369282
        xyz_qm_bohr = xyz_qm * ANGSTROM_TO_BOHR
        xyz_mm_bohr = xyz_mm * ANGSTROM_TO_BOHR

        # Compute the static energy.
        q_core = self._q_core[species_id]
        if self._alpha_mode == "species":
            k = self._k[species_id]
        else:
            k = self._gpr(aev, self._ref_mean_k, self._c_k, species_id)
        r_data = self._get_r_data(xyz_qm_bohr)
        mesh_data = self._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s)
        q = self._get_q(r_data, s, chi)
        q_val = q - q_core
        mu_ind = self._get_mu_ind(r_data, mesh_data, charges_mm, s, q_val, k)
        vpot_q_core = self._get_vpot_q(q_core, mesh_data[0])
        vpot_q_val = self._get_vpot_q(q_val, mesh_data[1])
        vpot_static = vpot_q_core + vpot_q_val
        E_static = _torch.sum(vpot_static @ charges_mm)

        # Compute the induced energy.
        vpot_ind = self._get_vpot_mu(mu_ind, mesh_data[2])
        E_ind = _torch.sum(vpot_ind @ charges_mm) * 0.5

        return _torch.stack([E_vac, E_static, E_ind])