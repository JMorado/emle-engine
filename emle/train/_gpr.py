#######################################################################
# EMLE-Engine: https://github.com/chemle/emle-engine
#
# Copyright: 2023-2025
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

"""Gaussian Process Regression (GPR) for EMLE training."""

import torch as _torch

from ._utils import pad_to_max as _pad_to_max


class GPR:
    """
    Gaussian Process Regression (GPR) for EMLE training.
    """

    @staticmethod
    def _aev_kernel(a, b):
        """
        Computes the kernel between two sets of AEV features.

        Parameters
        ----------

        a: torch.Tensor(N, AEV_DIM)
            First set of AEV features.

        b: torch.Tensor(M, AEV_DIM)
            Second set of AEV features.

        Returns
        -------

        torch.Tensor(N, M)
            Kernel matrix.
        """
        return (a @ b.T) ** 2

    @staticmethod
    def _get_K_mols_ref(K_ivm_allz, zid_mols):
        """
        Get the molecule-reference kernel matrix.

        Parameters
        ----------

        K_ivm_allz: list of torch.Tensor(N_SPECIES, N_Z, N_IVMZ)
            List of molecule-reference kernel matrices.

        zid_mols: torch.Tensor(N_MOLS, MAX_N_ATOMS)
            Species IDs for all molecules.

        Returns
        -------

        result: torch.Tensor(N_MOLS, MAX_N_ATOMS, MAX_N_IVMZ)
            Molecule-reference kernel matrix.
        """
        ivm_max = max([K_ivm_z.shape[1] for K_ivm_z in K_ivm_allz])
        result = _torch.zeros(
            (*zid_mols.shape, ivm_max),
            dtype=K_ivm_allz[0].dtype,
            device=K_ivm_allz[0].device,
        )
        for i, K_ivm_z in enumerate(K_ivm_allz):
            pad = (0, ivm_max - K_ivm_z.shape[1])
            result[zid_mols == i] = _torch.nn.functional.pad(K_ivm_z, pad)

        return result

    @staticmethod
    def _fit_sparse_gpr(y, K_sample_ref, K_ref_ref, sigma):
        """
        Fits GPR reference values to given samples

        y: torch.Tensor (N_SAMPLES,)
            Sample values.

        K_sample_ref: torch.Tensor (N_SAMPLES, MAX_N_REF)
            Sample-reference kernel matrix.

        K_ref_ref: torch.Tensor (MAX_N_REF, MAX_N_REF)
            Reference-reference kernel matrix.

        sigma: float
            GPR sigma value.

        Returns
        -------

        torch.Tensor (MAX_N_REF,)
            Fitted reference values.
        """
        K_ref_ref_sigma = K_ref_ref + sigma**2 * _torch.eye(
            len(K_ref_ref), device=K_ref_ref.device, dtype=K_ref_ref.dtype
        )
        A = _torch.linalg.inv(K_ref_ref_sigma) @ K_sample_ref.T
        B = _torch.linalg.pinv(A.T)

        By = B @ y
        B1 = _torch.sum(B, dim=1)
        y0 = _torch.sum(By) / _torch.sum(B1)
        y_ref = By - y0 * B1

        return y_ref + y0

    @staticmethod
    def get_gpr_kernels(aev_mols, zid_mols, aev_ivm_allz, n_ref):
        """
        Get kernels for performing GPR.

        Parameters
        ----------

        aev_mols : torch.Tensor(N_BATCH, MAX_N_ATOMS, AEV_DIM)
            AEV features for all molecules.

        zid_mols : torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Species IDs for all molecules.

        aev_ivm_allz : torch.Tensor(N_SPECIES, MAX_N_REF, AEV_DIM)
            AEV features for all reference atoms.

        species : torch.Tensor(N_SPECIES)
            Unique species in the dataset.

        n_ref: (N_SPECIES,)
            Number of IVM references for each specie

        Returns
        -------

        K_ref_ref_padded: torch.Tensor(N_SPECIES, MAX_N_REF, MAX_N_REF)
            Reference-reference kernel matrices.

        K_mols_ref: torch.Tensor(N_BATCH, MAX_N_ATOMS, MAX_N_REF)
            Molecules-reference kernel matrices.
        """
        n_species = len(aev_ivm_allz)
        aev_allz = [aev_mols[zid_mols == i] for i in range(n_species)]

        K_ref_ref = [
            GPR._aev_kernel(aev_ivm_z[:n_ref_z, :], aev_ivm_z[:n_ref_z, :])
            for aev_ivm_z, n_ref_z in zip(aev_ivm_allz, n_ref)
        ]

        K_ref_ref_padded = _pad_to_max(K_ref_ref)

        K_ivm_allz = [
            GPR._aev_kernel(aev_z, aev_ivm_z[:n_ref_z, :])
            for aev_z, aev_ivm_z, n_ref_z in zip(aev_allz, aev_ivm_allz, n_ref)
        ]
        K_mols_ref = GPR._get_K_mols_ref(K_ivm_allz, zid_mols)

        return K_ref_ref_padded, K_mols_ref

    @staticmethod
    def fit_atomic_sparse_gpr(values, K_mols_ref, K_ref_ref, zid, sigma, n_ref):
        """
        Fits GPR atomic values to given samples.

        Parameters
        ----------

        values: torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic properties to train on.

        K_mols_ref: torch.Tensor(N_BATCH, MAX_N_ATOMS, MAX_N_REF)
            Molecules-reference kernel matrices.

        K_ref_ref: torch.Tensor(N_SPECIES, MAX_N_REF, MAX_N_REF)
            Reference-reference kernel matrices.

        zid: torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Species IDs.

        sigma: float
            GPR sigma value.

        n_ref: torch.Tensor (N_SPECIES,)
            Number of IVM references for each specie

        Returns
        -------

        result: torch.Tensor(N_SPECIES, MAX_N_REF)
            Fitted atomic values.

        Notes
        -----

        Really only used for s, the rest are predicted by learning.
        """

        n_species, max_n_ref = K_ref_ref.shape[:2]

        result = _torch.zeros((n_species, max_n_ref), dtype=values.dtype)
        for i, n_ref_z in enumerate(n_ref):
            z_mask = zid == i
            result[i, :n_ref_z] = GPR._fit_sparse_gpr(
                values[z_mask],
                K_mols_ref[z_mask][:, :n_ref_z],
                K_ref_ref[i, :n_ref_z, :n_ref_z],
                sigma,
            )
        return result
