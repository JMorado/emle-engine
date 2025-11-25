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

import os as _os
import sys as _sys

import numpy as _np
import torch as _torch
from loguru import logger as _logger
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import TensorDataset as _TensorDataset

from ..models import EMLE as _EMLE
from ..models import EMLEAEVComputer as _EMLEAEVComputer
from ..models import EMLEBase as _EMLEBase
from ._gpr import GPR as _GPR
from ._ivm import IVM as _IVM
from ._loss import AtomicPropertyLoss as _AtomicPropertyLoss
from ._loss import ExchangeRepulsionLoss as _ExchangeRepulsionLoss
from ._loss import QEqLoss as _QEqLoss
from ._loss import DispersionCoefficientLoss as _DispersionCoefficientLoss
from ._loss import ShortRangeCorrectionLoss as _ShortRangeCorrectionLoss
from ._loss import SLoss as _SLoss
from ._loss import TholeLoss as _TholeLoss
from ._utils import mean_by_z as _mean_by_z
from ._utils import pad_to_max as _pad_to_max


class EMLETrainer:
    def __init__(
        self,
        emle_base=_EMLEBase,
        qeq_loss=_QEqLoss,
        thole_loss=_TholeLoss,
        dispersion_loss=_DispersionCoefficientLoss,
        log_level=None,
        log_file=None,
    ):
        if emle_base is not _EMLEBase:
            raise TypeError("emle_base must be a reference to EMLEBase")
        self._emle_base = emle_base

        if qeq_loss is not _QEqLoss:
            raise TypeError("qeq_loss must be a reference to QEqLoss")
        self._qeq_loss = qeq_loss

        if thole_loss is not _TholeLoss:
            raise TypeError("thole_loss must be a reference to TholeLoss")
        self._thole_loss = thole_loss

        if dispersion_loss is not _DispersionCoefficientLoss:
            raise TypeError(
                "dispersion_loss must be a reference to DispersionCoefficientLoss"
            )
        self._dispersion_loss = dispersion_loss

        # First handle the logger.
        if log_level is None:
            log_level = "INFO"
        else:
            if not isinstance(log_level, str):
                raise TypeError("'log_level' must be of type 'str'")

            # Delete whitespace and convert to upper case.
            log_level = log_level.upper().replace(" ", "")

            # Validate the log level.
            if log_level not in _logger._core.levels.keys():
                raise ValueError(
                    f"Unsupported logging level '{log_level}'. Options are: {', '.join(_logger._core.levels.keys())}"
                )
        self._log_level = log_level

        # Validate the log file.

        if log_file is not None:
            if not isinstance(log_file, str):
                raise TypeError("'log_file' must be of type 'str'")

            # Try to create the directory.
            dirname = _os.path.dirname(log_file)
            if dirname != "":
                try:
                    _os.makedirs(dirname, exist_ok=True)
                except:
                    raise IOError(
                        f"Unable to create directory for log file: {log_file}"
                    )
            self._log_file = _os.path.abspath(log_file)
        else:
            self._log_file = _sys.stdout

        # Update the logger.
        _logger.remove()
        _logger.add(self._log_file, level=self._log_level)

    @staticmethod
    def _get_zid_mapping(species):
        """
        Generate the species ID mapping.

        Parameters
        ----------

        species: torch.Tensor(N_SPECIES)
            Species IDs.

        Returns
        -------

        mapping: torch.Tensor
            Species ID mapping.
        """
        zid_mapping = (
            _torch.ones(max(species) + 1, dtype=_torch.int, device=species.device) * -1
        )
        for i, z in enumerate(species):
            zid_mapping[z] = i
        return zid_mapping

    @staticmethod
    def _write_model_to_file(emle_model, model_filename):
        """
        Write the trained model to a file.

        Parameters
        ----------

        emle_model: dict
            Trained EMLE model.

        model_filename: str
            Filename to save the trained model.
        """
        import scipy.io

        # Deatch the tensors, convert to numpy arrays and save the model.
        emle_model = {
            k: v.cpu().detach().numpy() if isinstance(v, _torch.Tensor) else v
            for k, v in emle_model.items()
            if v is not None
        }
        scipy.io.savemat(model_filename, emle_model)

    @staticmethod
    def _train_s(s, zid, aev_mols, aev_ivm_allz, sigma):
        """
        Train the s model.

        Parameters
        ----------

        s: torch.Tensor(N_BATCH, N_ATOMS)
            Atomic widths.

        zid: torch.Tensor(N_BATCH, N_ATOMS)
            Species IDs.

        aev_mols: torch.Tensor(N_BATCH, N_ATOMS, N_AEV)
            Atomic environment vectors.

        aev_ivm_allz: torch.Tensor(N_BATCH, N_ATOMS, N_AEV)
            Atomic environment vectors for all species.

        sigma: float
            GPR sigma value.

        Returns
        -------

        torch.Tensor(N_BATCH, N_ATOMS)
            Atomic widths.
        """
        n_ref = _torch.tensor([_.shape[0] for _ in aev_ivm_allz], device=s.device)
        K_ref_ref_padded, K_mols_ref = _GPR.get_gpr_kernels(
            aev_mols, zid, aev_ivm_allz, n_ref
        )

        ref_values_s = _GPR.fit_atomic_sparse_gpr(
            s, K_mols_ref, K_ref_ref_padded, zid, sigma, n_ref
        )

        return _pad_to_max(ref_values_s)

    @staticmethod
    def _train_model(
        loss_class,
        opt_param_names,
        lr,
        epochs,
        emle_base,
        print_every=10,
        *args,
        **kwargs,
    ):
        """
        Train a model.

        Parameters
        ----------

        loss_class: class
            Loss class.

        opt_param_names: list of str
            List of parameter names to optimize.

        lr: float
            Learning rate.

        epochs: int
            Number of training epochs.

        emle_base: EMLEBase
            EMLEBase instance.

        print_every: int
            How often to print training progress

        Returns
        -------

        model
            Trained model.
        """

        def _train_loop(
            loss_instance, optimizer, epochs, print_every=10, *args, **kwargs
        ):
            """
            Perform the training loop.

            Parameters
            ----------

            loss_instance: nn.Module
                Loss instance.

            optimizer: torch.optim.Optimizer
                Optimizer.

            epochs: int
                Number of training epochs.

            print_every: int
                How often to print training progress

            args: list
                Positional arguments to pass to the forward method.

            kwargs: dict
                Keyword arguments to pass to the forward method.

            Returns
            -------

            loss
                Forward loss.
            """
            for epoch in range(epochs):
                loss_instance.train()
                optimizer.zero_grad()
                loss, rmse, max_error = loss_instance(*args, **kwargs)
                loss.backward(retain_graph=True)
                optimizer.step()
                if (epoch + 1) % print_every == 0:
                    _logger.info(
                        f"Epoch {epoch+1}: Loss ={loss.item():9.4f}    "
                        f"RMSE ={rmse.item():9.4f}    "
                        f"Max Error ={max_error.item():9.4f}"
                    )

            return loss

        model = loss_class(emle_base)
        opt_parameters = [
            param
            for name, param in model.named_parameters()
            if name.split(".")[1] in opt_param_names
        ]

        optimizer = _torch.optim.Adam(opt_parameters, lr=lr)
        _train_loop(model, optimizer, epochs, print_every, *args, **kwargs)
        return model

    def train(
        self,
        z,
        xyz,
        s,
        q_core,
        q_val,
        alpha,
        c6=None,
        train_mask=None,
        alpha_mode="reference",
        sigma=1e-3,
        ivm_thr=0.05,
        epochs=100,
        lr_qeq=0.05,
        lr_thole=0.05,
        lr_sqrtk=0.05,
        lr_c6=0.01,
        print_every=10,
        computer_n_species=None,
        computer_zid_map=None,
        model_filename="emle_model.mat",
        emle_model=None,
        plot_data_filename=None,
        device=_torch.device("cuda"),
        dtype=_torch.float64,
    ):
        """
        Train an EMLE model.

        Parameters
        ----------

        z: numpy.array, List[numpy.array], torch.Tensor, List[torch.Tensor] (N_BATCH, N_ATOMS)
            Atomic numbers.

        xyz: numpy.array, List[numpy.array], torch.Tensor, List[torch.Tensor] (N_BATCH, N_ATOM, 3)
            Atomic coordinates.

        s: numpy.array, List[numpy.array], torch.Tensor, List[torch.Tensor] (N_BATCH, N_ATOMS)
            Atomic widths.

        q_core: numpy.array, List[numpy.array], torch.Tensor, List[torch.Tensor] (N_BATCH, N_ATOMS)
            Atomic core charges.

        q_val: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Atomic valence charges.

        alpha: array or tensor or list of tensor/arrays of shape (N_BATCH, 3, 3)
            Atomic polarizabilities.

        c6: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS), optional
            Atomic dispersion coefficients. If provided, the C6 model will also be trained.

        train_mask: torch.Tensor(N_BATCH,)
            Mask for training samples.

        alpha_mode: 'species' or 'reference'
            Mode for polarizability model.

        sigma: float
            GPR sigma value.

        ivm_thr: float
            IVM threshold.

        epochs: int
            Number of training epochs.

        lr_qeq: float
            Learning rate for QEq model.

        lr_thole: float
            Learning rate for Thole model.

        lr_sqrtk: float
            Learning rate for sqrtk.

        lr_c6: float
            Learning rate for C6 model.

        print_every: int
            How often to print training progress.

        computer_n_species: int
            Number of species supported by calculator (for ani2x backend)

        computer_zid_map: dict ({emle_zid: calculator_zid})
            Map between EMLE and calculator zid values (for ANI2x backend).

        model_filename: str or None
            Filename to save the trained model. If None, the model is not saved.

        emle_model: str or None
            Filename of an existing EMLE model to use as a starting point. If None, a new model is created.

        plot_data_filename: str or None
            Filename to write plotting data. If None, data is not written.

        device: torch.device
            Device to use for training.

        dtype: torch.dtype
            Data type to use for training. Default is torch.float64.

        Returns
        -------

        dict
            Trained EMLE model.
        """
        # Check input data.
        assert (
            len(z) == len(xyz) == len(s) == len(q_core) == len(q_val) == len(alpha)
        ), "z, xyz, s, q_core, q, and alpha must have the same number of samples"

        # Checks for alpha_mode.
        if not isinstance(alpha_mode, str):
            raise TypeError("'alpha_mode' must be of type 'str'")
        alpha_mode = alpha_mode.lower().replace(" ", "")
        if alpha_mode not in ["species", "reference"]:
            raise ValueError("'alpha_mode' must be 'species' or 'reference'")

        if train_mask is None:
            train_mask = _torch.ones(len(z), dtype=_torch.bool)

        # Prepare batch data.
        q_val = _pad_to_max(q_val)
        q_core = _pad_to_max(q_core)
        q = q_core + q_val
        q_mol = _torch.sum(q, dim=1)
        z = _pad_to_max(z)
        xyz = _pad_to_max(xyz)

        q_core_train = q_core[train_mask]
        q_mol_train = q_mol[train_mask]
        q_train = q[train_mask]
        z_train = z[train_mask]
        xyz_train = xyz[train_mask]
        s_train = _pad_to_max(s)[train_mask]
        alpha_train = _pad_to_max(alpha)[train_mask]
        species = _torch.unique(_torch.tensor(z_train[z_train > 0], device=device))

        # Place on the correct device and set the data type.
        q_mol = q_mol.to(device=device, dtype=dtype)
        q_mol_train = q_mol_train.to(device=device, dtype=dtype)
        z_train = z_train.to(device=device, dtype=_torch.int64)
        xyz_train = xyz_train.to(device=device, dtype=dtype)
        s_train = s_train.to(device=device, dtype=dtype)
        q_core_train = q_core_train.to(device=device, dtype=dtype)
        q_train = q_train.to(device=device, dtype=dtype)
        alpha_train = alpha_train.to(device=device, dtype=dtype)
        species = species.to(device=device, dtype=_torch.int64)

        if c6 is not None:
            c6 = _pad_to_max(c6)
            c6_train = c6[train_mask]
            c6_train = c6_train.to(device=device, dtype=dtype)

        # Get zid mapping.
        zid_mapping = self._get_zid_mapping(species)
        zid_train = zid_mapping[z_train]

        if computer_n_species is None:
            computer_n_species = len(species)

        if emle_model is not None:
            # Calculate AEVs.
            emle_aev_computer = _EMLEAEVComputer(
                num_species=computer_n_species,
                zid_map=computer_zid_map,
                dtype=dtype,
                device=device,
            )
            aev_mols = emle_aev_computer(zid_train, xyz_train)
            aev_mask = _torch.sum(aev_mols.reshape(-1, aev_mols.shape[-1]) ** 2, dim=0) > 0

            aev_mols = aev_mols[:, :, aev_mask]
            emle_aev_computer = _EMLEAEVComputer(
                num_species=computer_n_species,
                zid_map=computer_zid_map,
                mask=aev_mask,
                dtype=dtype,
                device=device,
            )

            # "Fit" q_core (just take averages over the entire training set).
            q_core_z = _mean_by_z(q_core_train, zid_train)

            _logger.info("Performing IVM...")
            # Create an array of (molecule_id, atom_id) pairs (as in the full
            # dataset) for the training set. This is needed to be able to locate
            # atoms/molecules in the original dataset that were picked by IVM.
            n_mols, max_atoms = q_train.shape
            atom_ids = _torch.stack(
                _torch.meshgrid(_torch.arange(n_mols), _torch.arange(max_atoms)), dim=-1
            ).to(device)

            # Perform IVM.
            ivm_mol_atom_ids_padded, aev_ivm_allz = _IVM.perform_ivm(
                aev_mols, z_train, atom_ids, species, ivm_thr, sigma
            )

            ref_features = _pad_to_max(aev_ivm_allz)
            ref_mask = ivm_mol_atom_ids_padded[:, :, 0] > -1
            n_ref = _torch.sum(ref_mask, dim=1)
            _logger.info("IVM done. Number of reference environments selected:")
            for atom_z, n in zip(species, n_ref):
                _logger.info(f"{atom_z:2d}: {n:5d}")

            # Fit s (pure GPR, no fancy optimization needed).
            ref_values_s = self._train_s(s_train, zid_train, aev_mols, aev_ivm_allz, sigma)

            # Good for debugging
            # _torch.autograd.set_detect_anomaly(True)
        else:
            emle = _EMLE(
                model=emle_model 
            )
            emle_base = emle._emle_base

        # Initial guess for the model parameters.
        params = {
            "a_QEq": _torch.tensor([1.0]).to(device=device, dtype=dtype),
            "a_Thole": _torch.tensor([2.0]).to(device=device, dtype=dtype),
            "ref_values_s": ref_values_s.to(device=device, dtype=dtype),
            "ref_values_chi": _torch.zeros(
                *ref_values_s.shape,
                dtype=ref_values_s.dtype,
                device=device,
            ),
            "k_Z": 0.5
            * _torch.ones(len(species), dtype=dtype, device=_torch.device(device)),
            "c6_Z": (
                _torch.ones(len(species), dtype=dtype, device=_torch.device(device))
                if c6 is not None
                else None
            ),
            "sqrtk_ref": (
                _torch.ones(
                    *ref_values_s.shape,
                    dtype=ref_values_s.dtype,
                    device=_torch.device(device),
                )
                if alpha_mode == "reference"
                else None
            ),
            "ref_values_c6": (
                _torch.ones(
                    *ref_values_s.shape,
                    dtype=ref_values_s.dtype,
                    device=_torch.device(device),
                )
                if c6 is not None
                else None
            ),
        }
    
        # Create the EMLE base instance.
        emle_base = self._emle_base(
            params=params,
            n_ref=n_ref,
            ref_features=ref_features,
            q_core=q_core_z,
            emle_aev_computer=emle_aev_computer,
            species=species,
            alpha_mode=alpha_mode,
            device=_torch.device(device),
            dtype=dtype,
        )

        # Fit chi, a_QEq (QEq over chi predicted with GPR).
        _logger.info("Fitting a_QEq and chi values...")
        self._train_model(
            loss_class=self._qeq_loss,
            opt_param_names=["a_QEq", "ref_values_chi"],
            lr=lr_qeq,
            epochs=epochs,
            print_every=print_every,
            emle_base=emle_base,
            atomic_numbers=z_train,
            xyz=xyz_train,
            q_mol=q_mol_train,
            q_target=q_train,
        )
        # Update GPR constants for chi
        # (now inconsistent since not updated after the last epoch)
        self._qeq_loss._update_chi_gpr(emle_base)

        _logger.debug(f"Optimized a_QEq: {emle_base.a_QEq.data.item()}")

        # Fit a_Thole, k_Z (uses volumes predicted by QEq model).
        _logger.info("Fitting a_Thole and k_Z values...")
        self._train_model(
            loss_class=self._thole_loss,
            opt_param_names=["a_Thole", "k_Z"],
            lr=lr_thole,
            epochs=epochs,
            print_every=print_every,
            emle_base=emle_base,
            atomic_numbers=z_train,
            xyz=xyz_train,
            q_mol=q_mol_train,
            alpha_mol_target=alpha_train,
        )

        _logger.debug(f"Optimized a_Thole: {emle_base.a_Thole.data.item()}")
        # Fit sqrtk_ref ( alpha = sqrtk ** 2 * k_Z * v).
        if alpha_mode == "reference":
            _logger.info("Fitting ref_values_sqrtk values...")
            self._train_model(
                loss_class=self._thole_loss,
                opt_param_names=["ref_values_sqrtk"],
                lr=lr_sqrtk,
                epochs=epochs,
                print_every=print_every,
                emle_base=emle_base,
                atomic_numbers=z_train,
                xyz=xyz_train,
                q_mol=q_mol_train,
                alpha_mol_target=alpha_train,
                opt_sqrtk=True,
                l2_reg=20.0,
            )
            # Update GPR constants for sqrtk
            # (now inconsistent since not updated after the last epoch)
            self._thole_loss._update_sqrtk_gpr(emle_base)

        if c6 is not None:
            _logger.info("Fitting ref_values_c6 values...")
            self._train_model(
                loss_class=self._dispersion_loss,
                opt_param_names=["ref_values_c6"],
                lr=lr_c6,
                epochs=2000,
                print_every=print_every,
                emle_base=emle_base,
                atomic_numbers=z_train,
                xyz=xyz_train,
                q_mol=q_mol_train,
                c6_target=c6_train,
            )

            # Update reference values for C6.
            self._dispersion_loss._update_c6_gpr(emle_base)

        # Create the final model.
        emle_model = {
            "q_core": q_core_z,
            "a_QEq": emle_base.a_QEq,
            "a_Thole": emle_base.a_Thole,
            "s_ref": emle_base.ref_values_s,
            "chi_ref": emle_base.ref_values_chi,
            "k_Z": emle_base.k_Z,
            "sqrtk_ref": (
                emle_base.ref_values_sqrtk if alpha_mode == "reference" else None
            ),
            "c6_ref": (emle_base.ref_values_c6 if c6 is not None else None),
            "species": species,
            "alpha_mode": alpha_mode,
            "n_ref": n_ref,
            "ref_aev": ref_features,
            "aev_mask": aev_mask,
            "zid_map": emle_aev_computer._zid_map,
            "computer_n_species": computer_n_species,
        }

        if model_filename is not None:
            self._write_model_to_file(emle_model, model_filename)

        if plot_data_filename is None:
            return emle_base

        emle_base._alpha_mode = "species"
        s_pred, q_core_pred, q_val_pred, A_thole, c6_pred = emle_base(
            z.to(device=device, dtype=_torch.int64),
            xyz.to(device=device, dtype=dtype),
            q_mol,
        )
        z_mask = _torch.tensor(z > 0, device=device)
        plot_data = {
            "s_emle": s_pred,
            "q_core_emle": q_core_pred,
            "q_val_emle": q_val_pred,
            "alpha_species": self._thole_loss._get_alpha_mol(A_thole, z_mask),
            "z": z,
            "s_qm": s,
            "q_core_qm": q_core,
            "q_val_qm": q_val,
            "alpha_qm": alpha,
        }

        if alpha_mode == "reference":
            emle_base._alpha_mode = "reference"
            *_, A_thole = emle_base(
                z.to(device=device, dtype=_torch.int64),
                xyz.to(device=device, dtype=dtype),
                q_mol,
            )
            plot_data["alpha_reference"] = self._thole_loss._get_alpha_mol(
                A_thole, z_mask
            )

        if c6 is not None:
            plot_data["c6_qm"] = c6
            plot_data["c6_emle"] = c6_pred

        self._write_model_to_file(plot_data, plot_data_filename)

        return emle_base

    def train_nagl(
        self,
        z_qm,
        z_mm,
        xyz_qm,
        xyz_mm,
        q_mol_qm,
        q_mm,
        e_exrep,
        e_sr_corr,
        train_mask,
        alpha_mode="reference",
        epochs=100,
        lr_s=1e-4,
        lr_exrep=1e-3,
        print_every=10,
        batch_size=1024,
        model_filename="emle_model.mat",
        device=_torch.device("cuda"),
        dtype=_torch.float64,
        shuffle=True,
        train_mode="sequential",
        loss_weight_exrep=1.0,
        loss_weight_sr_corr=1.0,
        z_s=None,
        xyz_s=None,
        s=None,
    ):
        """
        Train an EMLENAGL model.

        Parameters
        ----------

        train_mode: str, optional, default="sequential"
            Training mode for exrep and short-range correction parameters.
            Options are:
            - "sequential": Train A_exrep first, then A_sr_corr (default behavior)
            - "simultaneous": Train both A_exrep and A_sr_corr together with combined loss

        loss_weight_exrep: float, optional, default=1.0
            Weight for the exchange repulsion loss when train_mode="simultaneous".
            Only used in simultaneous training mode.

        loss_weight_sr_corr: float, optional, default=1.0
            Weight for the short-range correction loss when train_mode="simultaneous".
            Only used in simultaneous training mode.

        Returns
        -------
        """
        from ..models import NAGLEMLE

        # Validate train_mode parameter
        train_mode = train_mode.lower().replace(" ", "")
        if train_mode not in ("sequential", "simultaneous"):
            raise ValueError(
                f"train_mode must be 'sequential' or 'simultaneous', got '{train_mode}'"
            )

        # Validate loss weights
        if not isinstance(loss_weight_exrep, (int, float)) or loss_weight_exrep <= 0:
            raise ValueError("loss_weight_exrep must be a positive number")
        if (
            not isinstance(loss_weight_sr_corr, (int, float))
            or loss_weight_sr_corr <= 0
        ):
            raise ValueError("loss_weight_sr_corr must be a positive number")

        _logger.info(f"Training mode: {train_mode}")
        if train_mode == "simultaneous":
            _logger.info(
                f"Loss weights - exrep: {loss_weight_exrep}, sr_corr: {loss_weight_sr_corr}"
            )

        assert (
            len(z_qm) == len(z_mm) == len(xyz_qm) == len(xyz_mm) == len(e_exrep)
        ), "z, xyz, and e_exrep must have the same number of samples"

        if train_mask is None:
            train_mask = _torch.ones(len(z_qm), dtype=_torch.bool)

        # Prepare batch data.
        z_qm = _pad_to_max(z_qm)
        xyz_qm = _pad_to_max(xyz_qm)
        z_mm = _pad_to_max(z_mm)
        xyz_mm = _pad_to_max(xyz_mm)
        e_exrep = _pad_to_max(e_exrep)
        e_sr_corr = _pad_to_max(e_sr_corr)
        q_mol_qm = _pad_to_max(q_mol_qm)
        q_mm = _pad_to_max(q_mm)

        z_qm_train = z_qm[train_mask]
        xyz_qm_train = xyz_qm[train_mask]
        z_mm_train = z_mm[train_mask]
        xyz_mm_train = xyz_mm[train_mask]
        e_exrep_train = e_exrep[train_mask]
        e_sr_corr_train = e_sr_corr[train_mask]
        q_mol_qm_train = q_mol_qm[train_mask]
        q_mm_train = q_mm[train_mask]       

        species = _torch.unique(
            _torch.tensor(z_qm_train[z_qm_train > 0], device=device)
        )

        # Place on the correct device and set the data type.
        z_qm_train = z_qm_train.to(device=device, dtype=_torch.int64)
        xyz_qm_train = xyz_qm_train.to(device=device, dtype=dtype)
        z_mm_train = z_mm_train.to(device=device, dtype=_torch.int64)
        xyz_mm_train = xyz_mm_train.to(device=device, dtype=dtype)
        e_exrep_train = e_exrep_train.to(device=device, dtype=dtype)
        e_sr_corr_train = e_sr_corr_train.to(device=device, dtype=dtype)
        species = species.to(device=device, dtype=_torch.int64)
        q_mol_qm_train = q_mol_qm_train.to(device=device, dtype=dtype)
        q_mm_train = q_mm_train.to(device=device, dtype=dtype)
        q_mol_mm_train = _torch.sum(q_mm_train, dim=1)

        # Shuffle
        if shuffle:
            indices = _torch.randperm(z_qm_train.shape[0])
            z_qm_train = z_qm_train[indices]
            xyz_qm_train = xyz_qm_train[indices]
            z_mm_train = z_mm_train[indices]
            xyz_mm_train = xyz_mm_train[indices]
            e_exrep_train = e_exrep_train[indices]
            e_sr_corr_train = e_sr_corr_train[indices]
            q_mol_qm_train = q_mol_qm_train[indices]
            q_mm_train = q_mm_train[indices]
            q_mol_mm_train = q_mol_mm_train[indices]

        if z_s is not None and xyz_s is not None and s is not None:
            z_s = _pad_to_max(z_s)
            xyz_s = _pad_to_max(xyz_s)
            s = _pad_to_max(s)
            z_target_s = z_s.to(device=device, dtype=_torch.int64)
            xyz_target_s = xyz_s.to(device=device, dtype=dtype)
            s_target_s = s.to(device=device, dtype=dtype)
            
        # Create the EMLE model
        emle = _EMLE(
            model="/home/joaomorado/repos/emle-bespoke/examples/DES_dimers/ws/ligand_patched_species_iter2.mat",
            alpha_mode=alpha_mode,
            device=device,
            dtype=dtype,
        )
        emle_base = emle._emle_base

        # Create the NAGL model
        nagl = NAGLEMLE(
            species=[1, 6, 7, 8, 16],
            properties=["A_exrep", "A_sr_corr", "s"],
            n_conv_layers=4,
            hidden_dim=512,
            n_ffnn_layers=4,
            #model_filepath="/home/joaomorado/repos/emle-bespoke/examples/DES_dimers/ws/emle_nagl.pt"
        ).to(device=device, dtype=dtype)

        # Create OpenFF molecules and dataloaders
        _logger.info("Creating OpenFF molecules and dataloaders for QM region...")
        off_qm = nagl.create_openff_mol(z_qm_train, xyz_qm_train, charge=q_mol_qm_train)
        dataloader_qm = nagl.create_dataloader(
            off_qm, device=device, batch_size=batch_size
        )
        _logger.info("Creating OpenFF molecules and dataloaders for MM region...")
        off_mm = nagl.create_openff_mol(z_mm_train, xyz_mm_train, charge=q_mol_mm_train)
        dataloader_mm = nagl.create_dataloader(
            off_mm, device=device, batch_size=batch_size
        )

        if z_s is not None and xyz_s is not None and s is not None:
            _logger.info("Creating OpenFF molecules and dataloaders for s target region...")
            off_s = nagl.create_openff_mol(z_target_s, xyz_target_s, charge=_torch.zeros(len(z_target_s), dtype=dtype))
            dataloader_s = nagl.create_dataloader(
                off_s, device=device, batch_size=batch_size
            )

        # Create dataloaders for QM and MM data
        # Create dataloaders for QM and MM data
        _logger.info("Creating dataloaders for training valence widths...")
        dataset_tensor_s = _TensorDataset(
            z_target_s,
            xyz_target_s,
            s_target_s,
        )
        dataloader_tensor_s = _DataLoader(
            dataset_tensor_s, batch_size=batch_size, shuffle=False
        )

        if z_s is not None and xyz_s is not None and s is not None:
            _logger.info("Fitting s parameters...")
            loss_instance_s = _SLoss(
                emle_base, nagl, property_label=2, loss=_torch.nn.MSELoss()
            )
            loss_instance_s.eval()
            # Only optimize s-related parameters
            opt_parameters_s = [param for name, param in nagl.named_parameters() if  "joint" not in name]
            optimizer_s = _torch.optim.Adam(opt_parameters_s, lr=lr_s)
            for epoch in range(epochs):
                loss_instance_s.train()
                total_loss = total_rmse = total_max_error = 0.0
                values_s, target_s = [], []
                
                for (graphs_s_batch, _), s_batch in zip(
                    dataloader_s, dataloader_tensor_s
                ):
                    optimizer_s.zero_grad()
            
                    _, _, s_target_train = s_batch

                    # Loss for s only
                    loss_s, rmse_s, max_error_s, v_s, t_s = loss_instance_s(
                        graphs_s_batch, s_target_train
                    )
                    values_s.append(v_s.detach().cpu())
                    target_s.append(t_s.detach().cpu())
                    
                    loss_s.backward(retain_graph=False)
                    optimizer_s.step()

                    total_loss += loss_s.item()
                    total_rmse += rmse_s.item()
                    total_max_error = max(total_max_error, max_error_s.item())

                values_s = _torch.cat(values_s)
                target_s = _torch.cat(target_s)
                _np.savetxt("nagl_s_predicted.txt", values_s.numpy())
                _np.savetxt("nagl_s_target.txt", target_s.numpy())
                rmse_s = _torch.sqrt(_torch.mean((values_s - target_s) ** 2)).item()
                max_error_s = _torch.max(_torch.abs(values_s - target_s)).item()
                if (epoch + 1) % print_every == 0:
                    _logger.info(
                        f"Epoch {epoch+1}: s Loss ={total_loss:9.4f}    "
                        f"s RMSE ={rmse_s:9.4f}    s Max Error ={max_error_s:9.4f}"
                    )
        else:
            # Fit s to the widths predicted by EMLE
            _logger.info("Fitting s...")
            loss_instance = _AtomicPropertyLoss(
                emle_base, nagl, property_label="s", loss=_torch.nn.MSELoss()
            )
            loss_instance.eval()
            opt_parameters = [param for name, param in nagl.named_parameters()]
            optimizer = _torch.optim.Adam(opt_parameters, lr=lr_s)
            for epoch in range(epochs):
                loss_instance.train()
                total_loss = total_rmse = total_max_error = 0.0
                values, target = [], []
                for (graphs_qm_batch, _), (graphs_mm_batch, _), tensor_batch in zip(
                    dataloader_qm, dataloader_mm, dataloader_tensors
                ):
                    optimizer.zero_grad()
                    (
                        _,
                        q_val_qm_train,
                        _,
                        q_val_mm_train,
                        s_qm_train,
                        s_mm_train,
                        e_exrep_train,
                        _,
                        _,
                        _,
                        *mesh_data,
                    ) = tensor_batch
                    loss, rmse, max_error, v, t = loss_instance(
                        graphs_qm_batch, graphs_mm_batch, s_qm_train, s_mm_train
                    )
                    values.append(v.detach().cpu())
                    target.append(t.detach().cpu())
                    loss = loss / len(q_val_qm_train)
                    loss.backward(retain_graph=False)
                    optimizer.step()
                    total_loss += loss.item()
                    total_rmse += rmse.item()
                    total_max_error = max(total_max_error, max_error.item())

                values = _torch.cat(values)
                target = _torch.cat(target)
                _np.savetxt("nagl_s_predicted.txt", values.numpy())
                _np.savetxt("nagl_s_target.txt", target.numpy())

                rmse = _torch.sqrt(_torch.mean((values - target) ** 2)).item()
                max_error = _torch.max(_torch.abs(values - target)).item()

                if (epoch + 1) % print_every == 0:
                    _logger.info(
                        f"Epoch {epoch+1}: Loss ={total_loss:9.4f}    "
                        f"RMSE ={rmse:9.4f}    "
                        f"Max Error ={max_error:9.4f}"
                    )


        # Create dataloader
        dataset = _TensorDataset(z_qm_train, xyz_qm_train, q_mol_qm_train)
        dataloader = _DataLoader(dataset, batch_size=batch_size, shuffle=False)

        _logger.info(
            "Pre-computing valence charges and widths for the QM references..."
        )
        # Valence widths, core charges, valence charges, A_thole tensor, A_exrep parameters, A_short_range_corr parameters
        # Pre-compute valence charges and widths for the QM references
        q_val_qm_train = []
        s_qm_train = []
        q_core_qm_train = []
        A_thole_qm_train = []
        c6_qm_train = []
        with _torch.no_grad():
            for batch in dataloader:
                s_qm, q_core_qm, q_val_qm, A_thole, c6 = emle_base.forward(*batch)
                q_val_qm_train.append(q_val_qm)
                s_qm_train.append(s_qm)
                q_core_qm_train.append(q_core_qm)
                A_thole_qm_train.append(A_thole)
                c6_qm_train.append(A_thole) # TODO: placeholder, need to fix c6 in EMLE to output properly
        s_qm_train = _torch.cat(s_qm_train)
        q_val_qm_train = _torch.cat(q_val_qm_train)
        q_core_qm_train = _torch.cat(q_core_qm_train)
        A_thole_qm_train = _torch.cat(A_thole_qm_train)
        c6_qm_train = _torch.cat(c6_qm_train) 

        # Pre-compute valence charges and widths for the MM references
        dataset = _TensorDataset(z_mm_train, xyz_mm_train, q_mol_mm_train)
        dataloader = _DataLoader(dataset, batch_size=batch_size, shuffle=False)
        s_mm_train = []
        q_core_mm_train = []
        with _torch.no_grad():
            for batch in dataloader:
                s_mm, q_core_mm, _, *_ = emle_base.forward(*batch)
                s_mm_train.append(s_mm)
                q_core_mm_train.append(q_core_mm)
        s_mm_train = _torch.cat(s_mm_train)
        q_core_mm_train = _torch.cat(q_core_mm_train)

        species_id_mm = emle_base._species_map[z_mm_train]
        q_val_mm_train = q_mm_train - emle_base._q_core[species_id_mm]
        q_val_mm_train = q_val_mm_train * (s_mm_train > 0)

        # Get mesh data
        mask = (z_qm_train > 0).unsqueeze(-1)
        ANGSTROM_TO_BOHR = 1.8897261258369282
        mesh_data = emle_base._get_mesh_data(
            xyz_qm_train * ANGSTROM_TO_BOHR,
            xyz_mm_train * ANGSTROM_TO_BOHR,
            s_qm_train,
            mask
        )

        _logger.info("Creating dataloaders for training tensors...")
        dataset_tensors = _TensorDataset(
            q_core_qm_train,
            q_val_qm_train,
            q_core_mm_train,
            q_val_mm_train,
            s_qm_train,
            s_mm_train,
            e_exrep_train,
            e_sr_corr_train,
            A_thole_qm_train,
            c6_qm_train,
            *mesh_data,
        )
        dataloader_tensors = _DataLoader(
            dataset_tensors, batch_size=batch_size, shuffle=False
        )

        # Pre-compute valence widths for the MM references using NAGL
        s_mm_train = []
        with _torch.no_grad():
            for (graphs_mm_batch, _) in dataloader_mm:
                s_mm_batch = nagl(graphs_mm_batch)["s"]
                s_mm_batch = _torch.nn.functional.pad(
                    s_mm_batch, (0, z_mm_train.size(1) - s_mm_batch.size(1))
                )
                s_mm_train.append(s_mm_batch)
        s_mm_train = _torch.cat(s_mm_train)
        dataset_tensors = _TensorDataset(
            q_core_qm_train,
            q_val_qm_train,
            q_core_mm_train,
            q_val_mm_train,
            s_qm_train,
            s_mm_train,
            e_exrep_train,
            e_sr_corr_train,
            A_thole_qm_train,
            *mesh_data,
        )

        dataloader_tensors = _DataLoader(
            dataset_tensors, batch_size=batch_size, shuffle=False
        )

        # Fit A_exrep
        _logger.info("Fitting exchange-repulsion parameters...")
        loss_instance = _ExchangeRepulsionLoss( 
            emle_base, nagl, loss=_torch.nn.MSELoss()
        )
        loss_instance.eval()
        opt_parameters = [
            param for name, param in nagl.named_parameters() if "A_exrep" in name
        ]
        #opt_parameters += [
        #    param for name, param in nagl.named_parameters() if "B_exrep" in name
        #]

        _logger.info(
            f"Optimizing parameters: {[name for name, _ in nagl.named_parameters() if 'A_exrep' in name or 'B_exrep' in name]}"
        )
        optimizer = _torch.optim.Adam(opt_parameters, lr=lr_exrep)
        for epoch in range(epochs):
            loss_instance.train()
            total_loss = total_rmse = total_max_error = 0.0
            values, target = [], []
            for (graphs_qm_batch, _), (graphs_mm_batch, _), tensor_batch in zip(
                dataloader_qm, dataloader_mm, dataloader_tensors
            ):
                optimizer.zero_grad()
                (
                    _,
                    q_val_qm_train,
                    _,
                    q_val_mm_train,
                    s_qm_train,
                    s_mm_train,
                    e_exrep_train,
                    _,
                    _,
                    *mesh_data,
                ) = tensor_batch
                
                loss, rmse, max_error, v, t = loss_instance(
                    graphs_qm_batch,
                    graphs_mm_batch,
                    q_val_qm_train,
                    q_val_mm_train,
                    mesh_data,
                    s_qm_train,
                    s_mm_train,
                    e_exrep_train,
                )
                values.append(v.detach().cpu())
                target.append(t.detach().cpu())
                loss = loss / len(q_val_qm_train)
                loss.backward(retain_graph=False)
                optimizer.step()
                total_loss += loss.item()
                total_rmse += rmse.item()
                total_max_error = max(total_max_error, max_error.item())

            values = _torch.cat(values)
            target = _torch.cat(target)
            _np.savetxt("nagl_exrep_predicted.txt", values.numpy())
            _np.savetxt("nagl_exrep_target.txt", target.numpy())

            rmse = _torch.sqrt(_torch.mean((values - target) ** 2)).item()
            max_error = _torch.max(_torch.abs(values - target)).item()

            if (epoch + 1) % print_every == 0:
                _logger.info(
                    f"Epoch {epoch+1}: Loss ={total_loss:9.4f}    "
                    f"RMSE ={rmse:9.4f}    "
                    f"Max Error ={max_error:9.4f}"
                )
        # Pre-compute E_static_emle, E_ind_emle, and E_exrep_emle
        E_static_emle = []
        E_ind_emle = []
        E_exrep_emle = []
        E_disp_emle = []
        nagl.eval()
        for (graphs_qm_batch, _), (graphs_mm_batch, _), tensor_batch in zip(
            dataloader_qm, dataloader_mm, dataloader_tensors
        ):
            with _torch.no_grad():
                (
                    q_core_qm_train,
                    q_val_qm_train,
                    q_core_mm_train,
                    q_val_mm_train,
                    s_qm_train,
                    s_mm_train,
                    e_exrep_train,
                    e_sr_corr_train,
                    A_thole_qm_train,
                    *mesh_data,
                ) = tensor_batch
            
                # Static energy
                E_static = emle_base._get_static_energy_slater(
                    q_core_qm_train,
                    q_val_qm_train,
                    q_core_mm_train,
                    q_val_mm_train,
                    mesh_data,
                    s_qm_train,
                    s_mm_train,
                )

                # Induction energy
                charges_mm = q_core_mm_train + q_val_mm_train
                mask = s_qm_train > 0
                E_ind = emle_base.get_induced_energy(
                    A_thole_qm_train, charges_mm, s_qm_train, mesh_data, mask
                )
           
                # Exchange-repulsion
                A_exrep_qm = nagl(graphs_qm_batch)["A_exrep"]
                A_exrep_mm = nagl(graphs_mm_batch)["A_exrep"]
                #B_exrep_qm = nagl(graphs_qm_batch)["B_exrep"]
                #B_exrep_mm = nagl(graphs_mm_batch)["B_exrep"]
                A_exrep_qm = _torch.nn.functional.pad(
                    A_exrep_qm, (0, q_val_qm_train.size(1) - A_exrep_qm.size(1))
                )
                A_exrep_mm = _torch.nn.functional.pad(
                    A_exrep_mm, (0, q_val_mm_train.size(1) - A_exrep_mm.size(1))
                )
                #B_exrep_qm = _torch.nn.functional.pad(
                #    B_exrep_qm, (0, q_val_qm_train.size(1) - B_exrep_qm.size(1))
                #)
                #B_exrep_mm = _torch.nn.functional.pad(
                #    B_exrep_mm, (0, q_val_mm_train.size(1) - B_exrep_mm.size(1))
                #)
                E_exrep = emle_base.get_exchange_repulsion_energy(
                    A_exrep_qm,
                    A_exrep_mm,
                    #B_exrep_qm,
                    #B_exrep_mm,
                    q_val_qm_train,
                    q_val_mm_train,
                    mesh_data,
                    s_qm_train,
                    s_mm_train,
                )

                # Dispersion energy
                """
                alpha_qm = emle_base.get_isotropic_polarizabilities(A_thole_qm_train)
                c6_mm = 4 * epsilon_mm * sigma_mm**6
                c6_qm = c6 * alpha_qm * 0.5
                E_disp = emle_base.get_dispersion_energy(
                    c6_qm,
                    c6_mm,
                    s_qm_train,
                    s_mm_train,
                    mesh_data,
                )
                """

                # Append
                E_static_emle.append(E_static)
                E_ind_emle.append(E_ind)
                E_exrep_emle.append(E_exrep) 
                E_disp_emle.append(E_exrep) # Placeholder TODO: fix dispersion

        E_static_emle = _torch.cat(E_static_emle)
        E_ind_emle = _torch.cat(E_ind_emle)
        E_exrep_emle = _torch.cat(E_exrep_emle)
        E_disp_emle = _torch.cat(E_disp_emle)

        dataset = _torch.utils.data.TensorDataset(
            E_static_emle, E_ind_emle, E_exrep_emle, E_disp_emle
        )
        dataloader_energy = _torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        # Fit A_sr_corr
        _logger.info("Fitting short-range correction parameters...")
        loss_instance = _ShortRangeCorrectionLoss(
            emle_base, nagl, loss=_torch.nn.MSELoss()
        )
        loss_instance.eval()
        opt_parameters = [
            param for name, param in nagl.named_parameters() if "A_sr_corr" in name
        ]
        _logger.info(
            f"Optimizing parameters: {[name for name, _ in nagl.named_parameters() if 'A_sr_corr' in name]}"
        )
        optimizer = _torch.optim.Adam(opt_parameters, lr=lr_exrep)
        for epoch in range(epochs):
            loss_instance.train()
            total_loss = total_rmse = total_max_error = 0.0
            values, target = [], []
            for (
                (graphs_qm_batch, _),
                (graphs_mm_batch, _),
                tensor_batch,
                energies_batch,
            ) in zip(
                dataloader_qm, dataloader_mm, dataloader_tensors, dataloader_energy
            ):
                optimizer.zero_grad()
                (
                    _,  
                    q_val_qm_train,
                    _,
                    q_val_mm_train,
                    s_qm_train,
                    s_mm_train,
                    _,
                    e_sr_corr_train,
                    _,
                    *mesh_data,
                ) = tensor_batch
                e_static_train, e_ind_train, e_exrep_train, e_disp_train = energies_batch
                e_target_train = e_sr_corr_train
                offset = (e_static_train + e_ind_train + e_exrep_train) * 2625.5002
                e_target_train = e_target_train
                loss, rmse, max_error, v, t = loss_instance(
                    graphs_qm_batch,
                    graphs_mm_batch,
                    q_val_qm_train,
                    q_val_mm_train,
                    mesh_data,
                    s_qm_train,
                    s_mm_train,
                    e_target_train,
                    offset,
                )
                values.append(v.detach().cpu())
                target.append(t.detach().cpu())
                loss = loss / len(q_val_qm_train)
                loss.backward(retain_graph=False)
                optimizer.step()
                total_loss += loss.item()
                total_rmse += rmse.item()
                total_max_error = max(total_max_error, max_error.item())

            values = _torch.cat(values)
            target = _torch.cat(target)
            _np.savetxt("nagl_sr_corr_predicted.txt", values.numpy())
            _np.savetxt("nagl_sr_corr_target.txt", target.numpy())

            rmse = _torch.sqrt(_torch.mean((values - target) ** 2)).item()
            max_error = _torch.max(_torch.abs(values - target)).item()

            if (epoch + 1) % print_every == 0:
                _logger.info(
                    f"Epoch {epoch+1}: Loss ={total_loss:9.4f}    "
                    f"RMSE ={rmse:9.4f}    "
                    f"Max Error ={max_error:9.4f}"
                )

        # Save the model to a file
        nagl.save("emle_nagl.pt")


    def train_nagl_simultaneous(
        self,
        z_qm,
        z_mm,
        xyz_qm,
        xyz_mm,
        q_mol_qm,
        q_mm,
        e_exrep,
        e_sr_corr,
        train_mask,
        z_s=None,
        xyz_s=None,
        s=None,
        alpha_mode="reference",
        epochs=100,
        lr_s=1e-4,
        lr_exrep=1e-3,
        print_every=10,
        batch_size=1024,
        model_filename="emle_model.mat",
        device=_torch.device("cuda"),
        dtype=_torch.float64,
        shuffle=True,
        loss_weight_s=1.0,
        loss_weight_exrep=1.0,
        loss_weight_sr_corr=1.0,
    ):
        """
        Train an EMLENAGL model.

        Parameters
        ----------

        train_mode: str, optional, default="sequential"
            Training mode for exrep and short-range correction parameters.
            Options are:
            - "sequential": Train A_exrep first, then A_sr_corr (default behavior)
            - "simultaneous": Train both A_exrep and A_sr_corr together with combined loss

        loss_weight_exrep: float, optional, default=1.0
            Weight for the exchange repulsion loss when train_mode="simultaneous".
            Only used in simultaneous training mode.

        loss_weight_sr_corr: float, optional, default=1.0
            Weight for the short-range correction loss when train_mode="simultaneous".
            Only used in simultaneous training mode.

        Returns
        -------
        """
        from ..models import NAGLEMLE
    
        assert (
            len(z_qm) == len(z_mm) == len(xyz_qm) == len(xyz_mm) == len(e_exrep)
        ), "z, xyz, and e_exrep must have the same number of samples"

        if train_mask is None:
            train_mask = _torch.ones(len(z_qm), dtype=_torch.bool)

        # Prepare batch data.
        z_qm = _pad_to_max(z_qm)
        xyz_qm = _pad_to_max(xyz_qm)
        z_mm = _pad_to_max(z_mm)
        xyz_mm = _pad_to_max(xyz_mm)
        e_exrep = _pad_to_max(e_exrep)
        e_sr_corr = _pad_to_max(e_sr_corr)
        q_mol_qm = _pad_to_max(q_mol_qm)
        q_mm = _pad_to_max(q_mm)

        z_qm_train = z_qm[train_mask]
        xyz_qm_train = xyz_qm[train_mask]
        z_mm_train = z_mm[train_mask]
        xyz_mm_train = xyz_mm[train_mask]
        e_exrep_train = e_exrep[train_mask]
        e_sr_corr_train = e_sr_corr[train_mask]
        q_mol_qm_train = q_mol_qm[train_mask]
        q_mm_train = q_mm[train_mask]
        species = _torch.unique(
            _torch.tensor(z_qm_train[z_qm_train > 0], device=device)
        )

        # Place on the correct device and set the data type.
        z_qm_train = z_qm_train.to(device=device, dtype=_torch.int64)
        xyz_qm_train = xyz_qm_train.to(device=device, dtype=dtype)
        z_mm_train = z_mm_train.to(device=device, dtype=_torch.int64)
        xyz_mm_train = xyz_mm_train.to(device=device, dtype=dtype)
        e_exrep_train = e_exrep_train.to(device=device, dtype=dtype)
        e_sr_corr_train = e_sr_corr_train.to(device=device, dtype=dtype)
        species = species.to(device=device, dtype=_torch.int64)
        q_mol_qm_train = q_mol_qm_train.to(device=device, dtype=dtype)
        q_mm_train = q_mm_train.to(device=device, dtype=dtype)
        q_mol_mm_train = _torch.sum(q_mm_train, dim=1)

        # Shuffle
        if shuffle:
            indices = _torch.randperm(z_qm_train.shape[0])
            z_qm_train = z_qm_train[indices]
            xyz_qm_train = xyz_qm_train[indices]
            z_mm_train = z_mm_train[indices]
            xyz_mm_train = xyz_mm_train[indices]
            e_exrep_train = e_exrep_train[indices]
            e_sr_corr_train = e_sr_corr_train[indices]
            q_mol_qm_train = q_mol_qm_train[indices]
            q_mm_train = q_mm_train[indices]
            q_mol_mm_train = q_mol_mm_train[indices]


        if z_s is not None and xyz_s is not None and s is not None:
            z_s = _pad_to_max(z_s)
            xyz_s = _pad_to_max(xyz_s)
            s = _pad_to_max(s)
            z_target_s = z_s.to(device=device, dtype=_torch.int64)
            xyz_target_s = xyz_s.to(device=device, dtype=dtype)
            s_target_s = s.to(device=device, dtype=dtype)

        # Create the EMLE model
        emle = _EMLE(
            model="/home/joaomorado/repos/emle-bespoke/examples/DES_dimers/ws/ligand_patched_species_iter2.mat",
            alpha_mode=alpha_mode,
            device=device,
            dtype=dtype,
        )
        emle_base = emle._emle_base

        # Create dataloader
        dataset = _TensorDataset(z_qm_train, xyz_qm_train, q_mol_qm_train)
        dataloader = _DataLoader(dataset, batch_size=batch_size, shuffle=False)

    
        # Create the NAGL model
        nagl = NAGLEMLE(
            species=[1, 6, 7, 8, 16],
            properties=["A_exrep", "A_sr_corr", "s"],
            n_conv_layers=4,
            hidden_dim=512,
            n_ffnn_layers=4,
            joint_decoder=False,
            # model_filepath="/home/joaomorado/repos/emle-bespoke/examples/DES_dimers/ws/emle_nagl.pt"
        ).to(device)

        # Create OpenFF molecules and dataloaders
        _logger.info("Creating OpenFF molecules and dataloaders for QM region...")
        off_qm = nagl.create_openff_mol(z_qm_train, xyz_qm_train, charge=q_mol_qm_train)
        dataloader_qm = nagl.create_dataloader(
            off_qm, device=device, batch_size=batch_size
        )
        _logger.info("Creating OpenFF molecules and dataloaders for MM region...")
        off_mm = nagl.create_openff_mol(z_mm_train, xyz_mm_train, charge=q_mol_mm_train)
        dataloader_mm = nagl.create_dataloader(
            off_mm, device=device, batch_size=batch_size
        )

        if z_s is not None and xyz_s is not None and s is not None:
            _logger.info("Creating OpenFF molecules and dataloaders for s target region...")
            off_s = nagl.create_openff_mol(z_target_s, xyz_target_s, charge=_torch.zeros(len(z_target_s), dtype=dtype))
            dataloader_s = nagl.create_dataloader(
                off_s, device=device, batch_size=batch_size
            )

        # Create dataloaders for QM and MM data
        _logger.info("Creating dataloaders for training valence widths...")
        dataset_tensor_s = _TensorDataset(
            z_target_s,
            xyz_target_s,
            s_target_s,
        )
        dataloader_tensor_s = _DataLoader(
            dataset_tensor_s, batch_size=batch_size, shuffle=False
        )

        # First, fit s parameters
        _logger.info("Fitting s parameters...")
        loss_instance_s = _SLoss(
            emle_base, nagl, property_label=2, loss=_torch.nn.MSELoss()
        )
        loss_instance_s.eval()
        
        # Only optimize s-related parameters
        opt_parameters_s = [param for name, param in nagl.named_parameters() if  "joint" not in name]
        optimizer_s = _torch.optim.Adam(opt_parameters_s, lr=lr_s)
        
        for epoch in range(epochs):
            loss_instance_s.train()
            total_loss = total_rmse = total_max_error = 0.0
            values_s, target_s = [], []
            
            for (graphs_s_batch, _), s_batch in zip(
                dataloader_s, dataloader_tensor_s
            ):
                optimizer_s.zero_grad()
          
                _, _, s_target_train = s_batch

                # Loss for s only
                loss_s, rmse_s, max_error_s, v_s, t_s = loss_instance_s(
                    graphs_s_batch, s_target_train
                )
                values_s.append(v_s.detach().cpu())
                target_s.append(t_s.detach().cpu())
                
                loss_s.backward(retain_graph=False)
                optimizer_s.step()

                total_loss += loss_s.item()
                total_rmse += rmse_s.item()
                total_max_error = max(total_max_error, max_error_s.item())

            values_s = _torch.cat(values_s)
            target_s = _torch.cat(target_s)
            _np.savetxt("nagl_s_predicted.txt", values_s.numpy())
            _np.savetxt("nagl_s_target.txt", target_s.numpy())

            rmse_s = _torch.sqrt(_torch.mean((values_s - target_s) ** 2)).item()
            max_error_s = _torch.max(_torch.abs(values_s - target_s)).item()
            
            if (epoch + 1) % print_every == 0:
                _logger.info(
                    f"Epoch {epoch+1}: s Loss ={total_loss:9.4f}    "
                    f"s RMSE ={rmse_s:9.4f}    s Max Error ={max_error_s:9.4f}"
                )

        _logger.info(
            "Pre-computing valence charges and widths..."
        )
        # Valence widths, core charges, valence charges, A_thole tensor, A_exrep parameters, A_short_range_corr parameters
        # Pre-compute valence charges and widths for the QM references
        q_val_qm_train = []
        s_qm_train = []
        q_core_qm_train = []
        A_thole_qm_train = []
        with _torch.no_grad():
            for batch in dataloader:
                s_qm, q_core_qm, q_val_qm, A_thole, _ = emle_base.forward(*batch)
                q_val_qm_train.append(q_val_qm)
                s_qm_train.append(s_qm)
                q_core_qm_train.append(q_core_qm)
                A_thole_qm_train.append(A_thole)
        s_qm_train = _torch.cat(s_qm_train)
        q_val_qm_train = _torch.cat(q_val_qm_train)
        q_core_qm_train = _torch.cat(q_core_qm_train)
        A_thole_qm_train = _torch.cat(A_thole_qm_train)

        # Pre-compute valence charges for the MM references
        dataset = _TensorDataset(z_mm_train, xyz_mm_train, q_mol_mm_train)
        dataloader = _DataLoader(dataset, batch_size=batch_size, shuffle=False)
        q_core_mm_train = []
        s_mm_train = []
        with _torch.no_grad():
            for batch in dataloader:
                s_mm, q_core_mm, _, *_ = emle_base.forward(*batch)
                q_core_mm_train.append(q_core_mm)
                s_mm_train.append(s_mm)
        s_mm_train = _torch.cat(s_mm_train)
        q_core_mm_train = _torch.cat(q_core_mm_train)
        species_id_mm = emle_base._species_map[z_mm_train]
        q_val_mm_train = q_mm_train - emle_base._q_core[species_id_mm]
        q_val_mm_train = q_val_mm_train * (s_mm_train > 0)

        # Pre-compute valence widths for the MM references using NAGL
        s_mm_train = []
        with _torch.no_grad():
            for (graphs_mm_batch, _) in dataloader_mm:
                s_mm_batch = nagl(graphs_mm_batch)["s"]
                s_mm_batch = _torch.nn.functional.pad(
                    s_mm_batch, (0, z_mm_train.size(1) - s_mm_batch.size(1))
                )
                s_mm_train.append(s_mm_batch)
        s_mm_train = _torch.cat(s_mm_train)

        # Get mesh data
        mask = (z_qm_train > 0).unsqueeze(-1)
        ANGSTROM_TO_BOHR = 1.8897261258369282
        mesh_data = emle_base._get_mesh_data(
            xyz_qm_train * ANGSTROM_TO_BOHR,
            xyz_mm_train * ANGSTROM_TO_BOHR,
            s_qm_train,
            mask,
        )

        dataset_tensors = _TensorDataset(
            q_core_qm_train,
            q_val_qm_train,
            q_core_mm_train,
            q_val_mm_train,
            s_qm_train,
            s_mm_train,
            e_exrep_train,
            e_sr_corr_train,
            A_thole_qm_train,
            *mesh_data,
        )
        dataloader_tensors = _DataLoader(
            dataset_tensors, batch_size=batch_size, shuffle=False
        )

        # Now simultaneously fit A_exrep and A_sr_corr
        _logger.info("Simultaneously fitting exchange-repulsion and short-range correction parameters...")
        
        loss_instance_exrep = _ExchangeRepulsionLoss(
            emle_base, nagl, loss=_torch.nn.MSELoss()
        )
        loss_instance_exrep.eval()

        loss_instance_sr_corr = _ShortRangeCorrectionLoss(
            emle_base, nagl, loss=_torch.nn.MSELoss()
        )
        loss_instance_sr_corr.eval()

        # Only optimize A_exrep and A_sr_corr parameters
        opt_parameters_exrep_sr = [
            param for name, param in nagl.named_parameters() if "A_exrep" in name or "A_sr_corr" in name
        ]
        optimizer_exrep_sr = _torch.optim.Adam(opt_parameters_exrep_sr, lr=lr_exrep)

        for epoch in range(epochs):
            total_loss = total_rmse_exrep = total_rmse_sr_corr = 0.0
            total_max_error_exrep = total_max_error_sr_corr = 0.0
            values_exrep, target_exrep = [], []
            values_sr_corr, target_sr_corr = [], []

            for (graphs_qm_batch, _), (graphs_mm_batch, _), tensor_batch in zip(
                dataloader_qm, dataloader_mm, dataloader_tensors
            ):
                optimizer_exrep_sr.zero_grad()
                (
                    _,
                    q_val_qm_train,
                    _,
                    q_val_mm_train,
                    s_qm_train,
                    s_mm_train,
                    e_exrep_train,
                    e_sr_corr_train,
                    _,
                    *mesh_data,
                ) = tensor_batch

                # Loss for exchange-repulsion
                loss_exrep, rmse_exrep, max_error_exrep, v_exrep, t_exrep = loss_instance_exrep(
                    graphs_qm_batch,
                    graphs_mm_batch,
                    q_val_qm_train,
                    q_val_mm_train,
                    mesh_data,
                    s_qm_train,
                    s_mm_train,
                    e_exrep_train,
                )
                values_exrep.append(v_exrep.detach().cpu())
                target_exrep.append(t_exrep.detach().cpu())

                # Loss for short-range correction
                loss_sr_corr, rmse_sr_corr, max_error_sr_corr, v_sr_corr, t_sr_corr = loss_instance_sr_corr(
                    graphs_qm_batch,
                    graphs_mm_batch,
                    q_val_qm_train,
                    q_val_mm_train,
                    mesh_data,
                    s_qm_train,
                    s_mm_train,
                    e_sr_corr_train,
                    offset=0.0,
                )
                values_sr_corr.append(v_sr_corr.detach().cpu())
                target_sr_corr.append(t_sr_corr.detach().cpu())

                # Combined loss for A_exrep and A_sr_corr only
                loss = (
                    loss_weight_exrep * loss_exrep
                    + loss_weight_sr_corr * loss_sr_corr
                )

                loss.backward(retain_graph=False)
                optimizer_exrep_sr.step()

                total_loss += loss.item()
                total_rmse_exrep += rmse_exrep.item()
                total_rmse_sr_corr += rmse_sr_corr.item()
                total_max_error_exrep = max(total_max_error_exrep, max_error_exrep.item())
                total_max_error_sr_corr = max(total_max_error_sr_corr, max_error_sr_corr.item())

            values_exrep = _torch.cat(values_exrep)
            target_exrep = _torch.cat(target_exrep)
            _np.savetxt("nagl_exrep_predicted.txt", values_exrep.numpy())
            _np.savetxt("nagl_exrep_target.txt", target_exrep.numpy())

            values_sr_corr = _torch.cat(values_sr_corr)
            target_sr_corr = _torch.cat(target_sr_corr)
            _np.savetxt("nagl_sr_corr_predicted.txt", values_sr_corr.numpy())
            _np.savetxt("nagl_sr_corr_target.txt", target_sr_corr.numpy())

            rmse_exrep = _torch.sqrt(_torch.mean((values_exrep - target_exrep) ** 2)).item()
            max_error_exrep = _torch.max(_torch.abs(values_exrep - target_exrep)).item()    
            rmse_sr_corr = _torch.sqrt(_torch.mean((values_sr_corr - target_sr_corr) ** 2)).item()
            max_error_sr_corr = _torch.max(_torch.abs(values_sr_corr - target_sr_corr)).item()  
            
            if (epoch + 1) % print_every == 0:
                _logger.info(
                    f"Epoch {epoch+1}: Total Loss ={total_loss:9.4f}    "
                    f"Exrep RMSE ={rmse_exrep:9.4f}    Exrep Max Error ={max_error_exrep:9.4f}    "
                    f"SR Corr RMSE ={rmse_sr_corr:9.4f}    SR Corr Max Error ={max_error_sr_corr:9.4f}"
                )
            
        # Save the model to a file
        nagl.save("emle_nagl.pt")


    def train_c6(
        self,
        z,
        xyz,
        c6,
        train_mask=None,
        lr_c6=0.01,
        epochs=2000,
        print_every=10,
        emle_model=None,
        model_filename="emle_model_c6.mat",
        device=_torch.device("cuda"),
        dtype=_torch.float64,
    ):
        """
        Train only C6 parameters for an existing EMLE model.

        Notes
        -----
        trainer.train_c6(
            z=atomic_numbers,
            xyz=coordinates,
            c6=c6_targets,
            emle_model="path/to/existing_model.mat",
            model_filename="model_with_c6.mat",
            lr_c6=0.01,
            epochs=2000,
            device=torch.device("cuda")
        )
        
        Parameters
        ----------

        z: numpy.array, List[numpy.array], torch.Tensor, List[torch.Tensor] (N_BATCH, N_ATOMS)
            Atomic numbers.

        xyz: numpy.array, List[numpy.array], torch.Tensor, List[torch.Tensor] (N_BATCH, N_ATOM, 3)
            Atomic coordinates.

        c6: array or tensor or list of tensor/arrays of shape (N_BATCH, N_ATOMS)
            Atomic dispersion coefficients target values for training.

        train_mask: torch.Tensor(N_BATCH,), optional
            Mask for training samples. If None, all samples are used.

        lr_c6: float
            Learning rate for C6 model training.

        epochs: int
            Number of training epochs.

        print_every: int
            How often to print training progress.

        emle_model: str
            Path to an existing EMLE model file (.mat format).

        model_filename: str or None
            Filename to save the trained model. If None, the model is not saved.

        device: torch.device
            Device to use for training.

        dtype: torch.dtype
            Data type to use for training. Default is torch.float64.

        Returns
        -------

        EMLEBase
            Trained EMLE model instance with updated C6 parameters.
        """
        # Validate inputs
        if emle_model is None:
            raise ValueError("emle_model must be provided. This function only trains C6 for existing models.")
        
        if c6 is None:
            raise ValueError("c6 target values must be provided for training.")
        
        # Check input data.
        assert len(z) == len(xyz) == len(c6), "z, xyz, and c6 must have the same number of samples"

        if train_mask is None:
            train_mask = _torch.ones(len(z), dtype=_torch.bool)

        # Prepare batch data
        z = _pad_to_max(z)
        xyz = _pad_to_max(xyz)
        c6 = _pad_to_max(c6)

        z_train = z[train_mask]
        xyz_train = xyz[train_mask]
        c6_train = c6[train_mask]

        # Place on the correct device and set the data type
        z_train = z_train.to(device=device, dtype=_torch.int64)
        xyz_train = xyz_train.to(device=device, dtype=dtype)
        c6_train = c6_train.to(device=device, dtype=dtype)

        # Load existing EMLE model
        _logger.info(f"Loading existing EMLE model from: {emle_model}")
        _logger.info("Only C6 parameters will be trained...")
        emle = _EMLE(
            model=emle_model,
            device=device,
            dtype=dtype,
        )
        emle_base = emle._emle_base

        # Get molecular charges from the model for training
        species = _torch.unique(_torch.tensor(z_train[z_train > 0], device=device))
        zid_mapping = self._get_zid_mapping(species)
        zid_train = zid_mapping[z_train]
        
        # Compute q_mol for each molecule
        q_mol_train = _torch.zeros(len(z_train), device=device, dtype=dtype)
        for i in range(len(z_train)):
            mask = z_train[i] > 0
            q_mol_train[i] = _torch.sum(emle_base._q_core[zid_train[i][mask]])

        # Ensure C6 parameters are properly initialized
        if emle_base.ref_values_c6 is None:
            _logger.info("Initializing C6 reference values...")
            # Get the shape from existing reference values
            ref_shape = emle_base.ref_values_s.shape
            emle_base.ref_values_c6 = _torch.nn.Parameter(
                _torch.ones(
                    *ref_shape,
                    dtype=dtype,
                    device=device,
                )
            )
        else:
            _logger.info("Initializing C6 reference values to ones...")
            emle_base.ref_values_c6 = _torch.nn.Parameter(
                _torch.ones(
                    *emle_base.ref_values_c6.shape,
                    dtype=dtype,
                    device=device,
                )
            )
        
        if emle_base.c6_Z is None:
            _logger.info("Initializing c6_Z values...")
            emle_base.c6_Z = _torch.nn.Parameter(
                _torch.ones(
                    len(species),
                    dtype=dtype,
                    device=device,
                )
            )
        else:
            _logger.info("Initializing c6_Z to ones...")
            emle_base.c6_Z = _torch.nn.Parameter(
                _torch.ones(
                    *emle_base.c6_Z.shape,
                    dtype=dtype,
                    device=device,
                )
            )

        # Train C6 parameters
        _logger.info("Fitting c6_Z and ref_values_c6 values...")
        self._train_model(
            loss_class=self._dispersion_loss,
            opt_param_names=["c6_Z", "ref_values_c6"],
            lr=lr_c6,
            epochs=epochs,
            print_every=print_every,
            emle_base=emle_base,
            atomic_numbers=z_train,
            xyz=xyz_train,
            q_mol=q_mol_train,
            c6_target=c6_train,
        )

        # Update reference values for C6
        self._dispersion_loss._update_c6_gpr(emle_base)

        # Create the final model dictionary
        emle_model_dict = {
            "q_core": emle_base._q_core,
            "a_QEq": emle_base.a_QEq,
            "a_Thole": emle_base.a_Thole,
            "s_ref": emle_base.ref_values_s,
            "chi_ref": emle_base.ref_values_chi,
            "k_Z": emle_base.k_Z,
            "c6_Z": emle_base.c6_Z,
            "c6_ref": emle_base.ref_values_c6,
            "species": emle_base._species,
            "alpha_mode": emle_base._alpha_mode,
            "n_ref": emle_base._n_ref,
            "ref_aev": emle_base._ref_features,
            "aev_mask": emle_base._emle_aev_computer._mask,
            "zid_map": emle_base._emle_aev_computer._zid_map,
            "computer_n_species": len(emle_base._emle_aev_computer._zid_map) - 1,
        }

        if emle_base._alpha_mode == "reference":
            emle_model_dict["sqrtk_ref"] = emle_base.ref_values_sqrtk

        # Save plot data
        emle_base._alpha_mode = "species"
        _, _, _, _, c6_pred = emle_base(
            z_train.to(device=device, dtype=_torch.int64),
            xyz_train.to(device=device, dtype=dtype),
            q_mol_train,
            calc_c6=True
        )
        
        _np.savetxt("c6_target.txt", c6_train.detach().cpu().numpy().flatten())
        _np.savetxt("c6_predicted.txt", c6_pred.detach().cpu().numpy().flatten())

        # Save the model
        if model_filename is not None:
            _logger.info(f"Saving updated model to: {model_filename}")
            self._write_model_to_file(emle_model_dict, model_filename)

        return emle_base