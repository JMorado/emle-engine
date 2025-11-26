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

"""EMLE model implementation."""

__author__ = "Lester Hedges"
__email__ = "lester.hedges@gmail.com"

__all__ = ["EMLE"]

import numpy as _np
import os as _os
import scipy.io as _scipy_io
import torch as _torch
import torchani as _torchani

from torch import Tensor
from typing import Union

from . import _patches
from . import EMLEBase as _EMLEBase
from .interactions import (
    StaticElectrostatic,
    InducedElectrostatic,
    ExchangeRepulsion,
    ShortRangeCorrection,
    Dispersion,
    NullInteraction,
)

# Monkey-patch the TorchANI BuiltInModel and BuiltinEnsemble classes so that
# they call self.aev_computer using args only to allow forward hooks to work
# with TorchScript.
_torchani.models.BuiltinModel = _patches.BuiltinModel
_torchani.models.BuiltinEnsemble = _patches.BuiltinEnsemble
HARTREE_TO_KCALMOL = 627.5094740631


try:
    import NNPOps as _NNPOps

    _NNPOps.OptimizedTorchANI = _patches.OptimizedTorchANI

    _has_nnpops = True
except:
    _has_nnpops = False


# _torch.autograd.set_detect_anomaly(True)

class EMLE(_torch.nn.Module):
    """
    Torch model for predicting static and induced EMLE energy components.
    """

    # Class attributes.

    # A flag for type inference. TorchScript doesn't support inheritance, so
    # we need to check for an object of type torch.nn.Module, and that it has
    # the required _is_emle attribute.
    _is_emle = True

    # Store the expected path to the resources directory.
    _resource_dir = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), "..", "resources"
    )

    # Create the name of the default model file.
    _default_model = _os.path.join(_resource_dir, "emle_qm7_aev.mat")

    # Store the list of supported species.
    _species = [1, 6, 7, 8, 16]

    def __init__(
        self,
        model=None,
        method="electrostatic",
        alpha_mode="species",
        atomic_numbers=None,
        qm_charge=0,
        mm_charges=None,
        device=None,
        dtype=None,
        create_aev_calculator=True,
        nagl_params: dict = None,
        dispersion_mode: str = None,
        cp_mode: str = None,
        include_exrep: bool = False,
        include_sr_corr: bool = False,
        *args, **kwargs
    ):
        """
        Constructor.

        Parameters
        ----------

        model: str
            Path to a custom EMLE model parameter file. If None, then the
            default model for the specified 'alpha_mode' will be used.

        method: str
            The desired embedding method. Options are:
                "electrostatic":
                    Full ML electrostatic embedding.
                "mechanical":
                    ML predicted charges for the core, but zero valence charge.
                "nonpol":
                    Non-polarisable ML embedding. Here the induced component of
                    the potential is zeroed.
                "mm":
                    MM charges are used for the core charge and valence charges
                    are set to zero. If this option is specified then the user
                    should also specify the MM charges for atoms in the QM
                    region.

        alpha_mode: str
            How atomic polarizabilities are calculated.dispersion
                "species":
                    one volume scaling factor is used for each species
                "reference":
                    scaling factors are obtained with GPR using the values learned
                    for each reference environment

        atomic_numbers: List[int], Tuple[int], numpy.ndarray, torch.Tensor
            Atomic numbers for the QM region. This allows use of optimised AEV
            symmetry functions from the NNPOps package. Only use this option
            if you are using a fixed QM region, i.e. the same QM region for each
            evalulation of the module.

        qm_charge: int
            The charge on the QM region. This can also be passed when calling
            the forward method. The non-default value will take precendence.

        mm_charges: List[float], Tuple[Float], numpy.ndarray, torch.Tensor
            List of MM charges for atoms in the QM region in units of mod
            electron charge. This is required if the 'mm' method is specified.

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for the models floating point tensors.

        create_aev_calculator: bool
            Whether to create an AEV calculator instance. This can be set
            to False for derived classes that already have an AEV calculator,
            e.g. ANI2xEMLE. In that case, it's possible to hook the AEV
            calculator to avoid duplicating the computation.

        nagl_model: str
            Path to a NAGL model file for computing exchange-repulsion and
            short-range correction parameters. Also should contain the valence
            widths 's' used for charge penetration.

        nagl_params : dict
            A dictionary containing the following keys:
                'A_exrep_qm': torch.Tensor
                'A_sr_corr_qm': torch.Tensor
                's_qm': torch.Tensor
                'A_exrep_mm': torch.Tensor
                'A_sr_corr_mm': torch.Tensor
                's_mm': torch.Tensor
            These parameters can be precomputed using the NAGL model to avoid
            redundant calculations during the forward pass.

        dispersion_mode: str
            The dispersion interaction mode to use. Options are:
                "lj":
                    Lennard-Jones 12-6 potential.
                "c6":
                    C6/R^6 dispersion potential with Tang-Toennies damping.
                None
                    No dispersion interactions are computed.

        cp_mode: str
            Charge penetration mode:
                "gaussian":
                    Use Gaussian charge distributions for charge penetration.
                "slater":
                    Use Slater valence shells + point core charges for charge penetration.
                None
                    No charge penetration interactions are computed.

        include_exrep: bool
            Whether to include the exchange-repulsion interaction.

        include_sr_corr: bool
            Whether to include the short-range correction interaction.
        """

        # Call the base class constructor.
        super().__init__()

        from .._resources import _fetch_resources

        # Fetch or update the resources.
        if model is None:
            _fetch_resources()

        if method is None:
            method = "electrostatic"
        if not isinstance(method, str):
            raise TypeError("'method' must be of type 'str'")
        method = method.lower().replace(" ", "")
        if method not in ["electrostatic", "mechanical", "nonpol", "mm"]:
            raise ValueError(
                "'method' must be 'electrostatic', 'mechanical', 'nonpol', or 'mm'"
            )
        self._method = method

        if atomic_numbers is not None:
            if isinstance(atomic_numbers, (_np.ndarray, _torch.Tensor)):
                atomic_numbers = atomic_numbers.tolist()
            if not isinstance(atomic_numbers, (tuple, list)):
                raise TypeError(
                    "'atomic_numbers' must be of type 'list', 'tuple', or 'numpy.ndarray'"
                )
            if not all(isinstance(a, int) for a in atomic_numbers):
                raise TypeError(
                    "All elements of 'atomic_numbers' must be of type 'int'"
                )
            if not all(a > 0 for a in atomic_numbers):
                raise ValueError(
                    "All elements of 'atomic_numbers' must be greater than zero"
                )
        self._atomic_numbers = atomic_numbers

        if not isinstance(qm_charge, int):
            raise TypeError("'qm_charge' must be of type 'int'")
        self._qm_charge = qm_charge

        if method == "mm":
            if mm_charges is None:
                raise ValueError("MM charges must be provided for the 'mm' method")
            if isinstance(mm_charges, (list, tuple)):
                mm_charges = _np.array(mm_charges)
            elif isinstance(mm_charges, _torch.Tensor):
                mm_charges = mm_charges.cpu().numpy()
            if not isinstance(mm_charges, _np.ndarray):
                raise TypeError("'mm_charges' must be of type 'numpy.ndarray'")
            if mm_charges.dtype != _np.float64:
                raise ValueError("'mm_charges' must be of type 'numpy.float64'")
            if mm_charges.ndim != 1:
                raise ValueError("'mm_charges' must be a 1D array")

        if model is not None:
            if not isinstance(model, str):
                raise TypeError("'model' must be of type 'str'")

            # Convert to an absolute path.
            abs_model = _os.path.abspath(model)

            if not _os.path.isfile(abs_model):
                raise IOError(f"Unable to locate EMLE embedding model file: '{model}'")
            self._model = abs_model
        else:
            # Set to None as this will be used in any calculator configuration.
            self._model = None

            # Use the default model.
            model = self._default_model

            # Use the default species.
            species = self._species

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

        # Load the model parameters.
        try:
            params = _scipy_io.loadmat(model, squeeze_me=True)
        except:
            if model is self._default_model and not _os.path.isfile(model):
                raise IOError(
                    f"Unable to locate default EMLE embedding model file: '{model}'. "
                    "Please ensure that the resources are installed correctly. For "
                    "details, see: https://github.com/chemle/emle-models"
                )
            raise IOError(f"Unable to load model parameters from: '{model}'")

        q_core = _torch.tensor(params["q_core"], dtype=dtype, device=device)
        aev_mask = _torch.tensor(params["aev_mask"], dtype=_torch.bool, device=device)
        n_ref = _torch.tensor(params["n_ref"], dtype=_torch.int64, device=device)
        ref_features = _torch.tensor(params["ref_aev"], dtype=dtype, device=device)

        emle_params = {
            "a_QEq": _torch.tensor(params["a_QEq"], dtype=dtype, device=device),
            "a_Thole": _torch.tensor(params["a_Thole"], dtype=dtype, device=device),
            "a_Gauss": (
                _torch.tensor(params["a_Gauss"], dtype=dtype, device=device)
                if "a_Gauss" in params
                else None
            ),
            "ref_values_s": _torch.tensor(params["s_ref"], dtype=dtype, device=device),
            "ref_values_chi": _torch.tensor(
                params["chi_ref"], dtype=dtype, device=device
            ),
            "k_Z": _torch.tensor(params["k_Z"], dtype=dtype, device=device),
            "c6_Z": (
                _torch.tensor(params["c6_Z"], dtype=dtype, device=device)
                if "c6_Z" in params
                else None
            ),
            "sqrtk_ref": (
                _torch.tensor(params["sqrtk_ref"], dtype=dtype, device=device)
                if "sqrtk_ref" in params
                else None
            ),
            "ref_values_c6": (
                _torch.tensor(params["c6_ref"], dtype=dtype, device=device)
                if "c6_ref" in params
                else None
            ),
        }

        if method == "mm":
            q_core_mm = _torch.tensor(mm_charges, dtype=dtype, device=device)
        else:
            q_core_mm = _torch.empty(0, dtype=dtype, device=device)

        # Store the current device.
        self._device = device

        # Register constants as buffers.
        self.register_buffer("_q_core_mm", q_core_mm)

        if not isinstance(create_aev_calculator, bool):
            raise TypeError("'create_aev_calculator' must be of type 'bool'")

        # Create an AEV calculator to perform the feature calculations.
        from ._emle_aev_computer import EMLEAEVComputer

        if create_aev_calculator:
            num_species = params.get("computer_n_species", len(n_ref))
            emle_aev_computer = EMLEAEVComputer(
                mask=aev_mask, num_species=num_species, device=device, dtype=dtype
            )

            # Optimise the AEV computer using NNPOps if available.
            if _has_nnpops and atomic_numbers is not None:
                try:
                    import ase
                    from torchani import SpeciesConverter

                    # Work out the species.
                    species = [ase.Atom(i).symbol for i in atomic_numbers]

                    # Create a species converter.
                    species_converter = SpeciesConverter(species).to(device)

                    atomic_numbers = _torch.tensor(
                        atomic_numbers, dtype=_torch.int64, device=device
                    )

                    atomic_numbers = atomic_numbers.reshape(1, *atomic_numbers.shape)
                    emle_aev_computer._aev_computer = (
                        _NNPOps.SymmetryFunctions.TorchANISymmetryFunctions(
                            species_converter,
                            emle_aev_computer._aev_computer,
                            atomic_numbers,
                        )
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Unable to create optimised AEVComputer using NNPOps."
                    ) from e
        else:
            emle_aev_computer = EMLEAEVComputer(
                is_external=True,
                mask=aev_mask,
                zid_map=params.get("zid_map"),
                device=device,
            )

        # Create empty attributes to store the inputs to the forward method.
        self._atomic_numbers = _torch.empty(0, dtype=_torch.int64, device=device)
        self._charges_mm = _torch.empty(0, dtype=dtype, device=device)
        self._xyz_qm = _torch.empty(0, 3, dtype=dtype, device=device)
        self._xyz_mm = _torch.empty(0, 3, dtype=dtype, device=device)

        # Add NAGL parameters if provided.
        if nagl_params is not None:
            emle_params.update(nagl_params)

        # Create the base EMLE model.
        self._emle_base = _EMLEBase(
            emle_params,
            n_ref,
            ref_features,
            q_core,
            emle_aev_computer=emle_aev_computer,
            alpha_mode=alpha_mode if alpha_mode is not None else "species",
            species=params.get("species", self._species),
            device=device,
            dtype=dtype,
        )

        # Initialize interaction modules based on user options.
        self._static = StaticElectrostatic(
            emle_base=self._emle_base,
            method=method,
            cp_mode=cp_mode,
            q_core_mm=q_core_mm,
            device=device,
            dtype=dtype,
        )

        if method == "electrostatic":
            self._induced = InducedElectrostatic(
                emle_base=self._emle_base,
                alpha_mode=alpha_mode,
                device=device,
                dtype=dtype,
            )
        else:
            self._induced = NullInteraction(device=device, dtype=dtype)

        if nagl_params:
            self._exrep = (
                ExchangeRepulsion(emle_base=self._emle_base, device=device, dtype=dtype)
                if include_exrep
                else NullInteraction(device=device, dtype=dtype)
            )
            self._sr_corr = (
                ShortRangeCorrection(
                    emle_base=self._emle_base, device=device, dtype=dtype
                )
                if include_sr_corr
                else NullInteraction(device=device, dtype=dtype)
            )
            self._disp = (
                Dispersion(
                    emle_base=self._emle_base,
                    mode=dispersion_mode,
                    device=device,
                    dtype=dtype,
                )
                if dispersion_mode
                else NullInteraction(device=device, dtype=dtype)
            )
        else:
            self._exrep = NullInteraction(device=device, dtype=dtype)
            self._sr_corr = NullInteraction(device=device, dtype=dtype)
            self._disp = NullInteraction(device=device, dtype=dtype)

        self._alpha_mode = alpha_mode

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion on the model.
        """
        self._q_core_mm = self._q_core_mm.to(*args, **kwargs)
        self._emle_base = self._emle_base.to(*args, **kwargs)

        # Move interaction modules
        self._static = self._static.to(*args, **kwargs)
        self._induced = self._induced.to(*args, **kwargs)
        self._exrep = self._exrep.to(*args, **kwargs)
        self._sr_corr = self._sr_corr.to(*args, **kwargs)
        self._disp = self._disp.to(*args, **kwargs)

        # Check for a device type in args and update the device attribute.
        for arg in args:
            if isinstance(arg, _torch.device):
                self._device = arg
                break

        return self

    def cuda(self, **kwargs):
        """
        Move all model parameters and buffers to CUDA memory.
        """
        self._q_core_mm = self._q_core_mm.cuda(**kwargs)
        self._emle_base = self._emle_base.cuda(**kwargs)

        self._static = self._static.cuda(**kwargs)
        self._induced = self._induced.cuda(**kwargs)
        self._exrep = self._exrep.cuda(**kwargs)
        self._sr_corr = self._sr_corr.cuda(**kwargs)
        self._disp = self._disp.cuda(**kwargs)

        self._device = self._q_core_mm.device

        return self

    def cpu(self, **kwargs):
        """
        Move all model parameters and buffers to CPU memory.
        """
        self._q_core_mm = self._q_core_mm.cpu(**kwargs)
        self._emle_base = self._emle_base.cpu()

        self._static = self._static.cpu(**kwargs)
        self._induced = self._induced.cpu(**kwargs)
        self._exrep = self._exrep.cpu(**kwargs)
        self._sr_corr = self._sr_corr.cpu(**kwargs)
        self._disp = self._disp.cpu(**kwargs)

        self._device = self._q_core_mm.device

        return self

    def double(self):
        """
        Casts all floating point model parameters and buffers to float64 precision.
        """
        self._q_core_mm = self._q_core_mm.double()
        self._emle_base = self._emle_base.double()

        self._static = self._static.double()
        self._induced = self._induced.double()
        self._exrep = self._exrep.double()
        self._sr_corr = self._sr_corr.double()
        self._disp = self._disp.double()

        return self

    def float(self):
        """
        Casts all floating point model parameters and buffers to float32 precision.
        """
        self._q_core_mm = self._q_core_mm.float()
        self._emle_base = self._emle_base.float()

        self._static = self._static.float()
        self._induced = self._induced.float()
        self._exrep = self._exrep.float()
        self._sr_corr = self._sr_corr.float()
        self._disp = self._disp.float()

        return self

    def forward(
        self,
        atomic_numbers: Tensor,
        charges_mm: Tensor,
        xyz_qm: Tensor,
        xyz_mm: Tensor,
        qm_charge: Union[int, Tensor] = 0,
        idx_mm: Tensor = None,
    ) -> Tensor:
        """
        Computes the static and induced EMLE energy components.

        Parameters
        ----------

        atomic_numbers: torch.Tensor (N_QM_ATOMS,) or (BATCH, N_QM_ATOMS)
            Atomic numbers of QM atoms.

        charges_mm: torch.Tensor (MAX_MM_ATOMS,) or (BATCH, MAX_MM_ATOMS)
            MM point charges in atomic units.

        xyz_qm: torch.Tensor (N_QM_ATOMS, 3) or (BATCH, N_QM_ATOMS, 3)
            Positions of QM atoms in Angstrom.

        xyz_mm: torch.Tensor (N_MM_ATOMS, 3) or (BATCH, N_MM_ATOMS, 3)
            Positions of MM atoms in Angstrom.

        qm_charge: int or torch.Tensor (BATCH,)
            The charge on the QM region.

        idx_mm: torch.Tensor (N_MM_ATOMS,)
            Indices of MM atoms in the full system. This is required to
            select the correct NAGL parameters for the MM atoms.

        Returns
        -------

        result: torch.Tensor (5,) or (5, BATCH)
            The EMLE energy components in Hartree: [static, induced, exchange-repulsion, short-range correction, disperion].
        """
        # Store the inputs as internal attributes.
        self._atomic_numbers = atomic_numbers
        self._charges_mm = charges_mm
        self._xyz_qm = xyz_qm
        self._xyz_mm = xyz_mm

        # Batch the inputs if necessary.
        if self._atomic_numbers.ndim == 1:
            self._atomic_numbers = self._atomic_numbers.unsqueeze(0)
            self._charges_mm = self._charges_mm.unsqueeze(0)
            self._xyz_qm = self._xyz_qm.unsqueeze(0)
            self._xyz_mm = self._xyz_mm.unsqueeze(0)

        batch_size = self._atomic_numbers.shape[0]

        # Ensure qm_charge is a tensor and repeat for batch size if necessary
        if isinstance(qm_charge, int):
            qm_charge = _torch.full(
                (batch_size,),
                qm_charge if qm_charge != 0 else self._qm_charge,
                dtype=_torch.int64,
                device=self._device,
            )
        elif isinstance(qm_charge, _torch.Tensor):
            if qm_charge.ndim == 0:
                qm_charge = qm_charge.repeat(batch_size).to(self._device)

        # If there are no point charges, return zeros.
        if xyz_mm.shape[1] == 0:
            return _torch.zeros(
                2, batch_size, dtype=self._xyz_qm.dtype, device=self._xyz_qm.device
            )
       
        if idx_mm is not None:
            if not isinstance(idx_mm, _torch.Tensor):
                idx_mm = _torch.as_tensor(idx_mm, dtype=_torch.int64, device=self._device)
            if idx_mm.ndim == 1:
                idx_mm = idx_mm.unsqueeze(0)  
            max_n_mm_atoms = (idx_mm > 0).sum(dim=1).max()
            self._charges_mm = self._charges_mm[:, :max_n_mm_atoms]
            self._xyz_mm = self._xyz_mm[:, :max_n_mm_atoms, :]
       
        # Get the base data.
        #mask, species_id, aev = self._emle_base.get_aev_data(
        #    self._atomic_numbers, self._xyz_qm
        #)

        # Convert coordinates to Bohr (needed for r_data computation)
        ANGSTROM_TO_BOHR = 1.8897261258369282
        xyz_qm_bohr = self._xyz_qm * ANGSTROM_TO_BOHR
        xyz_mm_bohr = self._xyz_mm * ANGSTROM_TO_BOHR

        """
        # Compute properties required by several interactions.
        r_data = self._emle_base._get_r_data(xyz_qm_bohr, mask)
        s = self._emle_base.get_s(aev, species_id)
        chi = self._emle_base.get_chi(aev, species_id)
        q_core = self._emle_base.get_q_core(species_id, mask)
        q = self._emle_base.get_q_val(s, chi, qm_charge, r_data, mask)
        q_val = q - q_core
        A_thole = (
            self._emle_base.get_A_thole(
                xyz_qm_bohr, s, q_val, species_id, aev, mask.squeeze(-1), r_data
            )
            if self._method == "electrostatic"
            else None
        )
        """
        s, q_core, q_val, A_thole = self._emle_base.forward(self._atomic_numbers, self._xyz_qm, qm_charge)

        # Create mesh data for interactions.
        mask = self._atomic_numbers > 0
        mask_3d = mask.unsqueeze(-1)
        mesh_data = self._emle_base._get_mesh_data(xyz_qm_bohr, xyz_mm_bohr, s, mask_3d)

        # Calculate all energy components using interaction modules.
        E_static = self._static.forward(q_core, q_val, self._charges_mm, mesh_data, s, idx_mm)
        E_induced = self._induced.forward(A_thole, self._charges_mm, s, mesh_data, mask_3d)
        E_exrep = self._exrep(q_val, self._charges_mm, mesh_data, s, idx_mm)
        E_sr_corr = self._sr_corr(q_val, self._charges_mm, mesh_data, s, idx_mm)
        E_disp = self._disp()

        return _torch.stack((E_static, E_induced, E_exrep, E_sr_corr, E_disp), dim=0)
