"""AEVCalculator class for calculating AEV feature vectors using the ANI2x model."""
import numpy as _np
import torch as _torch
import torchani as _torchani

from ._utils import pad_to_max

ANGSTROM_TO_BOHR = 1.8897261258369282 #TODO: keep constants in one place

# From ANI-2x
DEFAULT_HYPERS_DICT = {
    "Rcr": _np.array(5.1000e+00),
    "Rca": _np.array(3.5000e+00),
    "EtaR": _np.array([1.9700000e+01]),
    "ShfR": _np.array([8.0000000e-01, 1.0687500e+00, 1.3375000e+00,
                      1.6062500e+00, 1.8750000e+00, 2.1437500e+00,
                      2.4125000e+00, 2.6812500e+00, 2.9500000e+00,
                      3.2187500e+00, 3.4875000e+00, 3.7562500e+00,
                      4.0250000e+00, 4.2937500e+00, 4.5625000e+00,
                      4.8312500e+00]),
    "Zeta": _np.array([1.4100000e+01]),
    "ShfZ": _np.array([3.9269908e-01, 1.1780972e+00, 1.9634954e+00,
                      2.7488936e+00]),
    "EtaA": _np.array([1.2500000e+01]),
    "ShfA": _np.array([8.0000000e-01, 1.1375000e+00, 1.4750000e+00,
                      1.8125000e+00, 2.1500000e+00, 2.4875000e+00,
                      2.8250000e+00, 3.1625000e+00])
}


def get_default_hypers(device, dtype):
    return {k: _torch.tensor(v, device=device, dtype=dtype)
            for k, v in DEFAULT_HYPERS_DICT.items()}


class EMLEAEVComputer(_torch.nn.Module):
    """
    Wrapper for AEVCalculator from torchani
    (not a subclass to make sure it works with TorchScript)
    """
    def __init__(self, num_species=7, hypers=None,
                 mask=None, external=False, zid_map=None,
                 device=None, dtype=None):
        """
        num_species: int
            number of supported species
        Hypers: dict
            hyperparameters for wrapped AEVComputer
        mask: torch.BoolTensor
            mask for the features returned from wrapped AEVComputer
        external: bool
            Whether the features are calculated externally
        zid_map: dict
            map from zid provided here to the ones passed to AEVComputer
        device: torch.device
            The device on which to run the model.
        dtype: torch.dtype
            The data type to use for the models floating point tensors.
        """
        super().__init__()

        if device is not None:
            if not isinstance(device, _torch.device):
                raise TypeError("'device' must be of type 'torch.device'")
        else:
            device = _torch.get_default_device()
        self._device = device

        self._external = external

        # Validate the AEV mask.
        if mask is not None:
            if not isinstance(mask, _torch.Tensor):
                raise TypeError("'mask' must be of type 'torch.Tensor'")
            if len(mask.shape) != 1:
                raise ValueError("'mask' must be a 1D tensor")
            if not mask.dtype == _torch.bool:
                raise ValueError("'mask' must have dtype 'torch.bool'")
        self._mask = mask

        self._aev = None

        if not external:
            hypers = hypers or get_default_hypers(device, dtype)
            self._aev_computer = _torchani.AEVComputer(**hypers,
                                                       num_species=num_species)

        if not zid_map:
            zid_map = {i: i for i in range(num_species)}
        self._zid_map = - _torch.ones(num_species + 1, dtype=_torch.int,
                                      device=device)
        for self_atom_zid, aev_atom_zid in zid_map.items():
            self._zid_map[self_atom_zid] = aev_atom_zid

    def forward(self, zid, xyz):
        """
        zid: (N_BATCH, MAX_N_ATOMS)
        xyz: (N_BATCH, MAX_N_ATOMS, 3)
        """
        if not self._external:
            zid_aev = self._zid_map[zid]
            self._aev = self._aev_computer((zid_aev, xyz))[1]

        norm = _torch.linalg.norm(self._aev, dim=2, keepdims=True)
        return self._apply_mask(self._aev / norm)

    def _apply_mask(self, aev):
        return aev[:, :, self._mask] if self._mask is not None else aev

    def to(self, *args, **kwargs):
        if self._aev_computer:
            self._aev_computer = self._aev_computer.to(*args, **kwargs)
        if self._mask:
            self._mask = self._mask.to(*args, **kwargs)
        self._zid_map = self._zid_map.to(*args, **kwargs)
        return self

    def cuda(self, **kwargs):
        if self._aev_computer:
            self._aev_computer = self._aev_computer.cuda(**kwargs)
        if self._mask:
            self._mask = self._mask.cuda(**kwargs)
        self._zid_map = self._zid_map.cuda(**kwargs)
        return self

    def cpu(self, **kwargs):
        if self._aev_computer:
            self._aev_computer = self._aev_computer.cpu(**kwargs)
        if self._mask:
            self._mask = self._mask.cpu(**kwargs)
        self._zid_map = self._zid_map.cpu(**kwargs)
        return self

    def double(self):
        if self._aev_computer:
            self._aev_computer = self._aev_computer.double()
        return self

    def float(self):
        if self._aev_computer:
            self._aev_computer = self._aev_computer.float()
        return self


class AEVCalculator:
    """
    Calculates AEV feature vectors using the ANI2x model.

    Parameters
    ----------
    device : str or torch.device
        Device to use for the calculations. Default is "cuda" if available, otherwise "cpu".

    Attributes
    ----------
    device : str or torch.device
        Device used for the calculations.
    model : torchani.models.ANIModel
        ANI model used for the calculations.
    aev_computer : torchani.AEVComputer
        AEV computer used for the calculations.
    """

    def __init__(self, device=None):
        self._device = device or _torch.device(
            "cuda" if _torch.cuda.is_available() else "cpu"
        )
        self._model = _torchani.models.ANI2x().to(self.device)
        self._aev_computer = self._model.aev_computer

    def _get_aev(self, zid, xyz):
        """
        Computes the AEVs for given atomic numbers and positions.

        Parameters
        ----------
        zid : torch.Tensor(N_ATOMS)
            Atomic numbers of the atoms.
        xyz : torch.Tensor(N_ATOMS, 3)
            Cartesian coordinates of the atoms.

        Returns
        -------
        np.ndarray
            AEV feature vectors.
        """
        natoms = sum(zid > -1)
        zid = zid[:natoms].to(self._device)
        xyz = xyz[:natoms].to(self._device)
        result = self.aev_computer.forward((zid, xyz))[1][0]
        return result.cpu().numpy()

    def calculate_aev(self, z, xyz, species):
        """
        Calculates the AEV feature vectors for all molecules.

        Parameters
        ----------
        z : torch.Tensor(N_BATCH, MAX_N_ATOMS)
            Atomic numbers for all molecules.
        xyz : torch.Tensor(N_BATCH, MAX_N_ATOMS, 3)
            Cartesian coordinates for all molecules.

        Returns
        -------
        torch.Tensor(N_BATCH, MAX_N_ATOMS, AEV_DIM)
            AEV feature vectors for all molecules.
        """
        # Generate the species ID mapping
        _species_id = _torch.zeros(max(species) + 1, dtype=_torch.int)
        for i, z in enumerate(species):
            _species_id[z] = i
        _species_id[0] = -1

        # Calculate AEVs
        aev_full = pad_to_max(
            [
                self._get_aev(_species_id[z_mol], xyz_mol / ANGSTROM_TO_BOHR)
                for z_mol, xyz_mol in zip(z, xyz)
            ]
        )
        aev_mask = _torch.sum(aev_full.reshape(-1, aev_full.shape[-1]) ** 2, dim=0) > 0
        aev = aev_full[:, :, aev_mask]
        aev_norm = aev / _torch.linalg.norm(aev, dim=2, keepdims=True)

        return aev_norm
