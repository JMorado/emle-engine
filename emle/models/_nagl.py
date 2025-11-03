import torch as _torch
from typing import List, Dict, Union

try:
    from openff.nagl.features import atoms
    from openff.nagl.config.model import (
        ConvolutionLayer,
        ConvolutionModule,
    )
    from openff.nagl import GNNModel
    from openff.nagl.config.model import ModelConfig
    from openff.nagl.config.model import (
        ForwardLayer,
        ReadoutModule,
    )
    from openff.nagl.nn import DGLMoleculeDataset, DGLMoleculeDataLoader
    from openff.toolkit import Molecule
except ImportError:
    raise ImportError(
        "Failed to import NAGL modules. Ensure that the openff-nagl package is installed."
    )

try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
except ImportError:
    raise ImportError(
        "Failed to import RDKit modules. Ensure that the rdkit package is installed."
    )


class NAGLEMLE(_torch.nn.Module):
    """
    Torch model for predicting EMLE parameters using NAGL (GraphSage GNN).

    Parameters
    ----------
    species: list[int]
        List of atomic numbers supported by the model.
    properties: list[str]
        List of properties to predict.
        Allowed properties are: "A_exrep", "A_sr_corr".
    n_conv_layers: int
        Number of convolutional layers in the GNN.
    hidden_dim: int
        Dimensionality of the hidden layers.
    n_ffnn_layers: int
        Number of feedforward layers in the GNN.
    """

    ALLOWED_PROPERTIES: List[str] = ["A_exrep", "A_sr_corr", "s"]
    ATOMIC_NUMBER_TO_SYMBOL: Dict[int, str] = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}

    def __init__(
        self,
        species: List[int],
        properties: List[str],
        n_conv_layers: int = 3,
        hidden_dim: int = 128,
        n_ffnn_layers: int = 4,
        model_filepath: str = None,
    ):
        super().__init__()

        # check that species are a list of int
        if not isinstance(species, list):
            raise TypeError(f"Expected list for species, got {type(species)}")
        for s in species:
            if not isinstance(s, int):
                raise TypeError(f"Expected int for species element, got {type(s)}")

        for name, value in {
            "n_conv_layers": n_conv_layers,
            "hidden_dim": hidden_dim,
            "n_ffnn_layers": n_ffnn_layers,
        }.items():
            if not isinstance(value, int):
                raise TypeError(f"Expected int for {name}, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"Expected positive int for {name}, got {value}")

        # Check that properties are a list of str and that the property is allowed
        if not isinstance(properties, list):
            raise TypeError(f"Expected list for properties, got {type(properties)}")
        for p in properties:
            if not isinstance(p, str):
                raise TypeError(f"Expected str for properties element, got {type(p)}")
            if p not in self.ALLOWED_PROPERTIES:
                raise ValueError(
                    f"Property '{p}' is not allowed. Allowed properties are: {self.ALLOWED_PROPERTIES}"
                )

        self._species = species
        self._properties = properties
        self._hidden_dim = hidden_dim
        self._n_conv_layers = n_conv_layers
        self._n_ffnn_layers = n_ffnn_layers
        self._atom_features = self._build_atom_features(species)
        if model_filepath is not None:
            self.load(model_filepath)
        else:
            self._gnn_model = self._build_gnn_model()
        self._dataloaders = {}

    def _build_atom_features(self, species):
        """Build atom features for the GNN model."""
        return (
            atoms.AtomicElement(
                categories=[self.ATOMIC_NUMBER_TO_SYMBOL[Z] for Z in species]
            ),
            atoms.AtomConnectivity(),
            atoms.AtomAverageFormalCharge(),
            atoms.AtomHybridization(),
            atoms.AtomInRingOfSize(ring_size=3),
            atoms.AtomInRingOfSize(ring_size=4),
            atoms.AtomInRingOfSize(ring_size=5),
            atoms.AtomInRingOfSize(ring_size=6),
        )

    def _build_gnn_model(self):
        """Helper function to build the GNN model."""
        # Convolution module (aggregator)
        conv_layer = ConvolutionLayer(
            hidden_feature_size=self._hidden_dim,
            aggregator_type="mean",
            activation_function="ReLU",
            dropout=0.0,
        )

        conv_module = ConvolutionModule(
            architecture="SAGEConv",
            layers=[conv_layer] * self._n_conv_layers,
        )

        readout_layer = ForwardLayer(
            hidden_feature_size=self._hidden_dim,
            activation_function="ReLU",
            dropout=0.0,
        )

        # Build the readout layers for each property
        output_layer = ForwardLayer(
            hidden_feature_size=1,
            activation_function="Identity",
            dropout=0.0,
        )

        readouts = {
            prop: ReadoutModule(
                pooling="atoms",
                layers=[readout_layer] * self._n_ffnn_layers + [output_layer],
                postprocess=None,
            )
            for prop in self._properties
        }

        config = ModelConfig(
            version="0.1",
            atom_features=self._atom_features,
            bond_features=[],
            convolution=conv_module,
            readouts=readouts,
        )

        return GNNModel(config=config)

    def train(self, *args, **kwargs):
        return self._gnn_model.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        return self._gnn_model.eval(*args, **kwargs)

    def to(self, *args, **kwargs):
        self._gnn_model.to(*args, **kwargs)
        return self

    def save(self, filepath: str):
        """
        Save the model to a file.

        Parameters
        ----------
        filepath : str
            The path to the file where the model will be saved.
        """
        self._gnn_model.save(filepath)

    def load(self, filepath: str):
        """
        Load the model from a file.

        Parameters
        ----------
        filepath : str
            The path to the file from which the model will be loaded.
        """
        self._gnn_model = GNNModel.load(filepath)

    def forward(self, batch):
        preds = self._gnn_model.forward(batch.to(self._gnn_model.device))
        natoms_mol = list(int(n) for n in batch.n_atoms_per_molecule)
        for key, val in preds.items():
            preds_split = _torch.split(val, natoms_mol)
            padded_preds = _torch.nn.utils.rnn.pad_sequence(
                preds_split, batch_first=True, padding_value=0.0
            ).to(val.device)
            preds[key] = padded_preds.squeeze(-1)
        return preds

    def create_dataloader(
        self,
        offmols: List[Molecule],
        device: _torch.device,
        batch_size: int = 1024,
        mm: bool = False,
        *args,
        **kwargs,
    ) -> DGLMoleculeDataLoader:
        """
        Create a DGL data loader for the given OpenFF molecules. Required for batched inference.

        Parameters
        ----------
        offmols : list[Molecule]
            A list of OpenFF Molecule instances.

        device : torch.device
            The device to which the data should be moved.

        batch_size : int
            The batch size to use for the data loader.

        mm : bool
            Whether to create a data loader for MM or QM molecules.

        Returns
        -------
        DGLMoleculeDataLoader
            A data loader for the specified molecules.
        """
        key = "mm" if mm else "qm"
        dataset = DGLMoleculeDataset.from_openff(
            offmols, atom_features=self._atom_features
        )
        dataloader = DGLMoleculeDataLoader(
            dataset, batch_size=batch_size, *args, **kwargs
        )
        self._dataloaders[key] = dataloader
        return dataloader

    @staticmethod
    def create_openff_mol(
        atomic_numbers: _torch.Tensor,
        xyz: _torch.Tensor,
        charge: Union[_torch.Tensor, int],
    ):
        """
        Create OpenFF Molecule instances from atomic numbers, coordinates, and charge.

        Parameters
        ----------
        atomic_numbers : _torch.Tensor(N_BATCH, N_ATOMS,)
            A tensor containing atomic numbers.
        xyz : _torch.Tensor(N_BATCH, N_ATOMS, 3)
            A tensor containing atomic coordinates.
        charge : _torch.Tensor(N_BATCH,) or int
            A tensor containing the molecular charge or a single integer.

        Returns
        -------
        list[Molecule]
            A list of OpenFF Molecules.
        """
        # Expand dims if needed
        if atomic_numbers.ndim == 1:
            atomic_numbers = atomic_numbers.unsqueeze(0)
            xyz = xyz.unsqueeze(0)
            if isinstance(charge, _torch.Tensor):
                charge = charge.unsqueeze(0)
            else:
                charge = (
                    _torch.ones((xyz.size(0),), dtype=xyz.dtype, device=xyz.device)
                    * charge
                )

        atomic_numbers = atomic_numbers.detach().cpu().numpy()
        xyz = xyz.detach().cpu().numpy()
        charge = charge.detach().cpu().numpy()

        off_mols = []
        for z, pos, q in zip(atomic_numbers, xyz, charge):
            valid_mask = z > 0
            z_valid = z[valid_mask]
            pos_valid = pos[valid_mask]
            # Build XYZ string block
            xyz_lines = [
                f"{NAGLEMLE.ATOMIC_NUMBER_TO_SYMBOL[int(zi)]} {x:.6f} {y:.6f} {z:.6f}"
                for zi, (x, y, z) in zip(z_valid, pos_valid)
            ]
            xyz_block = f"{len(z_valid)}\n\n" + "\n".join(xyz_lines)

            mol = Chem.MolFromXYZBlock(xyz_block)
            if mol is None:
                raise ValueError("Failed to create RDKit molecule from XYZ block")
            rdDetermineBonds.DetermineBonds(mol, charge=int(q))
            off_mols.append(Molecule.from_rdkit(mol))

        return off_mols
