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

"""Null interaction that returns zero energy."""

__author__ = "Joao Morado"
__email__ = "joaomorado@gmail.com"

__all__ = ["NullInteraction"]

import torch as _torch
from ._base import BaseInteraction


class NullInteraction(BaseInteraction):
    """
    Null interaction that always returns zero energy.

    Used as a placeholder when an interaction is disabled, avoiding
    the need for if-clauses in the forward pass.
    """

    def __init__(self, device=None, dtype=None):
        """
        Constructor.

        Parameters
        ----------

        device: torch.device
            The device on which to run the model.

        dtype: torch.dtype
            The data type to use for floating point tensors.
        """
        super().__init__(device=device, dtype=dtype)

    def forward(self, *args, **kwargs):
        """
        Always returns zero energy that does not contribute to gradients.

        Parameters
        ----------

        *args, **kwargs: Any
            Accepts any arguments (ignored).

        Returns
        -------

        E_zero: torch.Tensor
            Zero energy tensor (detached from computational graph).
            Infers batch size from first tensor argument.
        """
        # Infer batch size from first tensor argument
        for arg in args:
            if isinstance(arg, _torch.Tensor) and arg.ndim >= 1:
                batch_size = arg.shape[0]
                return _torch.zeros(batch_size, dtype=self._dtype, device=self._device, requires_grad=False)

        # Fallback: scalar zero
        return _torch.zeros(1, dtype=self._dtype, device=self._device, requires_grad=False)
