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

"""Base class for energy interactions."""

__author__ = "Joao Morado"
__email__ = "joaomorado@gmail.com"

__all__ = ["BaseInteraction"]

import torch as _torch


class BaseInteraction(_torch.nn.Module):
    """
    Abstract base class for all energy interaction modules.

    All interaction modules should inherit from this class and implement
    the forward method with their specific energy calculation.
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
        super().__init__()

        if device is None:
            device = _torch.get_default_device()

        if dtype is None:
            dtype = _torch.get_default_dtype()

        self._device = device
        self._dtype = dtype

    def forward(self, *args, **kwargs):
        """
        Dispatch to the appropriate implementation.
        """
        return self._forward_impl(*args, **kwargs)
