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

import torch as _torch


RCUBED_FREE = {
    0: 0.0,
    1: 7.912562569339352,
    6: 34.03997517722276,
    7: 25.544561093818928,
    8: 21.16239486693852,
    16: 74.3382807702708,
}


ALPHA_FREE = {
    0: 0.0,
    1: 4.5,
    2: 1.38,
    3: 164.0,
    4: 37.70,
    5: 20.50,
    6: 11.30,
    7: 7.40,
    8: 5.30,
    9: 3.740,
    10: 2.66,
    11: 163.0,
    12: 71.20,
    13: 57.80,
    14: 37.30,
    15: 25.0,
    16: 19.40,
    17: 14.60,
    18: 11.1,
    19: 290.0,
    20: 161.0,
    21: 97.0,
    22: 100.0,
    23: 87.0,
    24: 83.0,
    25: 68.0,
    26: 62.0,
    27: 55.0,
    28: 49.0,
    29: 47.0,
    30: 38.70,
    31: 50.0,
    32: 40.0,
    33: 30.0,
    34: 29.0,
    35: 21.0,
    36: 16.8,
    37: 320.0,
    38: 197.0,
    39: 162.0,
    40: 112.0,
    41: 98.0,
    42: 87.0,
    43: 79.0,
    44: 72.0,
    45: 66.0,
    46: 26.10,
    47: 55.0,
    48: 46.0,
    49: 65.0,
    50: 53.0,
    51: 43.0,
    52: 38.0,
    53: 32.90,
    54: 27.30,
    55: 401.0,
    56: 272.0,
    57: 215.0,
    58: 205.0,
    59: 216.0,
    60: 208.0,
    61: 200.0,
    62: 192.0,
    63: 184.0,
    64: 158.0,
    65: 170.0,
    66: 165.0,
    67: 156.0,
    68: 150.0,
    69: 144.0,
    70: 139.0,
    71: 137.0,
    72: 103.0,
    73: 74.0,
    74: 68.0,
    75: 62.0,
    76: 57.0,
    77: 54.0,
    78: 48.0,
    79: 36.0,
    80: 33.90,
    81: 50.0,
    82: 47.0,
    83: 48.0,
    84: 44.0,
    85: 42.0,
    86: 35.0,
    87: 318.0,
    88: 246.0,
    89: 203.0,
    90: 217.0,
    91: 154.0,
    92: 129.0,
    93: 151.0,
    94: 132.0,
    95: 131.0,
    96: 144.0,
    97: 125.0,
    98: 122.0,
    99: 118.0,
    100: 113.0,
    101: 109.0,
    102: 110.0,
    103: 320.0,
    104: 112.0,
    105: 42.0,
    106: 40.0,
    107: 38.0,
    108: 36.0,
    109: 34.0,
    110: 32.0,
    111: 32.0,
    112: 28.0,
    113: 29.0,
    114: 31.0,
    115: 71.0,
    117: 76.0,
    118: 58.0,
}

ALPHA_FREE_TENSOR = _torch.tensor([ALPHA_FREE.get(i, 0.0) for i in range(119)])
RCUBED_FREE_TENSOR = _torch.tensor([RCUBED_FREE.get(i, 0.0) for i in range(119)])
