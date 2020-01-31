# -*- coding: utf-8 -*-
""" Code for training and evaluating Self-Explaining Neural Networks.
Copyright (C) 2018 David Alvarez-Melis <dalvmel@mit.edu>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License,
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class additive_scalar_aggregator(nn.Module):
    """ Linear aggregator for interpretable classification.

        Aggregates a set of concept representations and their
        scores, generates a prediction probability output from them.

        Args:
            cdim (int):     input concept dimension
            nclasses (int): number of target classes

        Inputs:
            H:   H(x) vector of concepts (b x k x 1) [TODO: generalize to set maybe?]
            Th:  Theta(x) vector of concept scores (b x k x nclass)

        Output:
            - Vector of class probabilities (b x o_dim)

    """

    def __init__(self, cdim, nclasses):
        super(additive_scalar_aggregator, self).__init__()
        self.cdim      = cdim       # Dimension of each concept
        self.nclasses  = nclasses   # Numer of output classes
        self.binary = (nclasses == 1)

    def forward(self, H, Th):
        assert H.size(-2) == Th.size(-2), "Number of concepts in H and Th don't match"
        assert H.size(-1) == 1, "Concept h_i should be scalar, not vector sized"
        assert Th.size(-1) == self.nclasses, "Wrong Theta size"
        combined = torch.bmm(Th.transpose(1,2), H).squeeze(dim=-1)
        if self.binary:
            out = F.sigmoid(combined)
        else:
            out =  F.log_softmax(combined, dim = 1)
        return out

