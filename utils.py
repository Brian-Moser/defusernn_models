#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import torch


def flip(x, dim):
    """
    Flips a dimension (reverse order). BidiLSTM for example uses this feature
    to apply the a LSTM with reversed time step (opposite direction).

    :param x: Tensor, which has a dimension to be flipped. The dimensions of x
        can be arbitrary.
    :param dim: The dimension/axis to be flipped.
    :return: New tensor with flipped dimension

    :example:
        >>> flip([[1,2,3], [4,5,6], [7,8,9]], 0)
        [[7,8,9], [4,5,6], [1,2,3]]
    """
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i) - 1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
    return x[inds]
