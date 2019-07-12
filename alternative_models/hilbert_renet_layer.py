#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import torch
import torch.nn as nn
import numpy as np
import math


class HilbertReNetLayer(nn.Module):
    """
    A single ReNet Layer, which gets an input image (with channels),
    produces patches and applies bidi-RNNs on the vertical and horizontal axis
    to get the context of the whole image.
    """

    def __init__(self, window_size, hidden_dim, rnn, channel_size=3, bias=True):
        """
        Initialization of the ReNet Layer.

        :param window_size: The window size, which divides the image into
            patches
        :param hidden_dim: Amount of hidden units.
        :param rnn: RNN-Type like GRU, LSTM or vanilla RNN
        :param channel_size: The amount of input channels
        :param bias: Boolean value if bias should be allowed or not
        """
        super(HilbertReNetLayer, self).__init__()

        # Needed for forward()-step
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.rnn_type = rnn
        self.hilbert_arr = None

        # First vertical sweep RNN
        self.firstVRNN = getattr(nn, self.rnn_type)(
            self.window_size * self.window_size * channel_size,
            self.hidden_dim,
            bidirectional=True,
            batch_first=True,
            bias=bias
        )

        # Second horizontal sweep RNN
        self.secondHRNN = getattr(nn, self.rnn_type)(
            2 * self.hidden_dim,
            self.hidden_dim,
            bidirectional=True,
            batch_first=True,
            bias=bias
        )

        # Weight initialization
        for layer_p in self.firstVRNN._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.xavier_uniform_(self.firstVRNN.__getattr__(p))

        for layer_p in self.secondHRNN._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.xavier_uniform_(self.secondHRNN.__getattr__(p))

    def forward(self, x):
        """
        Applies the whole ReNet Layer to x.

        Example: 4-D input tensor with shape (128, 3, 32, 32), Window Size 2
        and Hidden Dim of 50 results in an 4-D output tensor of shape
        (128, 100, 16, 16)

        :param x: 4-D input tensor with Shape (Batch, Channels, Height, Width)
        :return:4-D output tensor with Shape (Batch, 2*hidden_dim,
            Height / WS, Width / WS) with WS as Window Size
        """
        # Get patches
        patches = self.get_valid_patches(x)

        # Swap first two dimensions
        input_array = patches.permute(2, 0, 3, 1)

        # Apply first RNN
        vertical_results = self.apply_one_direction(
            input_array,
            self.firstVRNN
        )

        vertical_results = nn.functional.relu(vertical_results)

        # Swap vertical and horizontal dimensions
        vertical_results = vertical_results.permute(2, 1, 0, 3)

        # Apply second RNN
        feature_map = self.apply_one_direction(
            vertical_results,
            self.secondHRNN
        )

        # Get Initial Shape style
        feature_map = feature_map.permute(1, 3, 2, 0)
        return feature_map

    def apply_one_direction(self, x, rnn):
        """
        Applies bidi-RNN on the patches in one direction
        (noted below as the dimension of the patch_first_direction).

        :param x: 4-D input tensor of shape
            (patch_first_direction, batch, patch_second_direction, features)
        :param rnn: Bidi-RNN in the first direction
        :return: 4-D output tensor of shape
            (patch_first_direction, batch, patch_second_direction, 2*hidden_dim)
        """
        _in = torch.cat(torch.split(x, 1, dim=0), dim=1)
        hn = self.get_init_hidden(_in)
        out, _ = rnn(_in[0], hn)
        out = out.contiguous().view(1, out.shape[0], out.shape[1], out.shape[2])
        out = torch.cat(torch.split(out, x.shape[1], dim=1), dim=0)

        return out

    def get_init_hidden(self, x):
        """
        Initializes the internal states, depending on the RNN-Type.

        :param x: Example input image to get the dimensions
        :return: Initialized internal states
        """
        if self.rnn_type == "LSTM":
            h0 = torch.zeros(2, x.size(1), self.hidden_dim).to(x.device)
            c0 = torch.zeros(2, x.size(1), self.hidden_dim).to(x.device)
            return h0, c0
        else:
            return torch.zeros(2, x.size(1), self.hidden_dim).to(x.device)

    def get_valid_patches(self, x):
        """
        Get patches from input image x with shape
        (Batchsize, Channels, Height, Width). Example: 4-D Input tensor has
        shape [128, 3, 32, 32] and Window Size is 2, then the output tensor has
        shape [128, 12, 16, 16].

        :param x: Input-batches
        :return: A 4-D tensor of shape
            (Batchsize, WS * WS * Channels, Height / WS, Width / WS) with
            WS as Window Size
        """
        patches = x.unfold(2,
                           self.window_size,
                           self.window_size).unfold(3,
                                                    self.window_size,
                                                    self.window_size)
        patches = patches.contiguous().view(patches.shape[0],
                                            -1,
                                            patches.shape[2],
                                            patches.shape[3])
        if self.hilbert_arr is None:
            assert patches.shape[-1] == patches.shape[-2], \
                "input is not quadratic"
            log2 = math.log2(patches.shape[-1])
            assert log2 % 1 == 0., \
                "not of size 2^n"
            self.hilbert_arr = self.hilbert(int(log2))
        index = np.argsort([self.hilbert_arr], axis=None)
        patches = patches.permute(2, 3, 0, 1)
        result_patches = torch.empty_like(patches)
        for i in range(patches.shape[0]):
            for j in range(patches.shape[0]):
                result_patches[i][j] = patches[
                    (index[patches.shape[0] * i + j]) // patches.shape[0]][
                    (index[patches.shape[0] * i + j]) % patches.shape[0]]
        result_patches = result_patches.permute(2, 3, 0, 1)
        return result_patches

    def hilbert(self, n):
        """
        Generate a 2D hilbert embedding of a 1D sequence.
        :param n: int. Order of the hilbert embedding.
            Hilbert transformations are restricted to squares of side 2**n.
            Hence, hilbert(1).shape --> (2, 2); hilbert(2) --> (4, 4);
            hilbert(3).shape --> (8, 8)
        :return: 2d array of floats.
            Each element is the index of the timestep as
            embedded in a hilbert curve.
        """
        if n == 1:
            return np.array([0, 3, 1, 2]).reshape(2, 2)
        else:
            hilbert_pattern = self.hilbert(n - 1)
            tpn = 2 ** n
            tpn_h = int(tpn / 2)
            n_prev = 2 ** (2 * (n - 1))  # same as hilbert_pattern.size
            board = np.empty((tpn, tpn), dtype=np.int64)
            board[0:tpn_h, 0:tpn_h] = hilbert_pattern.T
            board[tpn_h:tpn, 0:tpn_h] = hilbert_pattern + n_prev
            board[tpn_h:tpn, tpn_h:tpn] = hilbert_pattern + 2 * n_prev
            board[0:tpn_h, tpn_h:tpn] = np.rot90(hilbert_pattern + 3 * n_prev,
                                                 2).T
            return board
