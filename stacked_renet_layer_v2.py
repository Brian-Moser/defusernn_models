#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import torch
import torch.nn as nn


class ReNetLayer(nn.Module):
    def __init__(self, window_size, hidden_dim, rnn, direction="H", stack_size=1,
                 channel_size=3, bias=True, set_forget_gate_bias=False, dropout=0):
        super(ReNetLayer, self).__init__()

        if type(window_size) == int:
            self.window_size = (window_size, window_size)
        else:
            self.window_size = window_size
        self.rnn_type = rnn
        self.hidden_dim = hidden_dim
        self.stack_size=stack_size
        self.direction=direction

        # First vertical sweep RNN
        self.rnn = getattr(nn, self.rnn_type)(
            self.window_size[0] * self.window_size[1] * channel_size,
            self.hidden_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=stack_size,
            dropout=dropout,
            bias=bias
        )

        if bias and set_forget_gate_bias:
            self.forget_bias_init()

    def apply(self, fn):
        # Weight initialization
        for layer_p in self.rnn._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    fn(self.rnn.__getattr__(p))

    def forget_bias_init(self):
        for layer_p in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n, layer_p):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def forward(self, x):
        """
        Applies the whole ReNet layer to a given input x.

        Example: 4-D input tensor with shape (128, 3, 32, 32), a Window Size of
        2 and Hidden Dimensionality of 50 results in an 4-D output tensor
        in the shape of (128, 100, 16, 16).

        :param x: 4-D input tensor with shape (Batch, Channels, Height, Width)
        :return: 4-D output tensor with shape (Batch, 2*hidden_dim,
            Height / WS, Width / WS) with ws as window size
        """
        # Get patches
        patches = self.get_valid_patches(x)

        if self.direction == "V":
            input_array = patches.permute(2, 0, 3, 1)
        elif self.direction == "H":
            input_array = patches.permute(3, 0, 2, 1)
        else:
            raise NotImplementedError

        _in = torch.cat(torch.split(input_array, 1, dim=0), dim=1)
        hn = self.get_init_hidden(_in, self.stack_size)
        out, _ = self.rnn(_in[0], hn)
        out = out.contiguous().view(1, out.shape[0], out.shape[1], out.shape[2])
        out = torch.cat(torch.split(out, input_array.shape[1], dim=1), dim=0)

        # Get Initial Shape style
        if self.direction == "V":
            feature_map = out.permute(1, 3, 0, 2)
        else:
            feature_map = out.permute(1, 3, 2, 0)

        return feature_map

    def get_init_hidden(self, x, stack_size):
        """
        Initializes the internal states, depending on the RNN-type.

        :param x: Example input image to get the dimensions
        :return: Initialized internal states
        """
        if self.rnn_type == "LSTM":
            h0 = torch.zeros(2*stack_size, x.size(1), self.hidden_dim).to(x.device)
            c0 = torch.zeros(2*stack_size, x.size(1), self.hidden_dim).to(x.device)
            return h0, c0
        else:
            return torch.zeros(2*stack_size, x.size(1), self.hidden_dim).to(x.device)

    def get_valid_patches(self, x):
        """
        Get patches from input image x with shape
        (Batch-Size, Channels, Height, Width). Example: 4-D input tensor has
        shape [128, 3, 32, 32] and Window Size is 2, then the output tensor has
        shape [128, 12, 16, 16].
        :param x: Input-batches
        :return: A 4-D tensor of shape
            (Batch-Size, WS * WS * Channels, Height / WS, Width / WS) with
            WS as Window Size
        """
        patches = x.unfold(2,
                           self.window_size[0],
                           self.window_size[0]
                           ).unfold(3,
                                    self.window_size[1],
                                    self.window_size[1]
                                    ).permute(0, 2, 3, 1, 4, 5, )
        patches = patches.contiguous().view(
            patches.size(0), patches.size(1), patches.size(2), -1
        ).permute(0, 3, 1, 2)
        return patches
