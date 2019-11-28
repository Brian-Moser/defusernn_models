#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import torch
import torch.nn as nn


class ReNetLayer(nn.Module):
    """
    A single ReNet layer: It produces patches and applies two bidi-RNNs
    on the vertical and horizontal axis to get the context of the whole image.
    For further explanation take a look into the paper.
    """

    def __init__(self, window_size, hidden_dim, rnn, stack_size=(1, 1),
                 channel_size=3, bias=True, set_forget_gate_bias=False,
                 custom_activation=False, batchnorm_between=False):
        """
        Initialization of the ReNet layer.

        :param window_size: The Window Size for dividing the image into
            patches. Can be either a single int digit or a tuple for height and
            width.
        :param hidden_dim: Amount of hidden units. This will determine the depth
            (Channel-size) of the resulting activation/feature map.
        :param rnn: RNN-Type like "GRU", "LSTM" or vanilla "RNN". It must be
            existing in the PyTorch library.
        :param channel_size: The amount of expected input channels.
        :param bias: Boolean value if bias should be allowed or not.
        :param set_forget_gate_bias: It can take a while for a recurrent network
         to learn to remember information
        form the last time step. Initialize biases for LSTM’s forget gate to 1
        to remember more by default.
        Similarly, initialize biases for GRU’s reset gate to -1.
        """
        super(ReNetLayer, self).__init__()

        # Needed for forward()-step
        if type(window_size) == int:
            self.window_size = (window_size, window_size)
        else:
            self.window_size = window_size
        self.rnn_type = rnn
        self.hidden_dim = hidden_dim

        # First vertical sweep RNN
        self.firstVRNN = getattr(nn, self.rnn_type)(
            self.window_size[0] * self.window_size[1] * channel_size,
            self.hidden_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=stack_size[0],
            bias=bias
        )

        self.batchNorm = None
        if batchnorm_between:
            self.batchNorm = torch.nn.BatchNorm2d(2 * self.hidden_dim)

        # Second horizontal sweep RNN
        self.secondHRNN = getattr(nn, self.rnn_type)(
            2 * self.hidden_dim,
            self.hidden_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=stack_size[1],
            bias=bias
        )

        self.custom_activation = custom_activation

        self.apply(nn.init.xavier_uniform_)

        if bias and set_forget_gate_bias:
            self.forget_bias_init()

        if self.rnn_type == "LSTM":
            h0 = torch.zeros(2*stack_size[0], 1, self.hidden_dim).cuda()
            c0 = torch.zeros(2*stack_size[0], 1, self.hidden_dim).cuda()
            self.init_state_v = (nn.Parameter(h0), nn.Parameter(c0))
        else:
            self.init_state_v = nn.Parameter(torch.zeros(2*stack_size[0], 1, self.hidden_dim).cuda())

        if self.rnn_type == "LSTM":
            h0 = torch.zeros(2*stack_size[1], 1, self.hidden_dim).cuda()
            c0 = torch.zeros(2*stack_size[1], 1, self.hidden_dim).cuda()
            self.init_state_h = (nn.Parameter(h0), nn.Parameter(c0))
        else:
            self.init_state_h = nn.Parameter(torch.zeros(2*stack_size[1], 1, self.hidden_dim).cuda())

    def apply(self, fn):
        # Weight initialization
        for layer_p in self.firstVRNN._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    fn(self.firstVRNN.__getattr__(p))

        for layer_p in self.secondHRNN._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    fn(self.secondHRNN.__getattr__(p))

    def forget_bias_init(self):
        for layer_p in self.firstVRNN._all_weights:
            for name in filter(lambda n: "bias" in n, layer_p):
                bias = getattr(self.firstVRNN, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)
        for layer_p in self.secondHRNN._all_weights:
            for name in filter(lambda n: "bias" in n, layer_p):
                bias = getattr(self.secondHRNN, name)
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

        # Swap first two dimensions
        input_array = patches.permute(2, 0, 3, 1)

        # Apply first RNN
        vertical_results = self.apply_one_direction(
            input_array,
            self.firstVRNN,
            self.init_state_v
        )

        # Swap vertical and horizontal dimensions
        if self.batchNorm is not None:
            vertical_results = self.batchNorm(vertical_results.permute(1, 3, 0, 2)).permute(3, 0, 2, 1)
        else:
            vertical_results = vertical_results.permute(2, 1, 0, 3)

        # Apply second RNN
        feature_map = self.apply_one_direction(
            vertical_results,
            self.secondHRNN,
            self.init_state_h
        )

        # Get Initial Shape style
        feature_map = feature_map.permute(1, 3, 2, 0)

        return feature_map

    @staticmethod
    def atanh(_input):
        return 0.5 * torch.log((1 + _input) / (1 - _input))

    def apply_one_direction(self, x, rnn, hn):
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
        hn = self.get_init_hidden(hn, _in)
        out, _ = rnn(_in[0], hn)
        out = out.contiguous().view(1, out.shape[0], out.shape[1], out.shape[2])
        out = torch.cat(torch.split(out, x.shape[1], dim=1), dim=0)

        if self.custom_activation is not False:
            out = self.atanh(out)
            out = self.custom_activation(out)

        return out

    def get_init_hidden(self, hn, x):
        """
        Initializes the internal states, depending on the RNN-type.

        :param x: Example input image to get the dimensions
        :return: Initialized internal states
        """
        if self.rnn_type == "LSTM":
            hn.expand(-1, x.shape[1], -1).contiguous(), hn.expand(-1, x.shape[1], -1).contiguous()
        else:
            return hn.expand(-1, x.shape[1], -1).contiguous()

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
