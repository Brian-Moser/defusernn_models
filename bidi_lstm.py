#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as func

from torch.autograd import Variable

sys.path.insert(0, os.path.abspath('../'))   # noqa

from defusernn_models.utils import flip


class BidiLSTM(nn.Module):
    r"""Bidirectional LSTM model, which combines the two directions with a
    <addition> operation. PyTorch conventionally uses the <concatenation>
    operation to combine the results of both directions.
    """

    def __init__(self, start_input_dim, params):
        r"""
        Initialization of the BidiLSTM model. It creates the different layers
        with dropout between them. If no dropout is wanted, then set the rates
        to zero.

        Args:
            start_input_dim: Feature-Dimension of the input.
            params: Dictionary of model parameters like amount of units.
                Needed: 'layer_dim' [int], 'dropout_rate'[list],
                'hidden_dim' [list] and 'output_dim' [int].
        """
        super(BidiLSTM, self).__init__()

        # Needed for forward()-step
        self.layer_dim = params['layer_dim']
        self.dropout_rate = params['dropout_rate']

        # Layer definitions
        self.bidi_LSTM_layers = []
        for i in range(self.layer_dim):
            input_dim = start_input_dim if i == 0 else params['hidden_dim'][i-1]
            layer = BidiLSTMLayerAdd(input_dim, params['hidden_dim'][i])
            self.bidi_LSTM_layers.append(layer)
        self.bidi_LSTM_layers = nn.ModuleList(self.bidi_LSTM_layers)
        self.fc = nn.Linear(params['hidden_dim'][-1], params['output_dim'])

    def forward(self, x):
        r"""Forward-Step of the BidiLSTM Model.

        Args:
            x: Input tensor (batchsize, timesteps, features)

        Returns:
            out: Softmax-Distribution of size 'output_dim'

        """
        out = x
        for i in range(self.layer_dim):
            out = self.bidi_LSTM_layers[i](out)
            if self.layer_dim > 1:
                out = func.dropout(out,
                                   p=self.dropout_rate[i],
                                   training=self.training)
        out = self.fc(out[:, -1, :])
        return out


class BidiLSTMLayerAdd(nn.Module):
    """
    A single bidirectional LSTM layer, which calculates both directions.
    It uses <addition> to combine the results.
    """

    def __init__(self, input_dim, hidden_dim, bias=True):
        """
        Initialization of a single layer of the BidiLSTM Model

        Args:
            input_dim: This defines the expected input feature size of the
                layer.
            hidden_dim: This defines the amount of hidden units and also the
                output feature size
            bias: Boolean value to include/exclude bias. Default: True
        """
        super(BidiLSTMLayerAdd, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm_forward = nn.LSTM(input_dim,
                                    self.hidden_dim,
                                    batch_first=True,
                                    bias=bias)
        self.lstm_backward = nn.LSTM(input_dim,
                                     self.hidden_dim,
                                     batch_first=True,
                                     bias=bias)

    def forward(self, x):
        """
        todo
        :param x:
        :return:
        """
        return self.forward_with_states(x)[0]

    def forward_with_states(self, x):
        """
        todo
        :param x:
        :return:
        """
        # Initialize internal states
        h0_forward = Variable(torch.zeros(
            1, x.size(0), self.hidden_dim).to(x.device))
        h0_backward = Variable(torch.zeros(
            1, x.size(0), self.hidden_dim).to(x.device))
        c0_forward = Variable(torch.zeros(
            1, x.size(0), self.hidden_dim).to(x.device))
        c0_backward = Variable(torch.zeros(
            1, x.size(0), self.hidden_dim).to(x.device))

        # One time step
        out_forward, (hn_forward, cn_forward) = self.lstm_forward(
            x, (h0_forward, c0_forward))
        out_backward, (hn_backward, cn_backward) = self.lstm_backward(
            flip(x, 1), (h0_backward, c0_backward))

        return (out_forward + out_backward), \
               (hn_forward, cn_forward), \
               (hn_backward, cn_backward)
