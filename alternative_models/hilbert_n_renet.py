#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import os
import sys
import torch.nn as nn

sys.path.insert(0, os.path.abspath('../..'))   # noqa

from defusernn_models.alternative_models.hilbert_renet_layer import HilbertReNetLayer


class Hilbert_N_ReNet(nn.Module):
    """
    Implementation of the ReNet Model.
    """

    def __init__(self, input_image_shape, params):
        """Initialization of the ReNet Model.

        :param input_image_shape: Expected height and width of the input images
            (mainly to check if the partitions caused by the window sizes are
            possible or not).
        :param params: Dictionary with model parameters like window size
            or layer dimensions
        """
        super(Hilbert_N_ReNet, self).__init__()

        # Temporal values for initialization
        width, height = input_image_shape[1], input_image_shape[2]

        if 'batch_norm' in params.keys():
            self.batch_norm = params['batch_norm']
        else:
            self.batch_norm = False

        # Definition of the ReNet Layers
        self.reNet_layers = []
        if 'dropout_rate' in params.keys():
            self.reNet_layers.append(nn.Dropout2d(params['dropout_rate'][0]))
        for i in range(len(params['reNet_hidden_dim'])):
            # Check, if possible and save final output dim
            # of all ReNet Layers together
            assert width % params['window_size'][i] == 0, \
                "Width is not dividable by size"
            assert height % params['window_size'][i] == 0, \
                "Width is not dividable by size"
            width = int(width / params['window_size'][i])
            height = int(height / params['window_size'][i])

            # Get the channel size of the previous layer
            if i == 0:
                channel_size = input_image_shape[0]
            else:
                channel_size = 2 * params['reNet_hidden_dim'][i - 1]
            self.reNet_layers.append(HilbertReNetLayer(
                params['window_size'][i],
                params['reNet_hidden_dim'][i],
                params['rnn_types'][i],
                channel_size
            ))

            # BN -> ReLu -> Dropout
            if self.batch_norm:
                self.reNet_layers.append(
                    nn.BatchNorm2d(2 * params['reNet_hidden_dim'][i])
                )

            self.reNet_layers.append(nn.ReLU())

            if 'dropout_rate' in params.keys():
                self.reNet_layers.append(
                    nn.Dropout2d(params['dropout_rate'][i + 1])
                )

        self.reNet_layers = nn.Sequential(*self.reNet_layers)

        # Definition of the Fully Connected Layers
        self.linear_layers = []
        for i in range(len(params['linear_hidden_dim'])):
            if i == 0:
                input_size = width * height * 2 * params['reNet_hidden_dim'][-1]
            else:
                input_size = params['linear_hidden_dim'][i - 1]
            fc = nn.Linear(input_size, params['linear_hidden_dim'][i])
            nn.init.xavier_uniform_(fc.weight)
            self.linear_layers.append(fc)
            self.linear_layers.append(nn.ReLU())
            if not self.batch_norm:
                self.linear_layers.append(
                    nn.Dropout(
                        params['dropout_rate'][
                            len(params['reNet_hidden_dim']) + 1 + i]
                    )
                )
        fc = nn.Linear(params['linear_hidden_dim'][-1], params['output_dim'])
        nn.init.xavier_uniform_(fc.weight)
        self.linear_layers.append(fc)
        self.linear_layers = nn.Sequential(*self.linear_layers)

    def forward(self, x):
        """Performs the forward-step of the ReNet.

        :param x: Input Image
        :return: Output of the network
        """
        out = self.reNet_layers(x)
        out = out.contiguous().view(out.shape[0], -1)
        out = self.linear_layers(out)
        return out
