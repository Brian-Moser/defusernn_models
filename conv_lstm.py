#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import torch
import torch.nn as nn
import numpy as np


class ConvLSTM(nn.Module):
    """
    Implementation of the Encoding-Forecasting ConvLSTM Model.
    """

    def __init__(self, input_shape, params):
        """
        Initialization of the Encoding-Forecasting ConvLSTM Model.

        :param input_shape: Expected shape of the input
        :param params: Model Parameters Dictionary
        """
        super(ConvLSTM, self).__init__()

        # Ensure that the patch size is possible
        assert input_shape[2] % params['initial_patch_size'][0] == 0, \
            "Width is not dividable by patch size"
        assert input_shape[3] % params['initial_patch_size'][1] == 0, \
            "Height is not dividable by patch size"

        self.hidden_dim = params['conv_hidden_dim']
        self.conv_layer_dim = len(self.hidden_dim)
        self.input_kernel_size = params['input_kernel_size']
        self.kernel_size = params['kernel_size']
        self.patch_size = params['initial_patch_size']
        image_size = (input_shape[-2]//self.patch_size[0],
                      input_shape[-1]//self.patch_size[1])

        # Encoding Layers
        start_channels = input_shape[1]*self.patch_size[0]*self.patch_size[1]
        self.enc_layers = []
        for layer_index in range(self.conv_layer_dim):
            if layer_index == 0:
                input_channels = start_channels
            else:
                input_channels = self.hidden_dim[layer_index - 1]
            self.enc_layers.append(ConvLSTMCell(
                image_size=image_size,
                input_channels=input_channels,
                hidden_dim=self.hidden_dim[layer_index],
                input_kernel_size=self.input_kernel_size,
                hidden_kernel_size=self.kernel_size[layer_index]
            ))
        self.enc_layers = nn.ModuleList(self.enc_layers)

        # Forecasting/Decoder Layers
        self.dec_layers = []

        for layer_index in range(self.conv_layer_dim):
            input_channels = self.hidden_dim[layer_index - 1]
            self.dec_layers.append(ConvLSTMCell(
                image_size=image_size,
                input_channels=input_channels,
                hidden_dim=self.hidden_dim[layer_index],
                input_kernel_size=self.input_kernel_size,
                hidden_kernel_size=self.kernel_size[layer_index],
                with_input=(layer_index != 0)
            ))
        self.dec_layers = nn.ModuleList(self.dec_layers)

        # Final Prediction Layer
        self.pred_layer = nn.Conv2d(
            in_channels=np.sum(self.hidden_dim),
            out_channels=start_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        """
        Forward Step of Encoding-Forecasting ConvLSTM Model

        :param x: Input Image (B, T, C, H, W)
        :return: Output Image (B, T, C, H, W)
        """
        x_reshaped = x.contiguous().view(
            x.shape[0],
            x.shape[1],
            x.shape[2] * self.patch_size[0] * self.patch_size[1],
            x.shape[3] // self.patch_size[0],
            x.shape[3] // self.patch_size[1]
        )

        # Encoder steps
        last_states = []
        for layer_index in range(len(self.enc_layers)):
            last_states.append(
                self.enc_layers[layer_index].get_init_states(x_reshaped)
            )
        for timestep in range(x.shape[1]):
            current_input = x_reshaped[:, timestep, :, :, :]
            for layer_index in range(len(self.enc_layers)):
                last_state = (current_input, c) = self.enc_layers[layer_index](
                    current_input,
                    last_states[layer_index]
                )
                last_states[layer_index] = last_state

        # Decoder steps
        layer_outputs = None
        result = None
        current_input = torch.zeros_like(current_input).to(x.device)
        for timestep in range(x.shape[1]):
            for layer_index in range(len(self.dec_layers)):
                last_state = (current_input, c) = self.dec_layers[layer_index](
                    current_input,
                    last_states[layer_index]
                )
                last_states[layer_index] = last_state
                if layer_index == 0:
                    layer_outputs = current_input
                else:
                    layer_outputs = torch.cat((current_input, layer_outputs), 1)
                current_input = nn.functional.dropout2d(current_input, p=0.5)

            # prediction
            prediction = self.pred_layer(layer_outputs)
            prediction = torch.sigmoid(prediction.contiguous().view(
                prediction.shape[0],
                1,
                prediction.shape[1] // self.patch_size[0] // self.patch_size[1],
                prediction.shape[2] * self.patch_size[0],
                prediction.shape[3] * self.patch_size[1]
            ))
            if result is None:
                result = prediction
            else:
                result = torch.cat((result, prediction), 1)
        return result


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM Cell/Layer.
    """

    def __init__(self, image_size, input_channels, hidden_dim,
                 input_kernel_size, hidden_kernel_size, with_input=True):
        """
        Initialization of single ConvLSTM Cell/Layer.

        :param image_size: Height and width of the input
        :param input_channels: Channel size of the input
        :param hidden_dim: Amount of units used in this layer. The output
            channel size is equal to this amount
        :param input_kernel_size: Kernel size for convolving the input
        :param hidden_kernel_size: Kernel size for convolving the last output h
        """
        super(ConvLSTMCell, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_shape = image_size
        self.input_padding = (input_kernel_size - 1) // 2, \
                             (input_kernel_size - 1) // 2
        self.hidden_padding = (hidden_kernel_size - 1) // 2, \
                              (hidden_kernel_size - 1) // 2

        # Input-to-State Conv2D
        if with_input:
            self.conv_input = nn.Conv2d(in_channels=input_channels,
                                        out_channels=4 * self.hidden_dim,
                                        kernel_size=input_kernel_size,
                                        padding=self.input_padding,
                                        bias=False)
        else:
            self.conv_input = None

        # State-to-State Conv2D
        self.conv_hidden = nn.Conv2d(in_channels=self.hidden_dim,
                                     out_channels=4 * self.hidden_dim,
                                     kernel_size=hidden_kernel_size,
                                     padding=self.hidden_padding,
                                     bias=True)

        # Initialize Weights for Cell State
        self.Wci = nn.Parameter(torch.empty(1,
                                            self.hidden_dim,
                                            1,
                                            1,
                                            requires_grad=True))
        self.Wcf = nn.Parameter(torch.empty(1,
                                            self.hidden_dim,
                                            1,
                                            1,
                                            requires_grad=True))
        self.Wco = nn.Parameter(torch.empty(1,
                                            self.hidden_dim,
                                            1,
                                            1,
                                            requires_grad=True))
        nn.init.uniform_(self.Wci)
        nn.init.uniform_(self.Wcf)
        nn.init.uniform_(self.Wco)

    def forward(self, x, states):
        """
        A single forward for a given timestep (which means the input and the
        states belong to the same timestep)

        :param x: Input Image
        :param states: Current state
        :return: The next output h and the next cell state c
        """
        h, c = states

        # Convolution over input
        # Wxi, Wxf, Wxo, Wxg convolved with x
        # Wxg for candidate values
        if self.conv_input is not None:
            out_input = self.conv_input(x)
            in_conv_i, in_conv_f, in_conv_o, in_conv_g = torch.split(
                out_input,
                self.hidden_dim,
                dim=1
            )
        else:
            in_conv_i, in_conv_f, in_conv_o, in_conv_g = 0, 0, 0, 0

        # Convolution over h
        # Whi, Whf, Who, Whg convolved with h
        # Whg for candidate values
        out_hidden = self.conv_hidden(h)
        h_conv_i, h_conv_f, h_conv_o, h_conv_g = torch.split(
            out_hidden,
            self.hidden_dim,
            dim=1
        )

        # ConvLSTM equations
        i = (in_conv_i + h_conv_i + self.Wci * c).sigmoid()
        f = (in_conv_f + h_conv_f + self.Wcf * c).sigmoid()
        c_next = f * c + i * (in_conv_g + h_conv_g).tanh()
        o = (in_conv_o + h_conv_o + self.Wco * c_next).sigmoid()
        h_next = o * c_next.tanh()

        return h_next, c_next

    def get_init_states(self, x):
        """
        Returns initial zero-matrix states h and c with the right dimensions.

        :param x: Example Input, important to match the right weight matrix
            dimensions
        :return: The states h and c
        """
        return (torch.zeros(x.shape[0],
                            self.hidden_dim,
                            x.shape[-2],
                            x.shape[-1],
                            requires_grad=False).to(x.device),
                torch.zeros(x.shape[0],
                            self.hidden_dim,
                            x.shape[-2],
                            x.shape[-1],
                            requires_grad=False).to(x.device))
