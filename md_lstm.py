#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('../'))  # noqa

from defusernn_models.utils import flip


class MDMDLSTMModel(nn.Module):
    def __init__(self, start_channel_size, params):
        super(MDMDLSTMModel, self).__init__()

        # Layer definitions
        self.md_LSTM_layers = []
        for i in range(len(params['hidden_dim'])):
            channel_size = start_channel_size if i == 0 \
                      else params['hidden_dim'][i - 1]
            self.md_LSTM_layers.append(
                MDMDLSTMLayer(channel_size,
                              params['hidden_dim'][i],
                              params['hidden_dim'][i + 1]
                              if i + 1 != len(params['hidden_dim'])
                              else params['output_dim']))
        self.md_LSTM_layers = nn.Sequential(*self.md_LSTM_layers)

    def forward(self, x):
        return self.md_LSTM_layers(x)


class MDMDLSTMLayer(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_channels):
        super(MDMDLSTMLayer, self).__init__()

        self.md_LSTM_cells = []
        for j in range(4):
            self.md_LSTM_cells.append(MDLSTMCell(input_channels, hidden_dim))
        self.md_LSTM_cells.append(
            nn.Conv2d(4*hidden_dim, output_channels, kernel_size=1)
        )
        self.md_LSTM_cells = nn.ModuleList(self.md_LSTM_cells)

    def forward(self, x):
        out = x.permute(2, 3, 0, 1)
        out_hf_vf = self.calculate_one_layer(out, self.md_LSTM_cells[0])
        out_hr_vf = self.calculate_one_layer(flip(out, 0),
                                             self.md_LSTM_cells[1])
        out_hf_vr = self.calculate_one_layer(flip(out, 1),
                                             self.md_LSTM_cells[2])
        out_hr_vr = self.calculate_one_layer(flip(flip(out, 0), 1),
                                             self.md_LSTM_cells[3])
        out_combined = torch.cat([out_hf_vf, out_hr_vf, out_hf_vr, out_hr_vr],
                                 dim=3).permute(2, 3, 0, 1)
        out = self.md_LSTM_cells[4](out_combined)

        # out = x.permute(2, 3, 0, 1)
        # out = torch.cat(
        #     [out, flip(out, 0), flip(out, 1), flip(flip(out, 0), 1)], dim=3)
        # out = self.calculate_one_layer(out, self.md_cell)
        # out = self.pred_layer(out.permute(2, 3, 0, 1))

        return out

    @staticmethod
    def calculate_one_layer(_input, layer):
        h, c = layer.get_init_states(_input)
        for i in range(_input.shape[0]):
            for j in range(_input.shape[1]):
                x = i - 1 if i - 1 >= 0 else 0
                y = j - 1 if j - 1 >= 0 else 0

                h[i][j], c[i][j] = layer.forward(_input[i, j], (
                        torch.stack([h[x][j], h[i][y]], dim=0),
                        torch.stack([c[x][j], c[i][y]], dim=0)
                    ))

        return h


class MDLSTMCell(nn.Module):
    """
    Implementation of the MD-LSTM Cell
    """

    def __init__(self, channel_size, hidden_dim):
        """
        Initialization of the MD-LSTM Cell

        :param channel_size: Amount of expected channels
        :param hidden_dim: Amount of units
        """
        super(MDLSTMCell, self).__init__()

        self.hidden_dim = hidden_dim

        self.linear_layer = nn.Linear(2 * hidden_dim + channel_size,
                                      5 * hidden_dim,
                                      bias=True)

        self.Wci = nn.Parameter(torch.empty(1,
                                            self.hidden_dim,
                                            requires_grad=True))
        self.Wf1 = nn.Parameter(torch.empty(1,
                                            self.hidden_dim,
                                            requires_grad=True))
        self.Wf2 = nn.Parameter(torch.empty(1,
                                            self.hidden_dim,
                                            requires_grad=True))
        self.Wco = nn.Parameter(torch.empty(1,
                                            self.hidden_dim,
                                            requires_grad=True))
        nn.init.uniform_(self.Wci)
        nn.init.uniform_(self.Wf1)
        nn.init.uniform_(self.Wf2)
        nn.init.uniform_(self.Wco)

    def forward(self, x, states):
        # vanilla LSTM implementation
        h, c = states

        combined = torch.cat([x, h[0], h[1]], dim=1)
        i, f1, f2, g_term, o_term = torch.split(
            self.linear_layer(combined),
            self.hidden_dim,
            dim=1
        )

        return self.calc_states(c, i, f1, f2, g_term, o_term,
                                self.Wci, self.Wf1, self.Wf2, self.Wco)

    def get_init_states(self, x):
        """
        Returns initial zero-matrix states h and c with the right dimensions.

        :param x: Example Input, important to match the right weight matrix
            dimensions
        :return: The states h and c
        """

        return (torch.zeros(
            x.size(0), x.size(1), x.size(2), self.hidden_dim
                ).to(x.device),
                torch.zeros(
                    x.size(0), x.size(1), x.size(2), self.hidden_dim
                ).to(x.device))

    @staticmethod
    @torch.jit.script
    def calc_states(c, i, f1, f2, g_term, o_term, Wci, Wf1, Wf2, Wco):
        i = (i + Wci * c[0] + Wci * c[1]).sigmoid()
        f1 = (f1 + Wf1 * c[0]).sigmoid()
        f2 = (f2 + Wf2 * c[1]).sigmoid()
        c_next = f1 * c[0] + f2 * c[1] + i * g_term.tanh()
        o = (o_term + Wco * c_next).sigmoid()
        h_next = o * c_next.tanh()

        return h_next, c_next
