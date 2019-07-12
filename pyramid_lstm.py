#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('../../'))  # noqa

from src.utils import flip
from defusernn_models.md_lstm import MDLSTMCell


class PyraMiDLSTMModel(nn.Module):
    def __init__(self, start_channel_size, params):
        super(PyraMiDLSTMModel, self).__init__()

        # Layer definitions
        self.pyramid_LSTM_layers = []
        for i in range(len(params['hidden_dim'])):
            channel_size = start_channel_size if i == 0 \
                      else params['hidden_dim'][i - 1]
            self.pyramid_LSTM_layers.append(
                PyraMiDLSTMLayer(channel_size,
                              params['hidden_dim'][i],
                              params['hidden_dim'][i + 1]
                              if i + 1 != len(params['hidden_dim'])
                              else params['output_dim']))
        self.pyramid_LSTM_layers = nn.Sequential(*self.pyramid_LSTM_layers)

    def forward(self, x):
        return self.pyramid_LSTM_layers(x)


class PyraMiDLSTMLayer(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_channels):
        super(PyraMiDLSTMLayer, self).__init__()

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

        return out

    @staticmethod
    def calculate_one_layer(_input, layer):
        h, c = layer.get_init_states(_input)
        for i in range(_input.shape[0]):
            for j in range(_input.shape[1]):
                x = i - 1 if i - 1 > 0 else 0
                y_neg = j - 1 if j - 1 > 0 else 0
                y_pos = j + 1 if j + 1 < _input.shape[1] else j

                h[i][j], c[i][j] = layer.forward(_input[i, j], (
                        torch.stack([h[x][y_neg], h[x][y_pos]], dim=0),
                        torch.stack([c[x][y_neg], c[x][y_pos]], dim=0)
                    ))

        return h