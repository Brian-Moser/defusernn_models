# DefuseRNN - Recurrent CV models

Author: Brian B. Moser

<p>
The project provides an implementation of recurrent models (listed below) applied to computer vision problems like 
image classification or object recognition. The implementation is in PyTorch.
<p/>

## Supported Models (Paper links)

* [Bidirectional LSTM](https://pdfs.semanticscholar.org/4b80/89bc9b49f84de43acc2eb8900035f7d492b2.pdf)
* [ReNet](https://arxiv.org/abs/1505.00393)
* [ConvLSTM](https://arxiv.org/abs/1506.04214)
* [MD-LSTM](https://arxiv.org/abs/0705.2011)
* [PyraMiD-LSTM](https://arxiv.org/abs/1506.07452)

## Current Status

Complete:
- ReNet (optimized)

Paper results for
- ReNet (CIFAR10, SVHN)
- MD-LSTM (Clean MNIST Seg)
- ConvLSTM (Moving MNIST)

Currently:
All other models (not ReNet) need to be optimized.

