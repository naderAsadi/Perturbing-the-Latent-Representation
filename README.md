# Latent Representation Perturbation

Pytorch implementation of "A Novel Image Perturbation Approach: Perturbing
the Latent Representation".

## Prerequisites

- Python 3.3+
- [Pytorch](https://pytorch.org/)

## Usage

Pretrained models are availbale for LeNet, AlexNet ,and VGG16. To test with existing models you can specicy modelname and dataset.

For example:

    $ python latent_pert.py --model alexnet --dataset mnist
    $ python latent_pert.py --model vgg --dataset svhn
Pretrained models can be downloaded from [here](https://drive.google.com/open?id=1EpKWuXeMQWVqpPSpOd4VrNbA8l45QlOQ)
