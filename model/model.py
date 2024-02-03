"""Network modules for pytorch models.

Functions
---------
conv_block(in_channels, out_channels, *args, **kwargs)
dense_block(out_features, *args, **kwargs)
dense_lazy_block(out_features, *args, **kwargs)


Classes
---------
RescaleLayer()
TorchModel(base.base_model.BaseModel)

"""

import torch
from base.base_model import BaseModel
import matplotlib.pyplot as plt


# https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict


def conv_block(in_channels, out_channels, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
    )


def dense_block(out_features, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.LazyLinear(out_features=out_features, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=out_features, out_features=out_features, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=out_features, out_features=out_features, bias=True),
        torch.nn.ReLU(),
    )


def dense_lazy_block(out_features, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.LazyLinear(out_features=out_features, bias=True),
        torch.nn.ReLU(),
    )


class RescaleLayer:
    def __init__(self, scale, offset):
        self.offset = offset
        self.scale = scale

    def __call__(self, x):
        x = torch.multiply(x, self.scale)
        x = torch.add(x, self.offset)
        return x


class TorchModel(BaseModel):
    def __init__(self, target=None):
        super().__init__()

        if target is None:
            self.target_mean = torch.tensor(0.0)
            self.target_std = torch.tensor(1.0)
        else:
            self.target_mean = torch.tensor(target.mean(axis=0))
            self.target_std = torch.tensor(target.std(axis=0))

        self.pad_lons = torch.nn.CircularPad2d((5, 5, 0, 0))

        self.cnn_block = torch.nn.Sequential(
            conv_block(in_channels=1, out_channels=32, kernel_size=5, padding="same"),
            conv_block(in_channels=32, out_channels=32, kernel_size=3, padding="same"),
            conv_block(in_channels=32, out_channels=32, kernel_size=3, padding="same"),
            torch.nn.Flatten(start_dim=1),
        )

        self.finaldense_mu = dense_lazy_block(out_features=10)
        self.finaldense_sigma = dense_lazy_block(out_features=10)
        self.finaldense_gamma = dense_lazy_block(out_features=10)
        self.finaldense_tau = dense_lazy_block(out_features=10)

        self.denseblock_mu = dense_block(
            out_features=10,
        )
        self.denseblock_sigma = dense_block(
            out_features=10,
        )
        self.denseblock_gamma = dense_block(
            out_features=10,
        )
        self.denseblock_tau = dense_block(
            out_features=10,
        )

        self.rescale_mu = RescaleLayer(self.target_std, self.target_mean)
        self.rescale_sigma = RescaleLayer(torch.tensor(1.0), torch.log(self.target_std))
        self.rescale_tau = RescaleLayer(torch.tensor(0.0), torch.tensor(1.0))

        self.output_mu = torch.nn.Linear(in_features=10, out_features=1, bias=True)
        self.output_sigma = torch.nn.Linear(in_features=10, out_features=1, bias=True)
        self.output_gamma = torch.nn.Linear(in_features=10, out_features=1, bias=True)
        self.output_tau = torch.nn.Linear(in_features=10, out_features=1, bias=True)

    def forward(self, x, x_unit):

        x = self.pad_lons(x)
        x = self.cnn_block(x)

        # build mu_layers
        x_mu = torch.cat((x, x_unit[:, None]), dim=-1)
        x_mu = self.denseblock_mu(x_mu)
        x_mu = torch.cat((x_mu, x_unit[:, None]), dim=-1)
        x_mu = self.finaldense_mu(x_mu)
        mu_out = self.output_mu(x_mu)

        # build sigma_layers
        x_sigma = torch.cat((x, x_unit[:, None]), dim=-1)
        x_sigma = self.denseblock_sigma(x_sigma)
        x_sigma = torch.cat((x_sigma, x_unit[:, None]), dim=-1)
        x_sigma = self.finaldense_sigma(x_sigma)
        sigma_out = self.output_sigma(x_sigma)

        # build gamma_layers
        x_gamma = torch.cat((x, x_unit[:, None]), dim=-1)
        x_gamma = self.denseblock_gamma(x_gamma)
        x_gamma = torch.cat((x_gamma, x_unit[:, None]), dim=-1)
        x_gamma = self.finaldense_gamma(x_gamma)
        gamma_out = self.output_gamma(x_gamma)

        # build tau_layers
        x_tau = torch.cat((x, x_unit[:, None]), dim=-1)
        x_tau = self.denseblock_tau(x_tau)
        x_tau = torch.cat((x_tau, x_unit[:, None]), dim=-1)
        x_tau = self.finaldense_tau(x_tau)
        tau_out = self.output_tau(x_tau)

        # rescaling layers
        mu_out = self.rescale_mu(mu_out)

        sigma_out = self.rescale_sigma(sigma_out)
        sigma_out = torch.exp(sigma_out)

        tau_out = self.rescale_tau(tau_out)

        # final output, concatenate parameters together
        x = torch.cat((mu_out, sigma_out, gamma_out, tau_out), dim=-1)

        return x
