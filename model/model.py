"""Network modules for pytorch models.

Functions
---------
conv_couplet(in_channels, out_channels, act_fun, *args, **kwargs)
dense_lazy_couplet(out_features, act_fun, *args, **kwargs)
conv_block(in_channels, out_channels, act_fun, kernel_size)
dense_block(out_features, act_fun)


Classes
---------
RescaleLayer()
TorchModel(base.base_model.BaseModel)

"""

import torch
from base.base_model import BaseModel
import matplotlib.pyplot as plt


# https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict


def conv_couplet(in_channels, out_channels, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        getattr(torch.nn, act_fun)(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
    )


def dense_lazy_couplet(out_features, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.LazyLinear(out_features=out_features, bias=True),
        getattr(torch.nn, act_fun)(),
    )


def conv_block(in_channels, out_channels, act_fun, kernel_size):
    block = [
        conv_couplet(in_channels, out_channels, act_fun, kernel_size, padding="same")
        for in_channels, out_channels, act_fun, kernel_size in zip(
            [*in_channels],
            [*out_channels],
            [*act_fun],
            [*kernel_size],
        )
    ]
    return torch.nn.Sequential(*block)


def dense_block(out_features, act_fun):
    block = [
        dense_lazy_couplet(out_channels, act_fun)
        for out_channels, act_fun in zip([*out_features], [*act_fun])
    ]
    return torch.nn.Sequential(*block)


class RescaleLayer:
    def __init__(self, scale, offset):
        self.offset = offset
        self.scale = scale

    def __call__(self, x):
        x = torch.multiply(x, self.scale)
        x = torch.add(x, self.offset)
        return x


class TorchModel(BaseModel):
    def __init__(self, config, target=None):
        super().__init__()

        self.config = config

        assert len(self.config["cnn_act"]) == len(self.config["kernel_size"]) == len(self.config["filters"])
        assert len(self.config["hiddens_block"]) == len(self.config["hiddens_block_act"])

        if target is None:
            self.target_mean = torch.tensor(0.0)
            self.target_std = torch.tensor(1.0)
        else:
            self.target_mean = torch.tensor(target.mean(axis=0))
            self.target_std = torch.tensor(target.std(axis=0))

        # Longitude padding
        self.pad_lons = torch.nn.CircularPad2d(config["circular_padding"])

        # CNN block
        self.cnn_block = conv_block(
            [1, *config["filters"][:-1]],
            [*config["filters"]],
            [*config["cnn_act"]],
            [*config["kernel_size"]],
        )

        # Flat layer
        self.flat = torch.nn.Flatten(start_dim=1)

        # Dense blocks
        self.denseblock_mu = dense_block(
            config["hiddens_block"], config["hiddens_block_act"]
        )
        self.denseblock_sigma = dense_block(
            config["hiddens_block"], config["hiddens_block_act"]
        )
        self.denseblock_gamma = dense_block(
            config["hiddens_block"], config["hiddens_block_act"]
        )
        self.denseblock_tau = dense_block(
            config["hiddens_block"], config["hiddens_block_act"]
        )

        # Final dense layer
        self.finaldense_mu = dense_lazy_couplet(
            out_features=config["hiddens_final"], act_fun=config["hiddens_final_act"]
        )
        self.finaldense_sigma = dense_lazy_couplet(
            out_features=config["hiddens_final"], act_fun=config["hiddens_final_act"]
        )
        self.finaldense_gamma = dense_lazy_couplet(
            out_features=config["hiddens_final"], act_fun=config["hiddens_final_act"]
        )
        self.finaldense_tau = dense_lazy_couplet(
            out_features=config["hiddens_final"], act_fun=config["hiddens_final_act"]
        )

        # Rescaling layers
        self.rescale_mu = RescaleLayer(self.target_std, self.target_mean)
        self.rescale_sigma = RescaleLayer(torch.tensor(1.0), torch.log(self.target_std))
        self.rescale_tau = RescaleLayer(torch.tensor(0.0), torch.tensor(1.0))

        # Output layers
        self.output_mu = torch.nn.Linear(
            in_features=config["hiddens_final"], out_features=1, bias=True
        )
        self.output_sigma = torch.nn.Linear(
            in_features=config["hiddens_final"], out_features=1, bias=True
        )
        self.output_gamma = torch.nn.Linear(
            in_features=config["hiddens_final"], out_features=1, bias=True
        )
        self.output_tau = torch.nn.Linear(
            in_features=config["hiddens_final"], out_features=1, bias=True
        )

    def forward(self, x, x_unit):

        x = self.pad_lons(x)
        x = self.cnn_block(x)
        x = self.flat(x)

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
