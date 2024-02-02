import torch
from base.base_model import BaseModel


class RescaleLayer():
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
            self.target_mean = 0.0
            self.target_std = 1.0
        else:
            self.target_mean = torch.tensor(target.mean(axis=0), dtype=torch.float32)
            self.target_std = torch.tensor(target.std(axis=0), dtype=torch.float32)

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flat = torch.nn.Flatten(start_dim=1)

        self.fc1_lzx10 = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features=10, bias=True),
            torch.nn.ReLU(),
        )

        self.dense_block = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features=10, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=10, out_features=10, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=10, out_features=10, bias=True),
            torch.nn.ReLU(),
        )

        self.rescale_mu = RescaleLayer(self.target_std, self.target_mean)
        self.rescale_sigma = RescaleLayer(torch.tensor(1.0), torch.log(self.target_std))
        self.rescale_tau = RescaleLayer(torch.tensor(0.0), torch.tensor(1.0))

        self.output_mu = torch.nn.Linear(in_features=10, out_features=1, bias=True)
        self.output_sigma = torch.nn.Linear(in_features=10, out_features=1, bias=True)
        self.output_gamma = torch.nn.Linear(in_features=10, out_features=1, bias=True)
        self.output_tau = torch.nn.Linear(in_features=10, out_features=1, bias=True)

    def forward(self, x, x_unit):

        x = self.conv5(x)
        x = self.conv3(x)
        x = self.conv3(x)

        x = self.flat(x)

        # build mu_layers
        x_mu = torch.cat((x, x_unit[:, None]), dim=-1)
        x_mu = self.dense_block(x_mu)
        x_mu = torch.cat((x_mu, x_unit[:, None]), dim=-1)
        x_mu = self.fc1_lzx10(x_mu)
        mu_out = self.output_mu(x_mu)

        # build sigma_layers
        x_sigma = torch.cat((x, x_unit[:, None]), dim=-1)
        x_sigma = self.dense_block(x_sigma)
        x_sigma = torch.cat((x_sigma, x_unit[:, None]), dim=-1)
        x_sigma = self.fc1_lzx10(x_sigma)
        sigma_out = self.output_sigma(x_sigma)

        # build gamma_layers
        x_gamma = torch.cat((x, x_unit[:, None]), dim=-1)
        x_gamma = self.dense_block(x_gamma)
        x_gamma = torch.cat((x_gamma, x_unit[:, None]), dim=-1)
        x_gamma = self.fc1_lzx10(x_gamma)
        gamma_out = self.output_gamma(x_gamma)

        # build tau_layers
        tau_out = self.output_tau(x_gamma)

        # rescaling layers
        mu_out = self.rescale_mu(mu_out)

        sigma_out = self.rescale_sigma(sigma_out)
        sigma_out = torch.exp(sigma_out)

        tau_out = self.rescale_tau(tau_out)

        # final output, concatenate parameters together
        x = torch.cat((mu_out, sigma_out, gamma_out, tau_out), dim=-1)

        return x
