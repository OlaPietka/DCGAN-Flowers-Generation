from torch import nn


class Discriminator(nn.Module):
    def __init__(self, n_channels: int, hidden_dim: int):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            self.make_disc_block(n_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            self.make_disc_block(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, normalize=True),
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1, normalize=True),
            self.make_disc_block(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1, normalize=True),
            self.make_disc_block(hidden_dim * 8, hidden_dim * 16, kernel_size=4, stride=2, padding=1, normalize=True),
            self.make_disc_block(hidden_dim * 16, 1, kernel_size=4, stride=1, padding=0, activation=nn.Sigmoid()),
        )

    @staticmethod
    def make_disc_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            normalize: bool = False,
            activation: nn.Module = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    ) -> nn.Sequential:
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(activation)

        return nn.Sequential(*layers)

    def forward(self, image):
        disc_pred = self.disc(image)

        return disc_pred.view(len(disc_pred), -1)


class Generator(nn.Module):
    def __init__(self, n_channels: int, z_dim: int, hidden_dim: int):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 16, kernel_size=4, stride=1, padding=0, normalize=True),
            self.make_gen_block(hidden_dim * 16, hidden_dim * 8, kernel_size=4, stride=2, padding=1, normalize=True),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1, normalize=True),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1, normalize=True),
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1, normalize=True),
            self.make_gen_block(hidden_dim, n_channels, kernel_size=4, stride=2, padding=1, activation=nn.Tanh()),
        )

    @staticmethod
    def make_gen_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            normalize: bool = False,
            activation: nn.Module = nn.ReLU(inplace=True)
    ) -> nn.Sequential:
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(activation)

        return nn.Sequential(*layers)

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)

        return self.gen(x)
