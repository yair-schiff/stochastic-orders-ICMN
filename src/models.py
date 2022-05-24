import torch
from torch import nn

from src.choquet_utils import (
    generate_interpolates, get_model_grad_wrt_gen_params, get_model_grad_wrt_interpolates,
    count_nonzero_weights, get_weight_norms, project_weights_to_positive, test_model_convexity,
)


class MaxOut2D(nn.Module):
    """
    Pytorch implementation of MaxOut on channels for an input that is C x H x W.
    Reshape input from N x C x H x W --> N x H*W x C --> perform MaxPool1D on dim 2, i.e. channels --> reshape back to
    N x C//maxout_kernel x H x W.
    """
    def __init__(self, max_out):
        super(MaxOut2D, self).__init__()
        self.max_out = max_out
        self.max_pool = nn.MaxPool1d(max_out)

    def forward(self, x):
        batch_size = x.shape[0]
        channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        # Reshape input from N x C x H x W --> N x H*W x C
        x_reshape = torch.permute(x, (0, 2, 3, 1)).view(batch_size, height * width, channels)
        # Pool along channel dims
        x_pooled = self.max_pool(x_reshape)
        # Reshape back to N x C//maxout_kernel x H x W.
        return torch.permute(x_pooled, (0, 2, 1)).view(batch_size, channels // self.max_out, height, width).contiguous()


class DistributionGenerator(nn.Module):
    def __init__(self, dim=32, dimh=64, num_layers=2, activation='relu', max_out=4, dropout=False):
        super(DistributionGenerator, self).__init__()

        non_linearity = nn.MaxPool1d(kernel_size=max_out) if activation == 'max_out' else nn.ReLU()
        dimh_adjusted = dimh // max_out if activation == 'max_out' else dimh
        if dropout:
            input_to_hidden = nn.Sequential(
                nn.Linear(dim, dimh),
                non_linearity,
                nn.Dropout()
            )
            layers = []
            for _ in range(num_layers - 1):
                layers += [nn.Sequential(nn.Linear(dimh_adjusted, dimh), non_linearity, nn.Dropout())]
            main = nn.Sequential(*layers)

        else:
            input_to_hidden = nn.Sequential(
                nn.Linear(dim, dimh),
                non_linearity,
            )
            layers = []
            for _ in range(num_layers - 1):
                layers += [nn.Sequential(nn.Linear(dimh_adjusted, dimh), non_linearity)]
            main = nn.Sequential(*layers)

        self.residual = nn.Linear(dim, dimh_adjusted)
        self.input_to_hidden = input_to_hidden
        self.main = main
        self.main_output = nn.Linear(dimh_adjusted, 2)

    def forward(self, x):
        output = self.input_to_hidden(x)
        for i, layer in enumerate(self.main):
            output = layer(output) + self.residual(x)
        output = self.main_output(output)
        return output

    @staticmethod
    def get_model_args_as_dict(args):
        return {
            "dim": args.z_dim,
            "dimh": args.g_hidden_dim,
            "num_layers": args.g_n_layers,
            'activation': args.activation,
            'max_out': args.max_out,
            'dropout': args.dropout,
        }


class DistributionDiscriminator(nn.Module):
    def __init__(self, dim=2, dimh=64, num_layers=2, activation='relu', max_out=4, dropout=False):
        super(DistributionDiscriminator, self).__init__()

        non_linearity = nn.MaxPool1d(kernel_size=max_out) if activation == 'max_out' else nn.ReLU()
        dimh_adjusted = dimh // max_out if activation == 'max_out' else dimh
        if dropout:
            input_to_hidden = nn.Sequential(
                nn.Linear(dim, dimh),
                non_linearity,
                nn.Dropout()
            )
            layers = []
            for _ in range(num_layers - 1):
                layers += [nn.Sequential(nn.Linear(dimh_adjusted, dimh), non_linearity, nn.Dropout())]
            main = nn.Sequential(*layers)

        else:
            input_to_hidden = nn.Sequential(
                nn.Linear(dim, dimh),
                non_linearity,
            )
            layers = []
            for _ in range(num_layers - 1):
                layers += [nn.Sequential(nn.Linear(dimh_adjusted, dimh), non_linearity)]
            main = nn.Sequential(*layers)

        self.residual = nn.Linear(dim, dimh_adjusted)
        self.input_to_hidden = input_to_hidden
        self.main = main
        self.main_output = nn.Linear(dimh_adjusted, 1)

    def forward(self, x):
        output = self.input_to_hidden(x)
        for layer in self.main:
            output = layer(output) + self.residual(x)
        output = self.main_output(output)
        return output

    @staticmethod
    def get_model_args_as_dict(args):
        return {
            'dim': 2,
            'dimh': args.d_hidden_dim,
            'num_layers': args.d_n_layers,
            'activation': args.activation,
            'max_out': args.max_out,
            'dropout': args.dropout,
        }


class MnistGenerator(nn.Module):
    """
    This class is largely based on the generator from:
    https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
    """

    def __init__(self, dim=32, dimh=64, output_dim=(1, 28, 28), activation='relu', max_out=0, dropout=False):
        super(MnistGenerator, self).__init__()
        self.dimh_adjusted = dimh if activation == 'relu' else dimh // max_out
        self.output_dim = (-1, *output_dim)

        if dropout:
            preprocess = nn.Sequential(
                nn.Linear(dim, 4 * 4 * 4 * dimh),
                nn.ReLU(True) if activation == 'relu' else nn.MaxPool1d(kernel_size=max_out),
                nn.Dropout()
            )
            block1 = nn.Sequential(
                nn.ConvTranspose2d(4 * self.dimh_adjusted, 2 * dimh, (5, 5)),
                nn.ReLU(True) if activation == 'relu' else MaxOut2D(max_out=max_out),
                nn.Dropout2d()
            )
            block2 = nn.Sequential(
                nn.ConvTranspose2d(2 * self.dimh_adjusted, dimh, (5, 5)),
                nn.ReLU(True) if activation == 'relu' else MaxOut2D(max_out=max_out),
                nn.Dropout2d()
            )
        else:
            preprocess = nn.Sequential(
                nn.Linear(dim, 4 * 4 * 4 * dimh),
                nn.ReLU(True) if activation == 'relu' else MaxOut2D(max_out=max_out),
            )
            block1 = nn.Sequential(
                nn.ConvTranspose2d(4 * self.dimh_adjusted, 2 * dimh, (5, 5)),
                nn.ReLU(True) if activation == 'relu' else MaxOut2D(max_out=max_out),
            )
            block2 = nn.Sequential(
                nn.ConvTranspose2d(2 * self.dimh_adjusted, dimh, (5, 5)),
                nn.ReLU(True) if activation == 'relu' else MaxOut2D(max_out=max_out),
            )
        deconv_out = nn.ConvTranspose2d(self.dimh_adjusted, 1, (8, 8), stride=(2, 2))

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.preprocess(x)
        output = output.view(-1, 4 * self.dimh_adjusted, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(self.output_dim)

    @staticmethod
    def get_model_args_as_dict(args):
        return {
            'dim': args.z_dim,
            'dimh': args.g_hidden_dim,
            'output_dim': (1, 28, 28),
            'activation': args.activation,
            'max_out': args.max_out,
            'dropout': args.dropout,
        }


class MnistDiscriminator(nn.Module):
    """
    This class is largely based on the generator from:
    https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
    """

    def __init__(self, dim=1, dimh=64, activation='relu', max_out=0, dropout=False):
        super(MnistDiscriminator, self).__init__()
        self.dimh_adjusted = dimh if activation == 'relu' else dimh // max_out

        if dropout:
            input_to_hidden = nn.Sequential(
                nn.Conv2d(dim, dimh, (5, 5), stride=(2, 2), padding=2),
                nn.ReLU(True) if activation == 'relu' else MaxOut2D(max_out=max_out),
                nn.Dropout2d(),
            )
            main = nn.Sequential(
                nn.Conv2d(self.dimh_adjusted, 2*dimh, (5, 5), stride=(2, 2), padding=2),
                nn.ReLU(True) if activation == 'relu' else MaxOut2D(max_out=max_out),
                nn.Dropout2d(),
                nn.Conv2d(2*self.dimh_adjusted, 4 * dimh, (5, 5), stride=(2, 2), padding=2),
                nn.ReLU(True) if activation == 'relu' else MaxOut2D(max_out=max_out),
                nn.Dropout2d(),
            )
        else:
            input_to_hidden = nn.Sequential(
                nn.Conv2d(dim, dimh, (5, 5), stride=(2, 2), padding=2),
                nn.ReLU(True) if activation == 'relu' else MaxOut2D(max_out=max_out),
            )
            main = nn.Sequential(
                nn.Conv2d(dimh, 2*self.dimh_adjusted, (5, 5), stride=(2, 2), padding=2),
                nn.ReLU(True) if activation == 'relu' else MaxOut2D(max_out=max_out),
                nn.Conv2d(2*self.dimh_adjusted, 4 * dimh, (5, 5), stride=(2, 2), padding=2),
                nn.ReLU(True) if activation == 'relu' else MaxOut2D(max_out=max_out),
            )
        self.input_to_hidden = input_to_hidden
        self.main = main
        self.main_output = nn.Linear(64 * self.dimh_adjusted, 1)

    def forward(self, x):
        out = self.input_to_hidden(x)
        out = self.main(out)
        out = out.view(out.shape[0], -1)
        out = self.main_output(out)
        return out

    @staticmethod
    def get_model_args_as_dict(args):
        return {
            'dim': 1,
            'dimh': args.d_hidden_dim,
            'activation': args.activation,
            'max_out': args.max_out,
            'dropout': args.dropout,
        }


class UpsampleConv(nn.Module):
    """
    Code from:
    https://github.com/ozanciga/gans-with-pytorch/blob/2071efd166935f0b4fb321227e94aa2ad1cfa273/wgan-gp/models.py#L29
    """

    def __init__(self, n_input, n_output, k_size):
        super(UpsampleConv, self).__init__()

        self.model = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(n_input, n_output, k_size, stride=(1, 1), padding=(k_size - 1) // 2, bias=True)
        )

    def forward(self, x):
        x = x.repeat((1, 4, 1, 1))  # Weird concat of WGAN-GPs upsampling process.
        out = self.model(x)
        return out


class ResidualBlock(nn.Module):
    """
    Code from:
    https://github.com/ozanciga/gans-with-pytorch/blob/2071efd166935f0b4fb321227e94aa2ad1cfa273/wgan-gp/models.py#L29
    """

    def __init__(self, n_input, n_output, k_size, resample='up', bn=True, spatial_dim=None):
        super(ResidualBlock, self).__init__()

        self.resample = resample

        if resample == 'up':
            self.conv1 = UpsampleConv(n_input, n_output, k_size)
            self.conv2 = nn.Conv2d(n_output, n_output, k_size, padding=(k_size - 1) // 2)
            self.conv_shortcut = UpsampleConv(n_input, n_output, k_size)
            self.out_dim = n_output
        else:
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size - 1) // 2)
            self.conv2 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size - 1) // 2)
            self.conv_shortcut = None  # Identity
            self.out_dim = n_input
            self.ln_dims = [n_input, spatial_dim, spatial_dim]

        self.model = nn.Sequential(
            nn.BatchNorm2d(n_input) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True),
            self.conv1,
            nn.BatchNorm2d(self.out_dim) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True),
            self.conv2,
        )

    def forward(self, x):
        if self.conv_shortcut is None:
            return x + self.model(x)
        else:
            return self.conv_shortcut(x) + self.model(x)


class MaxOutResidualBlock(nn.Module):
    def __init__(self, n_input, k_size, max_out, upsample_dim=None, dropout=False):
        super(MaxOutResidualBlock, self).__init__()

        if upsample_dim:
            self.conv1 = UpsampleConv(n_input // max_out, upsample_dim, k_size)
            self.conv2 = nn.Conv2d(upsample_dim // max_out, upsample_dim, k_size, padding=(k_size - 1) // 2)
            self.conv_shortcut = UpsampleConv(n_input, upsample_dim, k_size)
            self.out_dim = upsample_dim

        else:
            self.conv1 = nn.Conv2d(n_input // max_out, n_input, k_size, padding=(k_size - 1) // 2)
            self.conv2 = nn.Conv2d(n_input // max_out, n_input, k_size, padding=(k_size - 1) // 2)
            self.conv_shortcut = None

        if dropout:
            self.model = nn.Sequential(
                MaxOut2D(max_out=max_out),
                nn.Dropout2d(),
                self.conv1,
                MaxOut2D(max_out=max_out),
                nn.Dropout2d(),
                self.conv2,
            )
        else:
            self.model = nn.Sequential(
                MaxOut2D(max_out=max_out),
                self.conv1,
                MaxOut2D(max_out=max_out),
                self.conv2,
            )

    def forward(self, x):
        if self.conv_shortcut is None:
            return x + self.model(x)
        else:
            return self.conv_shortcut(x) + self.model(x)

    def register_convex_modules(self):
        return self.conv1, self.conv2


class Cifar10ResidualGenerator(nn.Module):
    """
    Code from:
    https://github.com/ozanciga/gans-with-pytorch/blob/2071efd166935f0b4fb321227e94aa2ad1cfa273/wgan-gp/models.py#L29
    """

    def __init__(self, dim, dimh, activation='max_out', max_out=4, dropout=False):
        super(Cifar10ResidualGenerator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(dim, dimh, (4, 4), (1, 1), (0, 0)),
            ResidualBlock(dimh, dimh, 3, resample='up') if activation == 'relu'
            else MaxOutResidualBlock(n_input=dimh, k_size=3, max_out=max_out, upsample_dim=dimh, dropout=dropout),
            ResidualBlock(dimh, dimh, 3, resample='up') if activation == 'relu'
            else MaxOutResidualBlock(n_input=dimh, k_size=3, max_out=max_out, upsample_dim=dimh, dropout=dropout),
            ResidualBlock(dimh, dimh, 3, resample='up') if activation == 'relu'
            else MaxOutResidualBlock(n_input=dimh, k_size=3, max_out=max_out, upsample_dim=dimh, dropout=dropout),
            nn.BatchNorm2d(dimh),
            nn.ReLU(inplace=True) if activation == 'relu' else MaxOut2D(max_out=max_out),
            nn.Dropout(),
            nn.Conv2d(dimh // (1 if activation == 'relu' else max_out), 3, (3, 3), padding=(3 - 1) // 2),  # 3 x 32 x 32
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z.unsqueeze(2).unsqueeze(3))
        return img

    @staticmethod
    def get_model_args_as_dict(args):
        return {
            'dim': args.z_dim,
            'dimh': args.g_hidden_dim,
            'activation': args.activation,
            'max_out': args.max_out,
            'dropout': args.dropout,
        }


class Cifar10Discriminator(nn.Module):
    """
    This class is largely based on the generator from:
    https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    """

    def __init__(self, dim=3, dimh=64, activation='relu',
                 max_out=4, dropout=False):
        super(Cifar10Discriminator, self).__init__()

        non_linearity = MaxOut2D(max_out=max_out) if activation == 'max_out' else nn.ReLU()
        dimh_adjusted = dimh // max_out if activation == 'max_out' else dimh
        if dropout:
            input_to_hidden = nn.Sequential(
                nn.Conv2d(dim, dimh, (3, 3), (2, 2), padding=1),  # output: [bsz, dimh, 16, 16]
                non_linearity,
                nn.Dropout2d()
            )

            main = nn.Sequential(
                nn.Conv2d(dimh_adjusted, 2 * dimh, (3, 3), (2, 2), padding=1),  # output: [bsz, 2*dimh, 8, 8]
                non_linearity,
                nn.Dropout2d(),
                nn.Conv2d(2 * dimh_adjusted, 4 * dimh, (3, 3), (2, 2), padding=1),  # output: [bsz, 4*dimh, 4, 4]
                non_linearity,
                nn.Dropout2d()
            )

        else:
            input_to_hidden = nn.Sequential(
                nn.Conv2d(dim, dimh, (3, 3), (2, 2), padding=1),  # output: [bsz, dimh, 16, 16]
                non_linearity
            )
            main = nn.Sequential(
                nn.Conv2d(dimh_adjusted, 2 * dimh, (3, 3), (2, 2), padding=1),  # output: [bsz, 2*dimh, 8, 8]
                non_linearity,
                nn.Conv2d(2 * dimh_adjusted, 4 * dimh, (3, 3), (2, 2), padding=1),  # output: [bsz, 4*dimh, 4, 4]
                non_linearity
            )

        self.input_to_hidden = input_to_hidden
        self.main = main
        self.main_output = nn.Linear(4 * 4 * 4 * dimh_adjusted, 1)

    def forward(self, x):
        output = self.input_to_hidden(x)
        output = self.main(output)  # output: [bsz, 64*dimh]
        output = self.main_output(output.view(output.shape[0], -1))
        return output

    @staticmethod
    def get_model_args_as_dict(args):
        return {
            'dim': 3,
            'dimh': args.d_hidden_dim,
            'activation': args.activation,
            'max_out': args.max_out,
            'dropout': args.dropout,
        }


class CTDiscrepancy(nn.Module):
    """
    D_CT = inf_{u in cvx} int u d(mu_{+} - mu_{-}) + 0.5*int ||x||^2 d(mu_{-} - mu_{+})
    where mu_{+} is distribution on generated data and mu_{-} is distribution on real data.
    When training the discriminator (u), the norm term should not carry torch gradient.
    """

    def __init__(self, critic, name='ct_disc'):
        super(CTDiscrepancy, self).__init__()
        self.critic = critic
        self.positive_weight_module_names = ['main', 'main_output']
        self.name = name

    def forward(self, x):
        return self.critic(x)

    def objective(self, gen_data, real_data):
        """
        Choquet objective
        :param gen_data: Generated data
        :param real_data: Ground truth / Real data
        :return: dict with objective values. key `objective` contains term for .backward()
        """
        u_integral = self.critic(gen_data).mean() - self.critic(real_data).mean()
        objective = u_integral
        if self.critic.training:
            with torch.no_grad():
                data_norm = 0.5 * (torch.linalg.vector_norm(real_data, ord=2, dim=1) ** 2).mean()
                data_norm -= 0.5 * (torch.linalg.vector_norm(gen_data, ord=2, dim=1) ** 2).mean()
        else:
            data_norm = 0.5 * (torch.linalg.vector_norm(real_data, ord=2, dim=1) ** 2).mean()
            data_norm -= 0.5 * (torch.linalg.vector_norm(gen_data, ord=2, dim=1) ** 2).mean()
            objective += data_norm
        return {'objective': objective, 'u_integral': u_integral, 'data_norm': data_norm}

    def grad_reg_wrt_gen_params(self, z, generator, grad_reg_lambda):
        grad_norm = get_model_grad_wrt_gen_params(self.critic, z, generator)
        return grad_norm * grad_reg_lambda

    def grad_reg_wrt_interpolates(self, real_data, fake_data, grad_reg_lambda):
        grad_norm = get_model_grad_wrt_interpolates(self.critic, real_data, fake_data)
        return grad_norm * grad_reg_lambda

    def get_non_pos_params(self):
        params = {}
        for mod in self.non_positive_weight_module_names:
            params[f'critic_{mod}'] = dict(self.critic.named_modules())[mod]
        return nn.ModuleDict(params).parameters()

    def get_pos_params(self):
        pos_params = {}
        for pos_mod in self.positive_weight_module_names:
            pos_params[f'critic_{pos_mod}'] = dict(self.critic.named_modules())[pos_mod]
        return nn.ModuleDict(pos_params).parameters()

    def project_critic_weights_to_positive(self):
        project_weights_to_positive(self.critic, self.positive_weight_module_names)

    def log_critic_weight_norms(self, log):
        weight_norms = get_weight_norms(self.critic)
        for k, v in weight_norms.items():
            if v:
                log(f'weights/{self.name}_{k}', v)

    def log_critic_convexity(self, batch, log):
        convex_percentage = test_model_convexity(self.critic, batch)
        log(f'convexity/{self.name}_u', convex_percentage)
        return {f'{self.name}_u': convex_percentage}

    def log_critic_nonzero_weights(self, log):
        nonzero_percentage = count_nonzero_weights(self.critic, self.positive_weight_module_names)
        log(f'nonzero/{self.name}_u', nonzero_percentage)
        return {f'{self.name}_u': nonzero_percentage}


class VariationalDominanceCriterion(CTDiscrepancy):
    """
    VDC = inf_{u in cvx} int u d(mu_{+} - mu_{-})
    where mu_{+} is distribution on generated data and mu_{-} is distribution on real data.
    Same objective as d_CT, but without data norm term.
    """
    def __init__(self, critic, name='vdc'):
        super().__init__(critic, name)

    def objective(self, dominating_data, dominated_data):
        """
        Choquet objective
        :param dominating_data: Data from dominating distribution
        :param dominated_data: Data from distribution that is to be dominated
        :return: E[u(dominating)] - E[u(dominated)]
        """
        critic_on_dominating = self.critic(dominating_data).mean()
        critic_on_dominated = self.critic(dominated_data).mean()
        u_integral = critic_on_dominating - critic_on_dominated
        return u_integral

    def grad_reg_wrt_interpolates(self, data0, data1, grad_reg_lambda):
        grad_norm = get_model_grad_wrt_interpolates(self.critic, data0, data1)
        return grad_norm * grad_reg_lambda

    def reg_u_squared(self, data0, data1, reg_lambda):
        critic_squared_on_data0 = torch.square(self.critic(data0)).mean()
        critic_squared_on_data1 = torch.square(self.critic(data1)).mean()
        return reg_lambda*(critic_squared_on_data0 + critic_squared_on_data1)


class CTDistance(nn.Module):
    """
    d_CT = inf_{u_0 in cvx} int u_0 d(mu_{+} - mu_{-}) + inf_{u_1 in cvx} int u_1 d(mu_{-} - mu_{+})
    where mu_{+} is distribution on generated data and mu_{-} is distribution on real data.
    We experiment with different ways of combining the integral terms above:
        - `sum`
        - `min`
    """

    def __init__(self, critics, how_to_combine_integral_terms='sum', split_regularization=False, name='ct_dist'):
        super(CTDistance, self).__init__()
        self.critic_0 = critics[0]
        self.critic_1 = critics[1]
        self.how_to_combine_integral_terms = how_to_combine_integral_terms
        self.split_regularization = split_regularization
        self.non_positive_weight_module_names = ['input_to_hidden']
        self.positive_weight_module_names = ['main', 'main_output']
        self.name = name

    def forward(self, x):
        return self.critic_0(x), self.critic_1(x)

    def objective(self, gen_data, real_data):
        """
        Choquet objective
        :param gen_data: Generated data
        :param real_data: Ground truth / Real data
        :return: dict with objective values. Each critic receives its own objective value (i.e., u{i}_integral) for
        .backward() calls
        """
        critics_on_gen_data = self(gen_data)
        critics_on_real_data = self(real_data)
        u0_integral = critics_on_gen_data[0].mean() - critics_on_real_data[0].mean()
        u1_integral = critics_on_real_data[1].mean() - critics_on_gen_data[1].mean()
        if self.how_to_combine_integral_terms == 'sum':
            objective = u0_integral + u1_integral
        elif self.how_to_combine_integral_terms == 'min':
            if u0_integral < u1_integral:
                objective = u0_integral
                u1_integral = u1_integral.detach()
            else:
                objective = u1_integral
                u0_integral = u0_integral.detach()
        else:
            raise NotImplementedError
        return {'objective': objective, 'u0_integral': u0_integral, 'u1_integral': u1_integral}

    def grad_reg_wrt_gen_params(self, z, generator, grad_reg_lambda):
        gen_z = generator(z)
        crit_out = self(gen_z)
        gr_norm_sq = None
        gen_params_with_grad = [g for g in generator.parameters() if g.requires_grad]
        if self.split_regularization:
            gr_norm_sq = get_model_grad_wrt_gen_params(self.critic_0, z, generator)
            gr_norm_sq += get_model_grad_wrt_gen_params(self.critic_1, z, generator)
        else:
            grads = torch.autograd.grad(crit_out[0].mean() - crit_out[1].mean(), gen_params_with_grad,
                                        create_graph=True, retain_graph=True)[0]
            for gr in grads:
                if gr_norm_sq is None:
                    gr_norm_sq = (gr ** 2).sum()
                else:
                    gr_norm_sq += (gr ** 2).sum()
        return grad_reg_lambda * gr_norm_sq

    def grad_reg_wrt_interpolates(self, real_data, fake_data, grad_reg_lambda):
        interpolates = generate_interpolates(real_data, fake_data)
        if self.split_regularization:
            # critic_0
            grad_penalty = get_model_grad_wrt_interpolates(self.critic_0,
                                                           real_data=None, fake_data=None, interpolates=interpolates)

            # critic_1
            grad_penalty += get_model_grad_wrt_interpolates(self.critic_1,
                                                            real_data=None, fake_data=None, interpolates=interpolates)
        else:
            interpolates = interpolates.detach().clone().requires_grad_(True)
            crit_interpolates_diff = self.critic_0(interpolates) - self.critic_1(interpolates)
            gradients = torch.autograd.grad(outputs=crit_interpolates_diff, inputs=interpolates,
                                            grad_outputs=torch.ones(crit_interpolates_diff.size()).type_as(
                                                crit_interpolates_diff),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty * grad_reg_lambda

    def get_non_pos_params(self):
        params = {}
        for i, crit in enumerate([self.critic_0, self.critic_1]):
            for mod in self.non_positive_weight_module_names:
                params[f'critic{i}_{mod}'] = dict(crit.named_modules())[mod]
        return nn.ModuleDict(params).parameters()

    def get_pos_params(self):
        pos_params = {}
        for i, crit in enumerate([self.critic_0, self.critic_1]):
            for pos_mod in self.positive_weight_module_names:
                pos_params[f'critic{i}_{pos_mod}'] = dict(crit.named_modules())[pos_mod]
        return nn.ModuleDict(pos_params).parameters()

    def project_critic_weights_to_positive(self):
        for crit in [self.critic_0, self.critic_1]:
            project_weights_to_positive(crit, self.positive_weight_module_names)

    def log_critic_convexity(self, batch, log):
        convex_percentages = {}
        for i, crit in enumerate([self.critic_0, self.critic_1]):
            convex_percentage = test_model_convexity(crit, batch)
            convex_percentages[f'{self.name}_u_{i}'] = convex_percentage
            log(f'convexity/{self.name}_u_{i}', convex_percentage)
        return convex_percentages

    def log_critic_weight_norms(self, log):
        for i, crit in enumerate([self.critic_0, self.critic_1]):
            weight_norms = get_weight_norms(crit)
            for k, v in weight_norms.items():
                if v:
                    log(f'weights/{self.name}_u_{i}/{k}', v)

    def log_critic_nonzero_weights(self, log):
        nonzero_percentages = {}
        for i, crit in enumerate([self.critic_0, self.critic_1]):
            nonzero_percentage = count_nonzero_weights(crit, self.positive_weight_module_names)
            nonzero_percentages[f'{self.name}_u_{i}'] = nonzero_percentage
            log(f'nonzero/{self.name}_u_{i}', nonzero_percentage)
        return nonzero_percentages
