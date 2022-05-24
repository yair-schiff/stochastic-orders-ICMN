import copy

import torch
from torch.optim import SGD, Adam


from src.models import (
    DistributionGenerator, DistributionDiscriminator,
    MnistGenerator, MnistDiscriminator,
    Cifar10ResidualGenerator, Cifar10Discriminator,
)

MODELS = {
    'distribution_generator': DistributionGenerator,
    'distribution_discriminator': DistributionDiscriminator,
    'mnist_generator': MnistGenerator,
    'mnist_discriminator': MnistDiscriminator,
    'cifar10_residual_generator': Cifar10ResidualGenerator,
    'cifar10_discriminator': Cifar10Discriminator,
}


def get_model_args_as_dict(model_type, args):
    assert model_type in MODELS.keys(), \
        f'Model {model_type} not supported. Pass one of the allowed models: {list(MODELS.keys())}.'
    return MODELS[model_type].get_model_args_as_dict(args)


def get_model(model_type, model_args):
    assert model_type in MODELS.keys(), \
        f'Model {model_type} not supported. Pass one of the allowed models: {list(MODELS.keys())}.'
    return MODELS[model_type](**model_args)


def get_generator_optimizer_args_as_dict(args):
    generator_optim_args = {
        'lr': args.g_lr,
        'weight_decay': args.g_weight_decay
    }
    if args.generator_optim_type == 'adam' or args.generator_optim_type == 'lamb':
        generator_optim_args['betas'] = tuple(args.g_betas)
        generator_optim_args['eps'] = args.g_eps
    return generator_optim_args


def get_discriminator_optimizer_args_as_dict(args):
    discriminator_optim_args = {
        'lr': args.d_lr,
        'weight_decay': args.d_weight_decay,
    }
    if args.discriminator_optim_type == 'adam' or args.discriminator_optim_type == 'lamb':
        discriminator_optim_args['betas'] = tuple(args.d_betas)
        discriminator_optim_args['eps'] = args.d_eps
    return discriminator_optim_args


def get_optimizer(optim_type, optim_args, params):
    OPTIMIZERS = {'sgd': SGD, 'adam': Adam}
    assert optim_type in OPTIMIZERS.keys(), \
        f'Optimizer {optim_type} not supported. Pass one of the allowed optimizers: {list(OPTIMIZERS.keys())}.'
    optim_args = copy.deepcopy(optim_args)  # copy optim args to avoid errors in checkpointing
    optim_args['params'] = params
    return OPTIMIZERS[optim_type](**optim_args)


def calculate_grad_wrt_x(model, x):
    x_clone = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        fx = model(x_clone).mean()
        nabla = torch.autograd.grad(fx, x_clone, create_graph=True, retain_graph=True)[0]
    return nabla
