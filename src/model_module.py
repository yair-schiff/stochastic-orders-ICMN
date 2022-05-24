import os
from collections import OrderedDict

import pytorch_lightning as pl
import torch

from src.choquet_utils import get_model_grad_wrt_interpolates
from src.model_utils import (
    get_model, get_model_args_as_dict,
    get_generator_optimizer_args_as_dict, get_discriminator_optimizer_args_as_dict, get_optimizer
)
from src.models import CTDiscrepancy, CTDistance, VariationalDominanceCriterion


class WGanModule(pl.LightningModule):
    def __init__(self,
                 z_dim,
                 grad_reg_lambda,
                 generator_display_function,
                 generator_model_type, generator_model_args, generator_optim_type, generator_optim_args,
                 discriminator_model_type, discriminator_model_args, discriminator_optim_type, discriminator_optim_args,
                 train_gen_every,
                 gen_viz_every,
                 log_images_to_tb=False
                 ):
        super().__init__()
        self.automatic_optimization = False

        self.train_gen_every = train_gen_every
        self.train_generator = 0
        self.train_discriminator = train_gen_every - 1
        self.z_dim = z_dim

        self.generator_display_function = generator_display_function
        self.gen_viz_every = gen_viz_every
        self.log_images_to_tb = log_images_to_tb

        self.grad_reg_lambda = grad_reg_lambda

        self.generator = get_model(generator_model_type, generator_model_args)
        self.opt_g = None
        self.opt_g_type = generator_optim_type
        self.opt_g_args = generator_optim_args

        self.discriminator = get_model(discriminator_model_type, discriminator_model_args)
        self.opt_d = None
        self.opt_d_type = discriminator_optim_type
        self.opt_d_args = discriminator_optim_args

        self.d_steps = 0.0  # Using float here avoids pl warning when logging steps
        self.g_steps = 0.0

        self.save_hyperparameters(ignore=['generator_display_function'])

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    @staticmethod
    def get_model_module_args_as_dict(args):
        args.generator_model_args = get_model_args_as_dict(args.generator_model_type, args)
        args.discriminator_model_args = get_model_args_as_dict(args.discriminator_model_type, args)
        args.generator_optim_args = get_generator_optimizer_args_as_dict(args)
        args.discriminator_optim_args = get_discriminator_optimizer_args_as_dict(args)
        return {
            'z_dim': args.z_dim,
            'grad_reg_lambda': args.grad_reg_lambda,
            'train_gen_every': args.train_gen_every,
            'gen_viz_every': args.gen_viz_every,
            'generator_display_function': args.generator_display_function,
            'generator_model_type': args.generator_model_type,
            'generator_model_args': args.generator_model_args,
            'generator_optim_type': args.generator_optim_type,
            'generator_optim_args': args.generator_optim_args,
            'discriminator_model_type': args.discriminator_model_type,
            'discriminator_model_args': args.discriminator_model_args,
            'discriminator_optim_type': args.discriminator_optim_type,
            'discriminator_optim_args': args.discriminator_optim_args,
            'log_images_to_tb': args.log_images_to_tb
        }

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        data, _ = batch
        z = torch.randn(batch[0].shape[0], self.z_dim).type_as(batch[0])

        # Train generator
        if self.train_generator:
            self.discriminator.eval()
            model_trained = 'generator'
            self.trainer.progress_bar_callback.main_progress_bar.set_description_str(
                f'Epoch {self.current_epoch} (Generator)')
            generated_data = self(z)
            # Generator wants to max the inf, so we flip the sign
            g_cost = -self.discriminator(generated_data).mean()
            g_opt.zero_grad()
            self.manual_backward(g_cost)
            g_opt.step()
            self.log('train/g_cost', -g_cost.item())   # Flip objective back for logging
            self.log('g_cost', -g_cost.item(), logger=False, prog_bar=True)
            self.g_steps += 1
            self.log('epoch/g_steps', self.g_steps)

            # Switch training to discriminator
            self.train_generator = 0
            self.train_discriminator = self.train_gen_every - 1

        # Train discriminator
        else:
            self.generator.eval()
            model_trained = 'discriminator'
            self.trainer.progress_bar_callback.main_progress_bar.set_description_str(
                f'Epoch {self.current_epoch} (Discriminator - {self.train_discriminator} step(s) left)')
            generated_data = self(z)
            d_real = self.discriminator(data).mean()
            d_fake = self.discriminator(generated_data).mean()
            d_cost = d_fake - d_real
            self.log('train/wasserstein_d', d_cost.item())
            if self.grad_reg_lambda:
                grad_reg = self.grad_reg_lambda*get_model_grad_wrt_interpolates(model=self.discriminator,
                                                                                real_data=data,
                                                                                fake_data=generated_data)
                self.log('train/wgan_gradient_regularizer', grad_reg.item())
                d_cost += grad_reg
                self.log('train/d_cost', d_cost.item())
            self.log('d_cost', d_cost.item(), logger=False, prog_bar=True)
            d_opt.zero_grad()
            self.manual_backward(d_cost)
            d_opt.step()
            self.d_steps += 1
            self.log('epoch/d_steps', self.d_steps)

            self.train_discriminator -= 1
            if self.train_discriminator == 0:  # Switch training to generator
                self.train_generator = 1
        return OrderedDict({'model_trained': model_trained})

    def validation_step(self, batch, batch_idx):
        z = torch.randn(batch[0].shape[0], self.z_dim).type_as(batch[0])
        generated_data = self(z)
        g_cost = self.discriminator(generated_data).mean()
        self.log('val/g_cost', g_cost.item())
        d_real = self.discriminator(batch[0]).mean()
        wasserstein_d = g_cost - d_real
        self.log('val/wasserstein_d', wasserstein_d.item())
        return {'batch': batch, 'z': z}

    def validation_epoch_end(self, validation_step_outputs):
        """
        For each display function the same signature is expected and return value is dictionary with figure file name as
        key and dict of figure title and matplotlib.pyplot.figure object
        """
        batch = validation_step_outputs[0]['batch']
        z = validation_step_outputs[0]['z']
        if not self.trainer.fast_dev_run and self.current_epoch % self.gen_viz_every == 0:
            figures = self.generator_display_function(
                epoch=self.current_epoch,
                val_batch=batch,
                z=z,
                generator=self.generator,
                choquet=None
            )
            os.makedirs(os.path.join(self.trainer.log_dir, 'figures'), exist_ok=True)
            for fig_name, fig in figures.items():
                fig['fig'].savefig(os.path.join(self.trainer.log_dir, 'figures', fig_name))
                if self.log_images_to_tb:
                    self.logger.experiment.add_figure(f'images/{fig["title"]}', fig['fig'], self.current_epoch)

    def configure_optimizers(self):
        opt_g = get_optimizer(self.opt_g_type, self.opt_g_args, self.generator.parameters())
        opt_d = get_optimizer(self.opt_d_type, self.opt_d_args, self.discriminator.parameters())
        return [opt_g, opt_d]


class ChoquetGanModule(pl.LightningModule):
    def __init__(self,
                 disc_or_dist,
                 how_to_combine_integral_terms,
                 split_regularization,
                 z_dim,
                 train_gen_every,
                 gen_viz_every,
                 grad_reg_lambda,
                 grad_reg_wrt,
                 projected_gradient_descent,
                 generator_display_function,
                 generator_model_type, generator_model_args, generator_optim_type, generator_optim_args,
                 critic_model_type, critic_model_args, critic_optim_type, critic_optim_args,
                 log_images_to_tb=False
                 ):
        super().__init__()

        self.automatic_optimization = False

        self.disc_or_dist = disc_or_dist
        self.how_to_combine_integral_terms = how_to_combine_integral_terms if disc_or_dist == 'dist' else 'N/A'
        # split regularization applies for CT distance only and should always be used if combining terms with 'min'
        if disc_or_dist == 'dist' and (split_regularization or how_to_combine_integral_terms != 'sum'):
            self.split_regularization = True
        else:
            self.split_regularization = False

        self.train_gen_every = train_gen_every
        self.train_generator = 0
        self.train_discriminator = train_gen_every - 1
        self.z_dim = z_dim

        self.generator_display_function = generator_display_function
        self.gen_viz_every = gen_viz_every
        self.log_images_to_tb = log_images_to_tb

        self.grad_reg_lambda = grad_reg_lambda
        self.grad_reg_wrt = grad_reg_wrt

        self.generator = get_model(generator_model_type, generator_model_args)
        self.opt_g = None
        self.opt_g_type = generator_optim_type
        self.opt_g_args = generator_optim_args

        if self.disc_or_dist == 'disc':
            critic = get_model(critic_model_type, critic_model_args)
            self.choquet = CTDiscrepancy(critic=critic)
        else:  # CT distance
            critic_0 = get_model(critic_model_type, critic_model_args)
            critic_1 = get_model(critic_model_type, critic_model_args)
            self.choquet = CTDistance(critics=[critic_0, critic_1],
                                      how_to_combine_integral_terms=how_to_combine_integral_terms,
                                      split_regularization=split_regularization)
        self.projected_gradient_descent = projected_gradient_descent
        self.opt_d = None
        self.opt_d_type = critic_optim_type
        self.opt_d_args = critic_optim_args

        self.d_steps = 0.0  # Using float here avoids pl warning when logging steps
        self.g_steps = 0.0

        self.save_hyperparameters(ignore=['generator_display_function'])

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ChoquetModule Args')
        # Discriminator setup args
        parser.add_argument('--disc_or_dist', type=str, choices=['disc', 'dist'], default='psiK',
                            help='Type of choquet objective: one-sided (\'disc\') or symmetric (\'dist\').')
        parser.add_argument('--how_to_combine_integral_terms', type=str, choices=['sum', 'min'],
                            default='sum',
                            help='Type of aggregation for CT distance.')
        parser.add_argument('--split_regularization', action='store_true', default=False,
                            help='Regularize each discriminator in CT distance separately.')
        # Discriminator optim args
        parser.add_argument('--projected_gradient_descent', action='store_true', default=False,
                            help='Use projected gradient descent to set hidden to hidden weights non-negative.')
        return parent_parser

    @staticmethod
    def get_model_module_args_as_dict(args):
        args.generator_model_args = get_model_args_as_dict(args.generator_model_type, args)
        args.discriminator_model_args = get_model_args_as_dict(args.discriminator_model_type, args)
        args.generator_optim_args = get_generator_optimizer_args_as_dict(args)
        args.discriminator_optim_args = get_discriminator_optimizer_args_as_dict(args)
        return {
            'disc_or_dist': args.disc_or_dist,
            'how_to_combine_integral_terms': args.how_to_combine_integral_terms,
            'split_regularization': args.split_regularization,
            'projected_gradient_descent': args.projected_gradient_descent,
            'z_dim': args.z_dim,
            'train_gen_every': args.train_gen_every,
            'gen_viz_every': args.gen_viz_every,
            'generator_display_function': args.generator_display_function,
            'generator_model_type': args.generator_model_type,
            'generator_model_args': args.generator_model_args,
            'generator_optim_type': args.generator_optim_type,
            'generator_optim_args': args.generator_optim_args,
            'grad_reg_lambda': args.grad_reg_lambda,
            'grad_reg_wrt': args.grad_reg_wrt,
            'critic_model_type': args.discriminator_model_type,
            'critic_model_args': args.discriminator_model_args,
            'critic_optim_type': args.discriminator_optim_type,
            'critic_optim_args': args.discriminator_optim_args,
            'log_images_to_tb': args.log_images_to_tb
        }

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        if self.disc_or_dist == 'disc':
            g_opt, c_opt = self.optimizers()
        else:
            g_opt, c0_opt, c1_opt = self.optimizers()

        data, _ = batch
        z = torch.randn(batch[0].shape[0], self.z_dim).type_as(batch[0])

        # Train generator
        if self.train_generator:
            self.choquet.eval()
            model_trained = 'generator'
            self.trainer.progress_bar_callback.main_progress_bar.set_description_str(
                f'Epoch {self.current_epoch} (Generator)')
            generated_data = self(z)
            objectives = self.choquet.objective(gen_data=generated_data, real_data=data)
            for k, v in objectives.items():
                self.log(f'train/{k}', v.item())

            # Generator wants to max the inf, so we flip the sign
            objective = -objectives['objective']
            g_opt.zero_grad()
            self.manual_backward(objective)
            g_opt.step()

            self.log('objective', -objective.item(), logger=False, prog_bar=True)  # Flip objective back for logging
            self.g_steps += 1
            self.log('epoch/g_steps', self.g_steps)

            # Switch training to discriminator
            self.train_generator = 0
            self.train_discriminator = self.train_gen_every - 1

        # Train discriminator
        else:
            self.generator.eval()
            model_trained = 'discriminator'
            self.trainer.progress_bar_callback.main_progress_bar.set_description_str(
                f'Epoch {self.current_epoch} (Discriminator - {self.train_discriminator} step(s) left)')
            generated_data = self(z)
            objectives = self.choquet.objective(gen_data=generated_data, real_data=data)
            for k, v in objectives.items():
                self.log(f'train/{k}', v.item())

            objective = objectives['objective']
            if self.grad_reg_lambda:
                gradient_regularizer = None
                if self.grad_reg_wrt == 'generator_parameters':
                    gradient_regularizer = self.choquet.grad_reg_wrt_gen_params(z=z, generator=self.generator,
                                                                                grad_reg_lambda=self.grad_reg_lambda)
                elif self.grad_reg_wrt == 'interpolates':
                    gradient_regularizer = self.choquet.grad_reg_wrt_interpolates(real_data=data,
                                                                                  fake_data=generated_data,
                                                                                  grad_reg_lambda=self.grad_reg_lambda)
                self.log('train/gradient_regularizer', gradient_regularizer.item())
                objective += gradient_regularizer
                self.log('train/objective_with_reg', objective.item())
            if self.disc_or_dist == 'disc':
                c_opt.zero_grad()
                self.manual_backward(objective)
                c_opt.step()
            else:
                if objectives['u0_integral'].requires_grad:
                    c0_opt.zero_grad()
                    self.manual_backward(objectives['u0_integral'],
                                         retain_graph=objectives['u1_integral'].requires_grad)  # Save graph for 'sum'
                    c0_opt.step()
                if objectives['u1_integral'].requires_grad:
                    c1_opt.zero_grad()
                    self.manual_backward(objectives['u1_integral'])
                    c1_opt.step()
            if self.projected_gradient_descent:
                self.choquet.project_critic_weights_to_positive()

            self.log('objective', objective.item(), logger=False, prog_bar=True)
            self.d_steps += 1
            self.log('epoch/d_steps', self.d_steps)

            self.train_discriminator -= 1
            if self.train_discriminator == 0:  # Switch training to generator
                self.train_generator = 1

        return OrderedDict({'objective': objective.item(), 'model_trained': model_trained})

    def validation_step(self, batch, batch_idx):
        z = torch.randn(batch[0].shape[0], self.z_dim).type_as(batch[0])
        generated_data = self(z)
        objectives = self.choquet.objective(gen_data=generated_data, real_data=batch[0])
        for k, v in objectives.items():
            self.log(f'val/{k}', v.item(), batch_size=batch[0].shape[0])
        return {'objective': objectives['objective'].item(), 'batch': batch, 'z': z}

    def validation_epoch_end(self, validation_step_outputs):
        """
        For each display function the same signature is expected and return value is dictionary with figure file name as
        key and dict of figure title and matplotlib.pyplot.figure object
        """
        # Test convexity
        rand_batch_idx = torch.randint(low=0, high=len(validation_step_outputs), size=(1,)).item()  # Grab random batch
        cvx = self.choquet.log_critic_convexity(validation_step_outputs[rand_batch_idx]['batch'][0], self.log)
        for k, v in cvx.items():
            self.log(f'{k}_cvx', v, logger=False, prog_bar=True)

        # Log choquet weight norms and non-zero count
        self.choquet.log_critic_weight_norms(self.log)
        self.choquet.log_critic_nonzero_weights(self.log)

        # Create plots
        batch = validation_step_outputs[0]['batch']
        z = validation_step_outputs[0]['z']
        if not self.trainer.fast_dev_run and self.current_epoch % self.gen_viz_every == 0:
            figures = self.generator_display_function(
                epoch=self.current_epoch,
                val_batch=batch,
                z=z,
                generator=self.generator,
                choquet=self.choquet
            )
            os.makedirs(os.path.join(self.trainer.log_dir, 'figures'), exist_ok=True)
            for fig_name, fig in figures.items():
                fig['fig'].savefig(os.path.join(self.trainer.log_dir, 'figures', fig_name))
                if self.log_images_to_tb:
                    self.logger.experiment.add_figure(f'images/{fig["title"]}', fig['fig'], self.current_epoch)

    def configure_optimizers(self):
        opt_g = get_optimizer(self.opt_g_type, self.opt_g_args, self.generator.parameters())
        if self.disc_or_dist == 'disc':
            opt_c = get_optimizer(self.opt_d_type, self.opt_d_args, self.choquet.parameters())
            return [opt_g, opt_c]
        else:
            opt_c0 = get_optimizer(self.opt_d_type, self.opt_d_args, self.choquet.critic_0.parameters())
            opt_c1 = get_optimizer(self.opt_d_type, self.opt_d_args, self.choquet.critic_1.parameters())
            return [opt_g, opt_c0, opt_c1]


class WGanChoquetDominateModule(pl.LightningModule):
    def __init__(self,
                 wgan_module,
                 pretrained_wgan_generator,
                 choquet_weight,
                 max_choquet_epochs,
                 real_dom_gen,
                 train_gen_every,
                 gen_viz_every,
                 choquet_reg_lambda,
                 choquet_reg_type,
                 projected_gradient_descent,
                 generator_display_function,
                 critic_model_type, critic_model_args, critic_optim_type, critic_optim_args,
                 log_images_to_tb=False
                 ):
        super().__init__()
        self.automatic_optimization = False

        self.wgan = wgan_module
        self.pretrained_wgan_generator = pretrained_wgan_generator

        self.train_gen_every = train_gen_every
        self.train_generator = 0
        self.train_discriminator = train_gen_every - 1

        self.generator_display_function = generator_display_function
        self.gen_viz_every = gen_viz_every
        self.log_images_to_tb = log_images_to_tb

        self.choquet_weight = choquet_weight
        self.max_choquet_epochs = max_choquet_epochs
        self.choquet_reg_lambda = choquet_reg_lambda
        self.choquet_reg_type = choquet_reg_type
        critic = get_model(critic_model_type, critic_model_args)
        self.choquet_base = VariationalDominanceCriterion(critic=critic, name='vdc_gen_dom_base')
        self.real_dom_gen = real_dom_gen
        if real_dom_gen:
            critic = get_model(critic_model_type, critic_model_args)
            self.choquet_real = VariationalDominanceCriterion(critic=critic, name='vdc_real_dom_gen')
        self.projected_gradient_descent = projected_gradient_descent
        self.opt_d, self.opt_choquet = None, None
        self.opt_d_type = critic_optim_type
        self.opt_d_args = critic_optim_args

        self.d_steps = 0.0  # Using float here avoids pl warning when logging steps
        self.g_steps = 0.0

        self.save_hyperparameters(ignore=['wgan_module', 'pretrained_wgan_generator',
                                          'generator_display_function'])

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('WGanChoquetFinetuneModule Args')
        parser.add_argument('--wgan_chkpt_file', type=str,
                            help='Absolute path to WGAN-GP checkpoint file.')
        parser.add_argument('--wgan_chkpt_args', type=str,
                            help='Absolute path to WGAN-GP checkpoint args.')
        parser.add_argument('--choquet_weight', type=float, default=1.0,
                            help='Weight for choquet objective term.')
        parser.add_argument('--real_dom_gen', action='store_true',
                            help='Add VDC term for real data to dominated generator.')
        parser.add_argument('--max_choquet_epochs', type=int, default=400,
                            help='Number of epochs to include Choquet term.')
        parser.add_argument('--choquet_reg_type', type=str, choices=['interpolates', 'u_squared'],
                            default='interpolates',
                            help='Type of regularization to use for Choquet models.')
        parser.add_argument('--choquet_reg_lambda', type=float, default=10.0,
                            help='Hyperparameter to control Choquet regularization.')

        # Discriminator optim args
        parser.add_argument('--projected_gradient_descent', action='store_true', default=False,
                            help='Use projected gradient descent for Choquet to set hidden weights non-negative.')
        return parent_parser

    @staticmethod
    def add_wgan_override_args(parent_parser):
        parser = parent_parser.add_argument_group('WGan-GP Override Args')
        parser.add_argument('--wgan_args_override', action='store_true',
                            help='Override args from pre-trained WGan module.')
        # Generator_args
        parser.add_argument('--wgan_z_dim', type=int,
                            help='WGan Generator latent dimension.')
        parser.add_argument('--wgan_g_hidden_dim', type=int,
                            help='WGan Generator hidden dimension.')
        parser.add_argument('--wgan_g_n_layers', type=int,
                            help='WGan Generator number of hidden layers.')
        parser.add_argument('--wgan_generator_model_type', type=str,
                            help='WGan Generator architecture.')
        parser.add_argument('--wgan_generator_optim_type', type=str,
                            help='WGan Generator optimizer.')
        parser.add_argument('--wgan_g_lr', type=float,
                            help='WGan Generator learning rate.')
        parser.add_argument('--wgan_g_weight_decay', type=float,
                            help='WGan Generator weight decay.')
        parser.add_argument('--wgan_g_betas', nargs=2, type=float,
                            help='WGan Generator betas (for Adam optim).')
        parser.add_argument('--wgan_g_eps', type=float,
                            help='WGan Generator eps (for Adam optim).')

        # Model args (shared by generator and discriminator)
        parser.add_argument('--wgan_activation', type=str,
                            choices=['max_out', 'relu'],
                            help='WGan Discriminator activation function.')
        parser.add_argument('--wgan_max_out', type=int,
                            help='WGan Maxout kernel size.')
        parser.add_argument('--wgan_dropout', action='store_true',
                            help='WGan Use dropout in discriminator.')

        # Discriminator model args
        parser.add_argument('--wgan_discriminator_model_type', type=str,
                            help='WGan Discriminator architecture.')
        parser.add_argument('--wgan_d_hidden_dim', type=int,
                            help='WGan Discriminator hidden dimension.')
        parser.add_argument('--wgan_d_n_layers', type=int,
                            help='WGan Discriminator number of hidden layers.')
        parser.add_argument('--wgan_grad_reg_lambda', type=float,
                            help='WGan Hyperparameter to control gradient regularization.')

        # Discriminator optim args
        parser.add_argument('--wgan_discriminator_optim_type', type=str,
                            help='WGan Discriminator optimizer.')
        parser.add_argument('--wgan_d_lr', type=float,
                            help='WGan Discriminator learning rate.')
        parser.add_argument('--wgan_d_weight_decay', type=float,
                            help='WGan Discriminator weight decay.')
        parser.add_argument('--wgan_d_betas', nargs=2, type=float,
                            help='WGan Discriminator betas (for Adam optim).')
        parser.add_argument('--wgan_d_eps', type=float,
                            help='WGan Discriminator eps (for Adam optim).')
        return parent_parser

    @staticmethod
    def override_wgan_args(pretrained_args_dict, args):
        args_dict = vars(args)
        if args_dict['wgan_args_override']:
            for k, v in args_dict.items():
                if 'wgan' in k and v:
                    pretrained_args_dict[k.split('wgan_')[1]] = v
        return pretrained_args_dict

    @staticmethod
    def get_model_module_args_as_dict(args):
        args.discriminator_model_args = get_model_args_as_dict(args.discriminator_model_type, args)
        args.discriminator_optim_args = get_discriminator_optimizer_args_as_dict(args)
        return {
            'wgan_module': args.wgan_module,
            'pretrained_wgan_generator': args.pretrained_wgan_generator,
            'choquet_weight': args.choquet_weight,
            'real_dom_gen': args.real_dom_gen,
            'max_choquet_epochs': args.max_choquet_epochs,
            'projected_gradient_descent': args.projected_gradient_descent,
            'choquet_reg_type': args.choquet_reg_type,
            'choquet_reg_lambda': args.choquet_reg_lambda,
            'train_gen_every': args.train_gen_every,
            'gen_viz_every': args.gen_viz_every,
            'generator_display_function': args.generator_display_function,
            'critic_model_type': args.discriminator_model_type,
            'critic_model_args': args.discriminator_model_args,
            'critic_optim_type': args.discriminator_optim_type,
            'critic_optim_args': args.discriminator_optim_args,
            'log_images_to_tb': args.domain == 'images' or args.log_images_to_tb
        }

    def forward(self, z):
        return self.wgan.generator(z)

    def training_step(self, batch, batch_idx):
        if self.real_dom_gen:
            g_opt, d_opt, choquet_base_opt, choquet_real_opt = self.optimizers()
        else:
            g_opt, d_opt, choquet_base_opt = self.optimizers()
            choquet_real_opt = None

        data, _ = batch
        z = torch.randn(batch[0].shape[0], self.wgan.z_dim).type_as(batch[0])
        self.pretrained_wgan_generator.eval()
        baseline_generated_data = self.pretrained_wgan_generator(z)

        if self.current_epoch == self.max_choquet_epochs:
            self.logger.experiment.add_text('Choquet Turned Off',
                                            f'Choquet objective turned off at epoch {self.current_epoch}.')

        # Train generator
        if self.train_generator:
            self.wgan.discriminator.eval()
            self.choquet_base.eval()
            if self.real_dom_gen:
                self.choquet_real.eval()
            model_trained = 'generator'
            self.trainer.progress_bar_callback.main_progress_bar.set_description_str(
                f'Epoch {self.current_epoch} (Generator)')
            generated_data = self(z)
            # Objectives flipped so generator maximizes
            g_cost = -self.wgan.discriminator(generated_data).mean()
            if self.current_epoch < self.max_choquet_epochs:
                choquet_base_objective = -self.choquet_base.objective(dominating_data=generated_data,
                                                                      dominated_data=baseline_generated_data)
                self.log('train/choquet_base_objective', -self.choquet_weight*choquet_base_objective.item())
                total_objective = g_cost + self.choquet_weight * choquet_base_objective
                if self.real_dom_gen:
                    choquet_real_objective = -self.choquet_real.objective(dominating_data=data,
                                                                          dominated_data=generated_data)
                    self.log('train/choquet_real_objective', -self.choquet_weight * choquet_real_objective.item())
                    total_objective += self.choquet_weight * choquet_real_objective
                self.log('train/total_objective', -total_objective.item())
            else:
                total_objective = g_cost

            g_opt.zero_grad()
            self.manual_backward(total_objective)
            g_opt.step()

            # Flip objectives back for logging
            self.log('train/g_cost', -g_cost.item())
            total_objective = total_objective.item()
            self.log('objective', -total_objective, logger=False, prog_bar=True)
            self.g_steps += 1
            self.log('epoch/g_steps', self.g_steps)

            # Switch training to discriminator
            self.train_generator = 0
            self.train_discriminator = self.train_gen_every - 1

        # Train discriminator
        else:
            # Train WGAN-GP Discriminator
            self.wgan.generator.eval()
            model_trained = 'discriminator'
            self.trainer.progress_bar_callback.main_progress_bar.set_description_str(
                f'Epoch {self.current_epoch} (Discriminator - {self.train_discriminator} step(s) left)')
            generated_data = self(z)
            d_real = self.wgan.discriminator(data).mean()
            d_fake = self.wgan.discriminator(generated_data).mean()
            d_cost = d_fake - d_real
            if self.wgan.grad_reg_lambda:
                grad_reg = self.wgan.grad_reg_lambda * get_model_grad_wrt_interpolates(model=self.wgan.discriminator,
                                                                                       real_data=data,
                                                                                       fake_data=generated_data)
                self.log('train/wgan_gradient_regularizer', grad_reg.item())
                d_cost += grad_reg
            d_opt.zero_grad()
            self.manual_backward(d_cost, retain_graph=(self.current_epoch < self.max_choquet_epochs))
            d_opt.step()
            self.log('train/wasserstein_d', d_cost.item())
            self.log('train/d_cost', d_cost.item())
            total_objective = d_cost.item()

            # Train choquet functions
            if self.current_epoch < self.max_choquet_epochs:
                # Gen dominating baseline
                choquet_base_objective = self.choquet_base.objective(dominating_data=generated_data,
                                                                     dominated_data=baseline_generated_data)
                total_objective += self.choquet_weight * choquet_base_objective.item()
                self.log('train/choquet_base_objective', self.choquet_weight * choquet_base_objective.item())
                if self.choquet_reg_lambda:
                    if self.choquet_reg_type == 'interpolates':
                        choquet_base_reg = self.choquet_base.grad_reg_wrt_interpolates(
                            data0=baseline_generated_data,
                            data1=generated_data,
                            grad_reg_lambda=self.choquet_reg_lambda
                        )
                    elif self.choquet_reg_type == 'u_squared':
                        choquet_base_reg = self.choquet_base.reg_u_squared(
                            data0=baseline_generated_data,
                            data1=generated_data,
                            reg_lambda=self.choquet_reg_lambda
                        )
                    else:
                        raise ValueError('Unsupported regularization used for this training')
                    self.log('train/choquet_base_regularizer', choquet_base_reg.item())
                    total_objective += choquet_base_reg.item()
                    choquet_base_objective += choquet_base_reg
                choquet_base_opt.zero_grad()
                self.manual_backward(choquet_base_objective, retain_graph=self.real_dom_gen)
                choquet_base_opt.step()
                if self.projected_gradient_descent:
                    self.choquet_base.project_critic_weights_to_positive()

                if self.real_dom_gen:
                    # Real dominating gen
                    choquet_real_objective = self.choquet_real.objective(dominating_data=data,
                                                                         dominated_data=generated_data)
                    total_objective += self.choquet_weight * choquet_real_objective.item()
                    self.log('train/choquet_real_objective', self.choquet_weight * choquet_real_objective.item())
                    if self.choquet_reg_lambda:
                        if self.choquet_reg_type == 'interpolates':
                            choquet_real_reg = self.choquet_real.grad_reg_wrt_interpolates(
                                data0=generated_data,
                                data1=data,
                                grad_reg_lambda=self.choquet_reg_lambda
                            )
                        elif self.choquet_reg_type == 'u_squared':
                            choquet_real_reg = self.choquet_real.reg_u_squared(
                                data0=generated_data,
                                data1=data,
                                reg_lambda=self.choquet_reg_lambda
                            )
                        else:
                            raise ValueError('Unsupported regularization used for this training')
                        self.log('train/choquet_real_regularizer', choquet_real_reg.item())
                        total_objective += choquet_real_reg.item()
                        choquet_real_objective += choquet_real_reg

                    choquet_real_opt.zero_grad()
                    self.manual_backward(choquet_real_objective)
                    choquet_real_opt.step()
                    if self.projected_gradient_descent:
                        self.choquet_real.project_critic_weights_to_positive()
            self.log('train/total_objective', total_objective)
            self.log('objective', total_objective, logger=False, prog_bar=True)
            self.d_steps += 1
            self.log('epoch/d_steps', self.d_steps)

            self.train_discriminator -= 1
            if self.train_discriminator == 0:  # Switch training to generator
                self.train_generator = 1

        return OrderedDict({'objective': total_objective, 'model_trained': model_trained})

    def validation_step(self, batch, batch_idx):
        z = torch.randn(batch[0].shape[0], self.wgan.z_dim).type_as(batch[0])
        generated_data = self(z)
        baseline_generated_data = self.pretrained_wgan_generator(z)

        g_cost = self.wgan.discriminator(generated_data).mean()
        self.log('val/g_cost', g_cost.item())

        d_real = self.wgan.discriminator(batch[0]).mean()
        wasserstein_d = g_cost - d_real
        self.log('val/wasserstein_d', wasserstein_d.item())

        if self.current_epoch < self.max_choquet_epochs:
            choquet_base_objective = self.choquet_base.objective(dominating_data=generated_data,
                                                                 dominated_data=baseline_generated_data)

            self.log('val/choquet_base_objective', self.choquet_weight*choquet_base_objective.item())
            total_objective = wasserstein_d + self.choquet_weight * choquet_base_objective
            if self.real_dom_gen:
                choquet_real_objective = self.choquet_real.objective(dominating_data=batch[0],
                                                                     dominated_data=generated_data)
                self.log('val/choquet_real_objective', self.choquet_weight*choquet_real_objective.item())
                total_objective += self.choquet_weight * choquet_real_objective

            self.log('val/total_objective', total_objective.item())
        return {'batch': batch, 'z': z, 'gen_data': generated_data, 'base_gen_data': baseline_generated_data}

    def validation_epoch_end(self, validation_step_outputs):
        """
        For each display function the same signature is expected and return value is dictionary with figure file name as
        key and dict of figure title and matplotlib.pyplot.figure object
        """
        if self.current_epoch < self.max_choquet_epochs:
            rand_batch_idx = torch.randint(low=0, high=len(validation_step_outputs), size=(1,)).item()

            # Baseline dominate VDC
            cvx_real = self.choquet_base.log_critic_convexity(
                validation_step_outputs[rand_batch_idx]['batch'][0],
                self.log)
            for k, v in cvx_real.items():
                self.log(f'{k}_cvx', v, logger=False, prog_bar=True)
            # Log choquet weight norms and non-zero count
            self.choquet_base.log_critic_weight_norms(self.log)
            self.choquet_base.log_critic_nonzero_weights(self.log)

            if self.real_dom_gen:
                # Real data dominate VDC
                cvx_real = self.choquet_real.log_critic_convexity(
                    validation_step_outputs[rand_batch_idx]['batch'][0],
                    self.log)
                for k, v in cvx_real.items():
                    self.log(f'{k}_cvx', v, logger=False, prog_bar=True)
                # Log choquet weight norms and non-zero count
                self.choquet_real.log_critic_weight_norms(self.log)
                self.choquet_real.log_critic_nonzero_weights(self.log)

        # Create plots
        batch = validation_step_outputs[0]['batch']
        z = validation_step_outputs[0]['z']
        if not self.trainer.fast_dev_run and self.current_epoch % self.gen_viz_every == 0:
            figures = self.generator_display_function(
                epoch=self.current_epoch,
                val_batch=batch,
                z=z,
                generator=self.wgan.generator,
                choquet=None
            )
            os.makedirs(os.path.join(self.trainer.log_dir, 'figures'), exist_ok=True)
            for fig_name, fig in figures.items():
                fig['fig'].savefig(os.path.join(self.trainer.log_dir, 'figures', fig_name))
                if self.log_images_to_tb:
                    self.logger.experiment.add_figure(f'images/{fig["title"]}', fig['fig'], self.current_epoch)

    def configure_optimizers(self):
        wgan_opt_g = get_optimizer(self.wgan.opt_g_type, self.wgan.opt_g_args, self.wgan.generator.parameters())
        wgan_opt_d = get_optimizer(self.wgan.opt_d_type, self.wgan.opt_d_args, self.wgan.discriminator.parameters())
        opt_choquet_base = get_optimizer(self.opt_d_type, self.opt_d_args, self.choquet_base.parameters())
        if self.real_dom_gen:
            opt_choquet_real = get_optimizer(self.opt_d_type, self.opt_d_args, self.choquet_real.parameters())
            return [wgan_opt_g, wgan_opt_d, opt_choquet_base, opt_choquet_real]
        else:
            return [wgan_opt_g, wgan_opt_d, opt_choquet_base]
