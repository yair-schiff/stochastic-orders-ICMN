import argparse
import json
import os

import torch
from pytorch_lightning.callbacks import Callback
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm


class SaveArgsCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.args_ignore = ['wgan_module', 'pretrained_wgan_generator', 'generator_display_function',
                            'generator_model', 'generator_optim',
                            'discriminator_model', 'discriminator_optim']

    def on_sanity_check_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(trainer.log_dir, exist_ok=True)
            args_dict = {k: v for k, v in vars(self.args).items() if k not in self.args_ignore}
            # Remove 'params' from generator and discriminator optim args
            for k, v in args_dict.items():
                if isinstance(v, dict) and 'params' in v.keys():
                    args_dict[k] = {k_d: v_d for k_d, v_d in v.items() if k_d != 'params'}
            with open(os.path.join(trainer.log_dir, 'args.json'), 'w') as f:
                json.dump(args_dict, f, indent=4)
            trainer.logger.experiment.add_text('Args', json.dumps(args_dict, indent=4))


class SaveArchitectureCallback(Callback):
    def on_sanity_check_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(trainer.log_dir, exist_ok=True)
            with open(os.path.join(trainer.log_dir, 'architectures.txt'), 'w') as f:
                f.write(pl_module.__str__())
            trainer.logger.experiment.add_text('Architectures', pl_module.__str__())


class FidCallback(Callback):
    def __init__(self, args, num_fid_iters=5):
        self.batch_size = args.batch_size
        self.z_dim = args.z_dim
        self.num_fid_iters = 1 if args.debug else num_fid_iters
        self.fid = FrechetInceptionDistance(feature=2048)
        self.debug = args.debug

    def compute_fid(self, trainer, pl_module):
        # Initialize real features if not already done so
        if len(self.fid.real_features) == 0:
            self.fid.to(pl_module.device)
            if self.debug:
                val_dataloader_tqdm = tqdm([next(iter(trainer.datamodule.val_dataloader()))],
                                           desc='Extract Real Data Inception Features')
            else:
                val_dataloader_tqdm = tqdm(trainer.datamodule.val_dataloader(),
                                           desc='Extract Real Data Inception Features')
            for batch in val_dataloader_tqdm:
                image = (batch[0] + 1) * 128
                self.fid.update(image.type(torch.uint8).to(pl_module.device), real=True)

        # Compute FID a number of times and get mean/var:
        fid_scores = torch.zeros(self.num_fid_iters)
        fid_pbar = tqdm(range(self.num_fid_iters), desc=f'FID Scoring Epoch {pl_module.current_epoch}')
        for i in fid_pbar:
            for rf in self.fid.real_features:
                z = torch.randn(rf.shape[0], self.z_dim).type_as(rf)
                try:
                    gen_z = (pl_module.generator(z) + 1) * 128
                except AttributeError:
                    gen_z = (pl_module.wgan.generator(z) + 1) * 128
                self.fid.update(gen_z.type(torch.uint8), real=False)
            fid_scores[i] = self.fid.compute()
            fid_pbar.set_postfix({
                'FID mean': fid_scores[:i + 1].mean().item(),
                'FID var': torch.var(fid_scores[:i + 1]).item()
            })
            self.fid.fake_features = []  # reset fake features
        return {'FID/mean': fid_scores.mean().item(), 'FID/variance': torch.var(fid_scores).item()}

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking and pl_module.current_epoch % pl_module.train_gen_every == 0:
            fid_scores = self.compute_fid(trainer, pl_module)
            for k, v in fid_scores.items():
                pl_module.log(k, v)


def get_parser():
    parser = argparse.ArgumentParser(description='Train Model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Train args
    train_parser = parser.add_argument_group('Train Args')
    train_parser.add_argument('--seed', type=int, default=0,
                              help='Random seed initialization.')
    train_parser.add_argument('--epochs', type=int, default=10,
                              help='Number of epochs.')
    train_parser.add_argument('--checkpoint_save_path', type=str, default='saved_models',
                              help='Absolute path for saving checkpoints.')
    train_parser.add_argument('--checkpoint_every_n', type=int, default=1,
                              help='Checkpoint every n epochs.')
    train_parser.add_argument('--restart_from_last', action='store_true',
                              help='Restart from last checkpoint.')
    train_parser.add_argument('--version_number', type=int, default=-1,
                              help='Optional parameter to manually set version_# in log dir to avoid file access ' +
                                   'race conditions. To let lightning figure it out on its own, pass -1.')
    train_parser.add_argument('--device', type=str, default='cpu',
                              choices=['cpu', 'gpu'],
                              help='Device: cpu or gpu.')
    train_parser.add_argument('--num_devices', type=int, default=1,
                              help='Number of cpus/gpus to use.')
    train_parser.add_argument('--debug', action='store_true',
                              help='Run experiment in debug mode, i.e. using pytorch lightning fast-dev-run.')
    train_parser.add_argument('--log_images_to_tb', action='store_true',
                              help='Log plot outputs to tensorboard.')
    # Data args
    data_parser = parser.add_argument_group('Data Args')
    data_parser.add_argument('--batch_size', type=int, default=128,
                             help='Batch size.')
    data_parser.add_argument('--validation_batch_multiplier', type=int, default=1,
                             help='Multiple of batch size to use in validation steps.')
    data_parser.add_argument('--num_workers', type=int, default=0,
                             help='Number of workers for DataLoader.')
    data_parser.add_argument('--domain', type=str, choices=['distributions', 'images'],
                             help='Problem domain.')
    # Distribution args
    data_parser.add_argument('--distribution_type', type=str, default=None,
                             choices=['circle_of_gaussians', 'swiss_roll', 'image_point_cloud'],
                             help='Type of distribution to use.')
    # args for circle of gaussians
    data_parser.add_argument('--n_gaussians', type=int, default=0,
                             help='Number of gaussians (for \'circle_of_gaussians\' distribution).')
    data_parser.add_argument('--std', type=float, default=0.0,
                             help='Standard deviation of gaussians (for \'circle_of_gaussians\' distribution).')
    data_parser.add_argument('--radius', type=int, default=0,
                             help='Radius of gaussians (for \'circle_of_gaussians\' distribution).')
    # args for swiss roll
    data_parser.add_argument('--noise', type=float, default=0.0,
                             help='Amount of noise to add (for \'swiss_roll\' distribution).')
    # args for image point clouds
    data_parser.add_argument('--image_name', type=str, default=None,
                             choices=['github_icon'],
                             help='Name of image to be turned into point cloud.')
    data_parser.add_argument('--image_path', type=str, default=None,
                             help='Absolute path to image file.')

    # Dataset args
    data_parser.add_argument('--dataset_name', type=str, default=None,
                             choices=['mnist', 'fashion_mnist', 'cifar10', 'svhn'],
                             help='Image dataset to use.')
    data_parser.add_argument('--dataset_dir', type=str, default=None,
                             help='Absolute path to where image dataset is stored.')
    data_parser.add_argument('--train_split', type=float, default=None,
                             help='Train split.')

    # Model args (shared by generator and discriminator)
    shared_gan_args = parser.add_argument_group('Generator-Discriminator Shared Args')
    shared_gan_args.add_argument('--activation', type=str, default='relu',
                                 choices=['max_out', 'relu'],
                                 help='Discriminator activation function.')
    shared_gan_args.add_argument('--max_out', type=int, default=4,
                                 help='Maxout kernel size.')
    shared_gan_args.add_argument('--dropout', action='store_true',
                                 help='Use dropout in discriminator.')

    # Generator_args
    generator_parser = parser.add_argument_group('Generator Args')
    generator_parser.add_argument('--z_dim', type=int, default=256,
                                  help='Generator latent dimension.')
    generator_parser.add_argument('--train_gen_every', type=int, default=2,
                                  help='Train generator every x epochs. \'2\' will alternate training every epoch.')
    generator_parser.add_argument('--gen_viz_every', type=int, default=1000,
                                  help='Visualize generator output every x epochs.')
    generator_parser.add_argument('--g_hidden_dim', type=int, default=128,
                                  help='Generator hidden dimension.')
    generator_parser.add_argument('--g_n_layers', type=int, default=2,
                                  help='Generator number of hidden layers.')
    generator_parser.add_argument('--generator_model_type', type=str,
                                  help='Generator architecture.')
    generator_parser.add_argument('--generator_optim_type', type=str, default='adam',
                                  help='Generator optimizer.')
    generator_parser.add_argument('--g_lr', type=float, default=1e-4,
                                  help='Generator learning rate.')
    generator_parser.add_argument('--g_weight_decay', type=float, default=0,
                                  help='Generator weight decay.')
    generator_parser.add_argument('--g_betas', nargs=2, type=float, default=(0.5, 0.9),
                                  help='Generator betas (for Adam optim).')
    generator_parser.add_argument('--g_eps', type=float, default=1e-8,
                                  help='Generator eps (for Adam optim).')

    # Discriminator model args
    discriminator_parser = parser.add_argument_group('Discriminator Args')
    discriminator_parser.add_argument('--discriminator_model_type', type=str,
                                      help='Discriminator architecture.')
    discriminator_parser.add_argument('--d_hidden_dim', type=int, default=512,
                                      help='Discriminator hidden dimension.')
    discriminator_parser.add_argument('--d_n_layers', type=int, default=4,
                                      help='Discriminator number of hidden layers.')
    discriminator_parser.add_argument('--grad_reg_lambda', type=float, default=10.0,
                                      help='Hyperparameter to control gradient regularization.')
    discriminator_parser.add_argument('--grad_reg_wrt', type=str,
                                      choices=['generator_parameters', 'interpolates'],
                                      default='generator_parameters',
                                      help='What to calculate the gradient regularization with respect to.')
    # Discriminator optim args
    discriminator_parser.add_argument('--discriminator_optim_type', type=str, default='adam',
                                      help='Discriminator optimizer.')
    discriminator_parser.add_argument('--d_lr', type=float, default=1e-4,
                                      help='Discriminator learning rate.')
    discriminator_parser.add_argument('--d_weight_decay', type=float, default=0,
                                      help='Discriminator weight decay.')
    discriminator_parser.add_argument('--d_betas', nargs=2, type=float, default=(0.5, 0.9),
                                      help='Discriminator betas (for Adam optim).')
    discriminator_parser.add_argument('--d_eps', type=float, default=1e-8,
                                      help='Discriminator eps (for Adam optim).')
    return parser


def build_distribution_args(args):
    if args.distribution_type == 'circle_of_gaussians':
        return {
            'n_gaussians': args.n_gaussians,
            'std': args.std,
            'radius': args.radius
        }
    elif args.distribution_type == 'swiss_roll':
        return {
            'noise': args.noise,
        }
    elif args.distribution_type == 'image_point_cloud':
        return {
            'image_name': args.image_name,
            'image_path': args.image_path,
        }


def get_path_to_last_checkpoint(checkpoint_dir):
    if os.path.exists(os.path.join(checkpoint_dir, 'last.ckpt')):
        return os.path.join(checkpoint_dir, 'last.ckpt')
    last_epoch = 0
    for f in os.listdir(checkpoint_dir):
        epoch = int(f.split('=')[1].split('.')[0])
        last_epoch = max(last_epoch, epoch)
    if last_epoch:
        return os.path.join(checkpoint_dir, f'epoch={last_epoch}.ckpt')
    else:
        return None
