import json
import os
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from scripts.plot_utils import get_plot_function
from scripts.train_utils import (
    build_distribution_args, get_parser, get_path_to_last_checkpoint,
    FidCallback, SaveArgsCallback, SaveArchitectureCallback
)
from src import WGanModule, WGanChoquetDominateModule, ImageDataModule, DistributionDataModule


def main():
    parser = get_parser()
    parser = WGanChoquetDominateModule.add_model_specific_args(parser)
    parser = WGanChoquetDominateModule.add_wgan_override_args(parser)
    args = parser.parse_args()
    pl.seed_everything(seed=args.seed)  # Set seed for reproducibility
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # Load the WGAN GP generator:
    wgan_module = WGanModule.load_from_checkpoint(args.wgan_chkpt_file, generator_display_function=None)
    # Remove gradient from pre-trained model
    wgan_pretrained_generator = wgan_module.generator.requires_grad_(False).eval()
    with open(args.wgan_chkpt_args) as wgan_args_file:
        wgan_chkpt_args_dict = json.load(wgan_args_file)
    # Override any args that were passed for WGAN
    wgan_chkpt_args_dict = WGanChoquetDominateModule.override_wgan_args(pretrained_args_dict=wgan_chkpt_args_dict,
                                                                        args=args)

    wgan_chkpt_args = SimpleNamespace(**wgan_chkpt_args_dict)
    wgan_chkpt_args.generator_display_function = None
    wgan_args = WGanModule.get_model_module_args_as_dict(wgan_chkpt_args)
    wm = WGanModule(**wgan_args)

    # Setup Logger
    logger = TensorBoardLogger(
        save_dir=args.checkpoint_save_path,
        version=args.version_number if args.version_number >= 0 else None,
        name=os.path.join(args.checkpoint_save_path, 'lightning_logs')
    )

    # Initialize model and data modules
    args.generator_display_function = get_plot_function('choquet', args)
    args.distribution_params = build_distribution_args(args) if args.domain == 'distributions' else None
    args.pretrained_wgan_generator = wgan_pretrained_generator
    args.wgan_module = wm
    dom_args = WGanChoquetDominateModule.get_model_module_args_as_dict(args)
    dom = WGanChoquetDominateModule(**dom_args)
    if args.domain == 'distributions':
        dm = DistributionDataModule(**DistributionDataModule.get_data_module_args_as_dict(args))
    else:
        dm = ImageDataModule(**ImageDataModule.get_data_module_args_as_dict(args))

    # Initialize trainer
    os.makedirs(os.path.join(logger.log_dir, 'checkpoints'), exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, 'checkpoints'),
        filename="{epoch:d}",
        save_last=(args.domain == 'images'),
        every_n_epochs=args.checkpoint_every_n,
        save_top_k=-1
    )
    callbacks = [checkpoint_callback, SaveArgsCallback(args), SaveArchitectureCallback()]
    args.z_dim = wgan_chkpt_args.z_dim  # Needed to ensure FID does not fail
    fid_callback = None
    if args.domain == 'images' and 'mnist' not in args.dataset_name:
        fid_callback = FidCallback(args, num_fid_iters=5)
        callbacks.append(fid_callback)

    if args.device == 'cpu':
        trainer = pl.Trainer(
            benchmark=False,
            deterministic=True,
            max_epochs=args.epochs,
            default_root_dir=args.checkpoint_save_path,
            callbacks=callbacks,
            logger=logger,
            fast_dev_run=args.debug,
        )
    else:
        trainer = pl.Trainer(
            benchmark=False,
            deterministic=True,
            gpus=args.num_devices,
            max_epochs=args.epochs,
            default_root_dir=args.checkpoint_save_path,
            callbacks=callbacks,
            logger=logger,
            fast_dev_run=args.debug,
        )
    if args.restart_from_last and os.path.exists(os.path.join(logger.log_dir, 'checkpoints')):
        ckpt_path = get_path_to_last_checkpoint(os.path.join(logger.log_dir, 'checkpoints'))
    else:
        ckpt_path = None
    trainer.fit(model=dom, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
