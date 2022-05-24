import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from scripts.plot_utils import get_plot_function
from scripts.train_utils import (
    build_distribution_args, get_parser, get_path_to_last_checkpoint,
    FidCallback, SaveArgsCallback, SaveArchitectureCallback
)
from src import ChoquetGanModule, WGanModule, DistributionDataModule, ImageDataModule


def main(mode, argv):
    parser = get_parser()
    if mode == 'choquet':
        model_module = ChoquetGanModule
    elif mode == 'wgan':
        model_module = WGanModule
    else:
        raise ValueError('Must use one of \'choquet\', \'wgan\' modes for this script.')
    parser = model_module.add_model_specific_args(parser)
    args = parser.parse_args(argv)
    args.generator_display_function = get_plot_function(mode, args)
    args.distribution_params = build_distribution_args(args) if args.domain == 'distributions' else None
    pl.seed_everything(seed=args.seed)  # Set seed for reproducibility
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # Setup Logger
    logger = TensorBoardLogger(
        save_dir=args.checkpoint_save_path,
        version=args.version_number if args.version_number >= 0 else None,
        name=os.path.join(args.checkpoint_save_path, 'lightning_logs')
    )

    # Initialize model and data modules
    mm = model_module(**model_module.get_model_module_args_as_dict(args))
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
    trainer.fit(model=mm, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main(mode=sys.argv[1], argv=sys.argv[2:])
