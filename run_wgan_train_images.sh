#!/bin/bash

export PYTHONPATH="${PWD}:${PWD}/src"

domain="images"
dataset_name="cifar10"
dataset_dir="${PWD}/saved_data/${dataset_name}"

checkpoint_save_path="${PWD}/saved_models/${domain}/${dataset_name}/wgan/"
mkdir -p "${checkpoint_save_path}"
cd ./scripts || exit

# Train args
device="gpu"
version_number=0
seed=0
epochs=400
checkpoint_every_n=10
# Data args
batch_size=64
validation_batch_multiplier=1
train_split=0.95
# Generator_args
z_dim=128
g_hidden_dim=128
train_gen_every=6
gen_viz_every=6
generator_model_type="cifar10_residual_generator"
generator_optim_type="adam"
g_lr=1e-4
# Model args (shared by generator and discriminator)
activation="relu"
# Discriminator model args
discriminator_model_type='cifar10_discriminator'
d_hidden_dim=128
grad_reg_lambda=10
# Discriminator optim args
discriminator_optim_type="adam"
d_lr=1e-4

python gan_train.py 'wgan' \
  --seed ${seed} \
  --epochs ${epochs} \
  --checkpoint_save_path "${checkpoint_save_path}" \
  --checkpoint_every_n ${checkpoint_every_n} \
  --restart_from_last \
  --version_number ${version_number} \
  --device "${device}" \
  --num_devices 1 \
  --log_images_to_tb \
  --batch_size ${batch_size} \
  --validation_batch_multiplier ${validation_batch_multiplier} \
  --domain "${domain}" \
  --dataset_name "${dataset_name}" \
  --dataset_dir "${dataset_dir}" \
  --train_split ${train_split} \
  --z_dim ${z_dim} \
  --g_hidden_dim ${g_hidden_dim} \
  --train_gen_every ${train_gen_every} \
  --gen_viz_every ${gen_viz_every} \
  --generator_model_type "${generator_model_type}" \
  --generator_optim_type "${generator_optim_type}" \
  --g_lr ${g_lr} \
  --activation "${activation}" \
  --discriminator_model_type "${discriminator_model_type}" \
  --d_hidden_dim ${d_hidden_dim} \
  --grad_reg_lambda ${grad_reg_lambda} \
  --discriminator_optim_type "${discriminator_optim_type}" \
  --d_lr ${d_lr}
