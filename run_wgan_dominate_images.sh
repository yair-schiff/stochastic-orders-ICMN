#!/bin/bash

export PYTHONPATH="${PWD}:${PWD}/src"

# Dataset args
domain="images"
dataset_name="cifar10"
dataset_dir="${PWD}/saved_data/${dataset_name}"

# Finetune args
wgan_version=0
wgan_chkpt_file_name="last.ckpt"
wgan_path="${PWD}/saved_models/${domain}/${dataset_name}/wgan/lightning_logs/version_${wgan_version}"
wgan_chkpt_file="${wgan_path}/checkpoints/${wgan_chkpt_file_name}"
wgan_chkpt_args="${wgan_path}/args.json"


if [ -z "$1" ]; then
  v=0
else
  v=${1}
fi

# Experiment args
version_number=${v}
seed=$((v + 10))
choquet_weight=10
d_weight_decay=0
max_choquet_epochs=400
# Discriminator setup args
choquet_reg_lambda=10
choquet_reg_type="u_squared"
d_lr=1e-5

checkpoint_save_path="${PWD}/saved_models/${domain}/${dataset_name}/wgan_dominate"
mkdir -p "${checkpoint_save_path}"
cd ./scripts || exit

# Train args
device="gpu"
epochs=400
checkpoint_every_n=10
# Data args
batch_size=64
validation_batch_multiplier=1
train_split=0.95
# Generator_args
train_gen_every=6
gen_viz_every=6
# Discriminator Model args
activation="max_out"
max_out=16
dropout="--dropout"
discriminator_model_type="cifar10_discriminator"
d_hidden_dim=1024
# Discriminator optim args
discriminator_optim_type="adam"
projected_gradient_descent="--projected_gradient_descent"

python wgan_dominate.py \
  --wgan_chkpt_file "${wgan_chkpt_file}" \
  --wgan_chkpt_args "${wgan_chkpt_args}" \
  --choquet_weight ${choquet_weight} \
  --max_choquet_epochs ${max_choquet_epochs} \
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
  --train_gen_every ${train_gen_every} \
  --gen_viz_every ${gen_viz_every} \
  --activation "${activation}" \
  --max_out ${max_out} \
  ${dropout} \
  --choquet_reg_type "${choquet_reg_type}" \
  --choquet_reg_lambda "${choquet_reg_lambda}" \
  --discriminator_model_type "${discriminator_model_type}" \
  --d_hidden_dim ${d_hidden_dim} \
  --discriminator_optim_type "${discriminator_optim_type}" \
  ${projected_gradient_descent} \
  --d_lr ${d_lr} \
  --d_weight_decay ${d_weight_decay}
