#!/bin/bash

export PYTHONPATH="${PWD}:${PWD}/src"

# Data domain
domain="distributions"
data="swiss_roll"  # "circle_of_gaussians", "swiss_roll", or "image_point_cloud"
if [ ${data} = "circle_of_gaussians" ]; then
  data_params="--n_gaussians 8 --std 0.2 --radius 2"
elif [ ${data} = "swiss_roll" ]; then
  data_params="--noise 0.5"
else
  data_params="--image_name github_icon --image_path ${PWD}/assets/github.png"
fi

checkpoint_save_path="${PWD}/saved_models/${domain}/${data}/choquet/"
mkdir -p "${checkpoint_save_path}"
cd ./scripts || exit

version_number=0
seed=0
epochs=100001
checkpoint_every_n=1000
batch_size=512
validation_batch_multiplier=4
# Generator_args
z_dim=32
g_hidden_dim=32
g_n_layers=10
train_gen_every=6
gen_viz_every=1000
generator_model_type="distribution_generator"
generator_optim_type="adam"
g_lr=0.0005
# Discriminator setup args
disc_or_dist="dist"  # "disc" or "dist"
how_to_combine_integral_terms="sum"
split_regularization="--split_regularization"
grad_reg_lambda=0
grad_reg_wrt="generator_parameters" # "generator_parameters" or "interpolates"
# Discriminator Model args
discriminator_model_type="distribution_discriminator"
activation="max_out"
max_out=2
d_hidden_dim=32
d_n_layers=5
# Discriminator optim args
discriminator_optim_type="adam"
projected_gradient_descent="--projected_gradient_descent"
d_lr=0.0001

python gan_train.py  'choquet' \
  --seed ${seed} \
  --epochs ${epochs} \
  --batch_size ${batch_size} \
  --validation_batch_multiplier ${validation_batch_multiplier} \
  --num_workers 0 \
  --checkpoint_save_path "${checkpoint_save_path}" \
  --checkpoint_every_n ${checkpoint_every_n} \
  --restart_from_last \
  --version_number "${version_number}" \
  --device "cpu" \
  --num_devices 1 \
  --domain "${domain}" \
  --distribution_type "${data}" \
  ${data_params} \
  --disc_or_dist "${disc_or_dist}" \
  ${split_regularization} \
  --how_to_combine_integral_terms "${how_to_combine_integral_terms}" \
  --z_dim ${z_dim} \
  --grad_reg_lambda ${grad_reg_lambda} \
  --grad_reg_wrt "${grad_reg_wrt}" \
  --generator_model_type "${generator_model_type}" \
  --generator_optim_type "${generator_optim_type}" \
  --discriminator_model_type "${discriminator_model_type}" \
  --discriminator_optim_type "${discriminator_optim_type}" \
  --train_gen_every ${train_gen_every} \
  --gen_viz_every ${gen_viz_every} \
  --g_hidden_dim ${g_hidden_dim} \
  --g_n_layers ${g_n_layers} \
  --g_lr ${g_lr} \
  ${projected_gradient_descent} \
  --activation ${activation} \
  --max_out ${max_out} \
  --d_hidden_dim ${d_hidden_dim} \
  --d_n_layers ${d_n_layers} \
  --d_lr ${d_lr}
