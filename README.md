 # Learning with Stochastic Orders
This repository implements the algorithms and experiments described in [Learning with Stochastic orders](TODO_need_link).

## 0. Install
To get started, create and activate the `conda` environment below:
```shell
conda env create -f gmorder_env.yml
conda activate gmorder_env
```

## 1. Instructions to run 1D portfolio optimization
To run the 1D portfolio optimization experiment open and execute the [`portfolio_optimization`](notebooks/portfolio_optimization.ipynb) notebook. 

## 2. Generative modeling with d<sub>CT
<p float="left">
    <img src="assets/swiss_roll.gif" alt="swiss roll training" width="150px"/>
    <img src="assets/gaussians.gif" alt="gaussians training" width="150px"/>
    <img src="assets/github.gif" alt="github icon training" width="150px"/>
</p>

To run the GAN training using the Choquet-Toland (CT) distance use the shell script below:
```shell
sh run_choquet_train_distributions.sh
```
Open this script and change `data` (Line 7) to one of `circle_of_gaussians`, `swiss_roll`, `image_point_cloud`.

## 3. Baseline generator domination with VDC
<p float="center">
<img src="assets/cifar10_vdc.png" alt="CIFAR10 generation" width="300px"/>
</p>

To train a baseline WGAN-GP model run
```shell
sh run_wgan_train_images.sh
```

Once training is complete, to reproduce the WGAN-GP + VDC results from the paper, execute:
```shell
sh run_wgan_dominate_images.sh
```
If needed, change file paths in this script to point to where the WGAN-GP checkpoint file and hyperparameter args are saved.

## Acknowledgements
For several of our generator, discriminator, and Choquet critics, we draw inspiration and leverage code from the following public Github repositories:
1. https://github.com/caogang/wgan-gp,
2. https://github.com/ozanciga/gans-with-pytorch
3. https://github.com/CW-Huang/CP-Flow

## Citation:
To cite our work please use: TODO: need bibtex citation here 
```
@article{name,
  title={Learning with Stochastic Orders},
  author={Domingo-Enrich, Carles and Schiff, Yair and Mroueh, Youssef},
  year={2022}
}
```