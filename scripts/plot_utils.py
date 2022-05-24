from collections import OrderedDict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.utils import make_grid

from src.model_utils import calculate_grad_wrt_x


def plot_distribution_domain(epoch, val_batch, z, generator, choquet, mode='choquet'):
    figures = OrderedDict()

    # Real vs Generated scatter plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    title = 'Real vs. Generated Data'
    data, _ = val_batch
    real = data.detach().cpu().numpy()
    gen_z = generator(z)
    fake = gen_z.detach().cpu().numpy()
    if hasattr(choquet, 'how_to_combine_integral_terms') and choquet.how_to_combine_integral_terms == 'min':
        with torch.no_grad():
            disc_on_gen = choquet(gen_z)
            disc_on_real = choquet(data)
            title += f'u_0(gen) - u_0(real) = {(disc_on_gen[0].mean() - disc_on_real[0].mean()).item():0.4f}\n' + \
                     f'u_1(real) - u_1(gen) = {(disc_on_real[1].mean() - disc_on_gen[1].mean()).item():0.4f}'
            plt.subplots_adjust(top=0.85)
    ax.set_title(title)
    ax.scatter(real[:, 0], real[:, 1], edgecolor='none', alpha=0.6)
    ax.scatter(fake[:, 0], fake[:, 1], c='g', edgecolor='none', alpha=0.6)
    figures[f'real_vs_fake_ep{epoch}.png'] = {'fig': fig, 'title': title}

    # Density estimation plot of generated data
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    title = f'Density estimation'
    ax.set_title(title)
    ax.set_facecolor(sns.color_palette('Greens', n_colors=256)[0])
    try:
        sns.kdeplot(ax=ax, x=fake[:, 0], y=fake[:, 1], shade=True,
                    cmap='Greens', n_levels=20, clip=[[real.min(), real.max()]] * 2)
    except ValueError as ve:
        print(ve)
    figures[f'density_estimation_ep{epoch}.png'] = {'fig': fig, 'title': title}

    if mode == 'choquet':
        # Discriminator gradient with respect to Generator output
        using_symmetric_d = hasattr(choquet, 'split_regularization')  # check if we are using symmetric discriminator
        fig = plt.figure(figsize=(5 * (1 + using_symmetric_d), 5), tight_layout=True)
        title = 'Discriminator Grad wrt Generator output'
        fig.suptitle(title)
        if using_symmetric_d:
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_title(f'Discriminator 1 Step {epoch}')
            grad_u = calculate_grad_wrt_x(choquet.critic_0, gen_z).cpu().numpy()
            CS = ax1.scatter(grad_u[:, 0], grad_u[:, 1], c=np.linalg.norm(grad_u, axis=1))
            # ax1.quiver(fake[:, 0], fake[:, 1], grad_u[:, 0], grad_u[:, 1], np.linalg.norm(grad_u, axis=1))
            ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)
            ax2.set_title(f'Discriminator 2 Step {epoch}')
            ax2.tick_params(labelleft=False)
            grad_u = calculate_grad_wrt_x(choquet.critic_1, gen_z).cpu().numpy()
            ax2.scatter(grad_u[:, 0], grad_u[:, 1], c=np.linalg.norm(grad_u, axis=1))
            # ax2.quiver(fake[:, 0], fake[:, 1], grad_u[:, 0], grad_u[:, 1], np.linalg.norm(grad_u, axis=1))
        else:
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(f'Step {epoch}')
            grad_u = calculate_grad_wrt_x(choquet.critic, gen_z).cpu().numpy()
            CS = ax.scatter(grad_u[:, 0], grad_u[:, 1], c=np.linalg.norm(grad_u, axis=1))
            # ax.quiver(fake[:, 0], fake[:, 1], grad_u[:, 0], grad_u[:, 1], np.linalg.norm(grad_u, axis=1))
        divider = make_axes_locatable(fig.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        fig.colorbar(CS, cax=cax)
        fig.tight_layout()
        figures[f'd_grad_wrt_g_out_ep{epoch}.png'] = {'fig': fig, 'title': title}

        # Discriminator contour plot
        fig = plt.figure(figsize=(5 * (1 + 2 * using_symmetric_d), 5))
        title = 'Discriminator Contour plot'
        fig.suptitle(title)
        ax = fig.add_subplot(1, (3 if using_symmetric_d else 1), 1)
        ax.set_title(f'Discriminator Step {epoch}')
        contour_pts = 500
        X_dim0 = torch.linspace(fake[:, 0].min() - 0.1, fake[:, 0].max() + 0.1, contour_pts)
        X_dim1 = torch.linspace(fake[:, 1].min() - 0.1, fake[:, 1].max() + 0.1, contour_pts)
        x_cont, y_cont = torch.meshgrid(X_dim0, X_dim1, indexing='ij')
        X_cont = torch.cat([x_cont.reshape(-1, 1), y_cont.reshape(-1, 1)], dim=1)
        with torch.no_grad():
            if using_symmetric_d:
                discriminator_output = choquet(X_cont)
                Z_cont = (discriminator_output[0] - discriminator_output[1]).reshape(contour_pts, -1)
            else:
                Z_cont = choquet.critic(X_cont).reshape(contour_pts, -1)
        CS = ax.contourf(X_cont[:, 0].reshape(Z_cont.shape), X_cont[:, 1].reshape(Z_cont.shape), Z_cont)
        if using_symmetric_d:
            ax2 = fig.add_subplot(1, 3, 2, sharex=ax, sharey=ax)
            ax2.set_title(f'Discriminator 1 Step {epoch}')
            ax2.tick_params(labelleft=False)
            with torch.no_grad():
                Z_cont = choquet.critic_0(X_cont).reshape(contour_pts, -1)
            ax2.contourf(X_cont[:, 0].reshape(Z_cont.shape), X_cont[:, 1].reshape(Z_cont.shape), Z_cont)
            ax3 = fig.add_subplot(1, 3, 3, sharex=ax, sharey=ax)
            ax3.set_title(f'Discriminator 2 Step {epoch}')
            ax3.tick_params(labelleft=False)
            with torch.no_grad():
                Z_cont = choquet.critic_1(X_cont).reshape(contour_pts, -1)
            ax3.contourf(X_cont[:, 0].reshape(Z_cont.shape), X_cont[:, 1].reshape(Z_cont.shape), Z_cont)
        divider = make_axes_locatable(fig.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        fig.colorbar(CS, cax=cax)
        fig.tight_layout()
        figures[f'd_contour_ep{epoch}.png'] = {'fig': fig, 'title': title}
    return figures


def plot_image_domain(epoch, val_batch, z, generator, choquet, mean=None, std=None, cmap=None):
    figures = OrderedDict()
    indices = torch.randperm(z.shape[0])[:64]
    z = z[indices]
    gen_z = generator(z)
    fake = gen_z.detach().cpu()
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    title = 'Generated samples'
    grid = make_grid(fake, nrow=8)
    grid = grid * std + mean if (mean is not None and std is not None) else grid
    ax.imshow(np.transpose(grid.numpy(), (1, 2, 0)), cmap=cmap)
    figures[f'generated_images_ep{epoch}.png'] = {'fig': fig, 'title': title}
    return figures


def get_plot_function(mode, args):
    mean_std_dict = {
        'mnist': {'mean': 0.1307, 'std': 0.3081},
        'fashion_mnist': {'mean': 0.2860, 'std': 0.3530},
        'cifar10': {
            'mean': torch.tensor([0.5, 0.5, 0.5]).reshape((3, 1, 1)),
            'std': torch.tensor([0.5, 0.5, 0.5]).reshape((3, 1, 1))
        }
    }
    plot_func_dict = {
        'distributions': partial(plot_distribution_domain, mode=mode),
        'images': partial(plot_image_domain,
                          mean=np.array(mean_std_dict[args.dataset_name]['mean']) if args.domain == 'images' else None,
                          std=np.array(mean_std_dict[args.dataset_name]['std']) if args.domain == 'images' else None,
                          cmap='gray' if args.dataset_name == 'mnist' else None,
                          )
    }
    return plot_func_dict[args.domain]
