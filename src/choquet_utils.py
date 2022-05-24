import math

import torch


def get_model_grad_wrt_gen_params(model, z, generator):
    gen_z = generator(z)
    gen_params_with_grad = [g for g in generator.parameters() if g.requires_grad]
    fx = model(gen_z).mean()
    grads = torch.autograd.grad(fx, gen_params_with_grad, create_graph=True, retain_graph=True)[0]
    gr_norm_sq = None
    for gr in grads:
        if gr_norm_sq is None:
            gr_norm_sq = (gr ** 2).sum()
        else:
            gr_norm_sq += (gr ** 2).sum()
    return gr_norm_sq


def generate_interpolates(real_data, fake_data):
    batch_size = real_data.shape[0]
    alpha = torch.rand(batch_size, 1).type_as(real_data)
    alpha = alpha.expand(batch_size, real_data.nelement() // batch_size).contiguous().view(real_data.shape)
    return alpha * real_data + ((1 - alpha) * fake_data)


def get_model_grad_wrt_interpolates(model, real_data, fake_data, interpolates=None):
    if interpolates is None:
        interpolates = generate_interpolates(real_data, fake_data)
    interpolates = interpolates.detach().clone().requires_grad_(True)
    model_interpolates = model(interpolates)
    gradients = torch.autograd.grad(outputs=model_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(model_interpolates.size()).type_as(model_interpolates),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def count_nonzero_weights(model, positive_weight_module_names):
    count_non_zero = 0
    numel = 0
    for name, module in model.named_modules():
        if any(pos_name in name for pos_name in positive_weight_module_names):
            if hasattr(module, 'weight'):
                with torch.no_grad():
                    count_non_zero += torch.count_nonzero(module.weight.data).item()
                    numel += module.weight.data.numel()
    return count_non_zero / numel


def get_weight_norms(model):
    weight_norms = {}
    for name, module in model.named_modules():
        param_norm = 0
        numel = 0
        if name == '':
            continue
        for param in module.parameters():
            if param.requires_grad:
                with torch.no_grad():
                    param_norm += param.norm(2).detach().item() ** 2
                    numel += param.numel()
        weight_norms[name] = math.sqrt(param_norm) / numel if numel else 0
    return weight_norms


def project_weights_to_positive(model, positive_weight_module_names):
    for name, module in model.named_modules():
        if name in positive_weight_module_names:
            for pos_module in module.modules():
                if hasattr(pos_module, 'weight'):
                    with torch.no_grad():
                        pos_module.weight.data.relu_()


def test_model_convexity(model, batch, eps=1e-6):
    with torch.no_grad():
        batch_size = batch.shape[0]
        # Sample lambda uniformly
        lambdas = torch.rand((batch_size, 1)).type_as(batch)
        lambdas_expand = lambdas.expand(batch_size, batch.nelement() // batch_size).contiguous().view(
            batch.shape).type_as(batch)

        # Shuffle indices to randomly select images
        indices1 = torch.randperm(batch.shape[0])
        indices2 = torch.randperm(batch.shape[0])

        # Check that `f(lambda x + (1-lambda )y) <= lambda f(x) + (1-lambda) f(y)`
        convex_combo_of_inputs = model(lambdas_expand * batch[indices1] + (1 - lambdas_expand) * batch[indices2])
        convex_combo_of_outputs = lambdas * model(batch[indices1]) + (1 - lambdas) * model(batch[indices2])
        return torch.sum(convex_combo_of_inputs <= convex_combo_of_outputs + eps).item() / batch_size
