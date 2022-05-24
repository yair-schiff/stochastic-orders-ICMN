import torchvision.datasets as datasets
import torchvision.transforms as transforms

from src.data import ImagePointCloud, CircleOfGaussians, SwissRoll


def build_transforms(dataset_name):
    transforms_list = []
    # Case statement for dispatching different datasets' normalization factors:
    if dataset_name == 'mnist':
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(0.1307, 0.3081))
        # transforms_list.append(transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)))
    elif dataset_name == 'fashion_mnist':
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(0.2860, 0.3530))
        # transforms_list.append(transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)))
    elif dataset_name == 'cifar10':
        transforms_list.append(transforms.ToTensor())
        # transforms_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)))
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif dataset_name == 'svhn':
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)))
    return transforms.Compose(transforms_list)


def get_dataset(dataset_name):
    DATASETS = {'mnist': datasets.MNIST,
                'fashion_mnist': datasets.FashionMNIST,
                'cifar10': datasets.CIFAR10,
                'svhn': datasets.SVHN}
    assert dataset_name in DATASETS.keys(), \
        f'Dataset {dataset_name} not supported. Use one of: {DATASETS.keys()}.'
    return DATASETS[dataset_name]


def get_distribution(distribution_type, distribution_params):
    ALLOWED_DISTRIBUTIONS = ['gaussian', 'circle_of_gaussians', 'swiss_roll', 'image_point_cloud']
    distribution = None
    assert distribution_type in ALLOWED_DISTRIBUTIONS, \
        f'Distribution {distribution_type} not supported. Use one of: {ALLOWED_DISTRIBUTIONS}.'
    if distribution_type == 'circle_of_gaussians':
        distribution = CircleOfGaussians(**distribution_params)
    elif distribution_type == 'swiss_roll':
        distribution = SwissRoll(**distribution_params)
    elif distribution_type == 'image_point_cloud':
        distribution = ImagePointCloud(**distribution_params)
    return distribution
