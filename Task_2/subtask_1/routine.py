from torch.nn import init
import torch
import numpy as np


def generate_latent_points_tensors(latent_dim: int, n_samples: int) -> torch.Tensor:
    return torch.from_numpy(np.random.random((n_samples, latent_dim)))


def weights_init_normal(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def real_loss(discriminator_out, criterion, device):
    """
    Calculates how close discriminator outputs are to being real.
    param, D_out: discriminator logits
    return: real loss
    """

    batch_size = discriminator_out.size(0)
    labels = torch.FloatTensor(batch_size).uniform_(0.9, 1).to(device)

    loss = criterion(discriminator_out.squeeze(), labels)
    return loss


def fake_loss(discriminator_out, criterion, device):
    """
    Calculates how close discriminator outputs are to being fake.
    param, D_out: discriminator logits
    return: fake loss
    """

    batch_size = discriminator_out.size(0)
    labels = torch.FloatTensor(batch_size).uniform_(0, 0.1).to(device)

    loss = criterion(discriminator_out.squeeze(), labels)
    return loss


def print_torch_version_info() -> torch.device:
    print("using torch version:", torch.__version__)
    print("will use cuda:", torch.cuda.is_available())
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")