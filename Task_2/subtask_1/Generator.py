import torch.nn as nn
import numpy as np
import torch


class Generator(nn.Module):
    def __init__(self, input_shape, n_classes, latent_dim):
        """
        Initialize the Discriminator Module
        """
        super(Generator, self).__init__()

        #   label embedding
        self.embedding_g = nn.Embedding(n_classes, latent_dim)
        self.fc_1_g = nn.Linear(2 * latent_dim, 128)
        self.fc_2_g = nn.Linear(128, 256)
        self.fc_3_g = nn.Linear(256, 512)
        self.fc_4_g = nn.Linear(512, input_shape)
        self.latent_dim = latent_dim

    def forward(self, noise_z: np.ndarray, labels_y: np.ndarray) -> torch.Tensor:
        labels_y_tensor = torch.from_numpy(labels_y).type(torch.LongTensor)
        noise_z_tensor = torch.from_numpy(noise_z)
        return self.forward_tensor(noise_z_tensor, labels_y_tensor)

    def forward_tensor(self, noise_vector_z: torch.Tensor, labels_y: torch.LongTensor) -> torch.Tensor:
        embedded = self.embedding_g(labels_y)
        concatenated_noise_labels = torch.cat((noise_vector_z, embedded), dim=1).float()
        fc_1 = torch.relu(self.fc_1_g(concatenated_noise_labels))
        fc_2 = torch.relu(self.fc_2_g(fc_1))
        fc_3 = torch.relu(self.fc_3_g(fc_2))
        output_fc = torch.tanh(self.fc_4_g(fc_3))
        return output_fc
