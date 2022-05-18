import torch.nn as nn
import numpy as np
import torch


class Discriminator(nn.Module):

    def __init__(self, input_shape, n_classes, latent_dim):
        """
        Initialize the Discriminator Module
        """
        super(Discriminator, self).__init__()

        #   label embedding
        self.embedding_d = nn.Embedding(n_classes, latent_dim)
        self.fc_1_d = nn.Linear(latent_dim, input_shape)

        self.fc_2_d = nn.Linear(input_shape * 2, 512)
        self.fc_3_d = nn.Linear(512, 256)
        self.fc_4_d = nn.Linear(256, 128)
        self.fc_5_d = nn.Linear(128, 1)

    def forward(self, data: np.ndarray, labels_y: np.ndarray) -> torch.Tensor:
        labels_y_tensor = torch.from_numpy(labels_y).type(torch.LongTensor)
        data_tensor = torch.from_numpy(data)
        return self.forward_tensor(data_tensor, labels_y_tensor)

    def forward_tensor(self, data: torch.Tensor, labels_y: torch.LongTensor) -> torch.Tensor:
        embedded = self.embedding_d(labels_y)
        embedded_fc = torch.relu(self.fc_1_d(torch.squeeze(embedded)))
        concatenated_data_labels = torch.cat((data, embedded_fc), dim=1)
        fc_2 = torch.relu(self.fc_2_d(concatenated_data_labels))
        fc_3 = torch.relu(self.fc_3_d(fc_2))
        fc_4 = torch.relu(self.fc_4_d(fc_3))
        output_fc = torch.sigmoid(self.fc_5_d(fc_4))
        return output_fc
