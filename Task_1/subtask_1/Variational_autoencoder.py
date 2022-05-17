import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functions
from torch.utils.data import DataLoader, TensorDataset


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_vector_size, h_dim=350, z_dim=10):
        super(VariationalAutoencoder, self).__init__()
        self.criterion = None
        self.input_vector_size = input_vector_size
        self.fc1 = nn.Linear(input_vector_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_vector_size // 2)

    def encode(self, input_data):
        h = functions.relu(self.fc1(input_data))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = functions.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, input_data):
        mu, log_var = self.encode(input_data)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

    def fit(self, input_data: np.ndarray, batch_size, num_epochs, criterion, optimizer, device):

        print("=============== Learning stage ===============")
        data_loader = DataLoader(TensorDataset(torch.from_numpy(input_data)), batch_size=batch_size, shuffle=True)
        self.criterion = criterion

        for epoch in range(num_epochs):
            for data_point in data_loader:
                # Forward pass
                data_point = data_point[0].to(device).view(-1, self.input_vector_size)
                x_reconst, mu, log_var = self.forward(data_point.float())

                # Compute reconstruction loss and kl divergence
                reconst_loss = criterion(x_reconst, data_point)
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                # Backprop and optimize
                loss = reconst_loss + kl_div
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Epoch[{}/{}], Loss: {:.4f} Reconst Loss: {:.4f}, KL Div: {:.8f}".format(epoch + 1, num_epochs, loss.item(),
                                                                              reconst_loss.item(), kl_div.item()))

    def validate(self, input_data, batch_size, criterion, device):
        print("=============== Validating stage ===============")
        data_loader = DataLoader(TensorDataset(torch.from_numpy(input_data)), batch_size=batch_size, shuffle=True)

        for data_point in data_loader:
            # Forward pass
            data_point = data_point[0].to(device).view(-1, self.input_vector_size)
            x_reconst, mu, log_var = self.forward(data_point.float())

            # Compute reconstruction loss and kl divergence
            reconst_loss = criterion(x_reconst, data_point)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Backprop and optimize
            loss = reconst_loss + kl_div

        # log
        print("Loss: {:.4f} Reconst Loss: {:.4f}, KL Div: {:.8f}".format(loss.item(), reconst_loss.item(), kl_div.item()))
