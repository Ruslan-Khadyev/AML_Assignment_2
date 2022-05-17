import torch
from torch.utils.data import DataLoader, TensorDataset
from Undercomplete_autoencoder import UnderCompleteAutoencoder
import numpy as np


class RegularizedAutoencoder(UnderCompleteAutoencoder):
    def __init__(self, input_size, latent_dim):
        super().__init__(input_size, latent_dim)
        self.l1_lambda = 0.01

    def fit(self, input_data: np.ndarray, batch_size, num_epochs, criterion, optimizer, device):
        print("=============== Learning stage ===============")
        data_loader = DataLoader(TensorDataset(torch.from_numpy(input_data)), batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for data_point in data_loader:
                data_point = data_point[0].to(device)

                optimizer.zero_grad()
                # forward
                output = self.forward(data_point.float())
                loss = criterion(output, data_point)

                l1_penalty = 0.
                for param in self.parameters():
                    l1_penalty += torch.abs((torch.norm(param, 1)))
                l1_penalty *= self.l1_lambda

                loss += l1_penalty

                # backward
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # log
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    def validate(self, input_data, batch_size, criterion, device):
        print("=============== Validating stage ===============")
        data_loader = DataLoader(TensorDataset(torch.from_numpy(input_data)), batch_size=batch_size, shuffle=True)
        epoch_loss = 0.0
        for data_point in data_loader:
            data_point = data_point[0].to(device)

            # forward
            output = self.forward(data_point.float())
            loss = criterion(output, data_point)
            epoch_loss += loss.item()

            l1_penalty = 0.
            for param in self.parameters():
                l1_penalty += torch.abs((torch.norm(param, 1)))
            l1_penalty *= self.l1_lambda

            loss += l1_penalty

        # log
        print('loss: {:.4f}'.format(loss.item()))