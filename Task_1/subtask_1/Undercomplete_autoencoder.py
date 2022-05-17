import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class UnderCompleteAutoencoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(UnderCompleteAutoencoder, self).__init__()
        # Step 1 : Define the encoder
        # Step 2 : Define the decoder
        # Step 3 : Initialize the weights (optional)
        self.criterion = None
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(True),
            nn.Linear(input_size // 2, input_size // 3),
            nn.Linear(input_size // 3, input_size // 4),
            nn.Tanh(),
            nn.Linear(input_size // 4, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_size // 8),
            nn.ReLU(True),
            nn.Linear(input_size // 8, input_size // 6),
            nn.Linear(input_size // 6, input_size // 4),
            nn.Tanh(),
            nn.Linear(input_size // 4, input_size // 2)
        )
        self.encoder.apply(self.__init_weights)
        self.decoder.apply(self.__init_weights)

    def forward(self, x):
        # Step 1: Pass the input through encoder to get latent representation
        # Step 2: Take latent representation and pass through decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, input):
        # Step 1: Pass the input through the encoder to get latent representation
        return self.encoder(input)

    def fit(self, input_data: np.ndarray, batch_size, num_epochs, criterion, optimizer, device):

        print("=============== Learning stage ===============")
        data_loader = DataLoader(TensorDataset(torch.from_numpy(input_data)), batch_size=batch_size, shuffle=True)
        self.criterion = criterion

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for data_point in data_loader:
                data_point = data_point[0].to(device)
                optimizer.zero_grad()

                # forward
                output = self.forward(data_point.float())
                loss = criterion(output, data_point)

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
        # log
        print('loss: {:.4f}'.format(loss.item()))

    def __init_weights(self, m):
        # Init the weights (optional)
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)