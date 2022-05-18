import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimplePerceptron(nn.Module):
    def __init__(self, input_shape: int, n_classes: int):
        super(SimplePerceptron, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_classes)

        self.optimizer = None
        self.criterion = None

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.log_softmax(x, dim=1)

    def compile(self, optimizer, learning_rate=0.001, criterion=nn.CrossEntropyLoss()):
        self.optimizer = optimizer(self.parameters(), learning_rate)
        self.criterion = criterion

    def fit(self, epoch_num: int, batch_size: int, x_data, y_labels, device):
        print("{:-^70s}".format(" Learning stage "))
        # create data loader
        data_loader = DataLoader(TensorDataset(torch.from_numpy(x_data), torch.from_numpy(y_labels)),
                                 batch_size=batch_size, shuffle=True)

        self.train()
        for epoch in range(epoch_num):
            loss_val = 0
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.float().to(device), target.type(torch.LongTensor).to(device)
                self.optimizer.zero_grad()
                output = self.forward(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                loss_val = loss.item()

            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch + 1, loss_val))

    def predict(self, x_test, device):
        _, pred = torch.max(self.forward(torch.from_numpy(x_test).float().to(device)), dim=1)
        return pred.data.cpu().detach().numpy()




