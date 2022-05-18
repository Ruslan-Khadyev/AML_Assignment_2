import torch
from routine import print_torch_version_info
from data_preprocess import DataLoader
from conditional_generative_adversarial_network import ConditionalGAN


def main():
    device = print_torch_version_info()

    data_loader = DataLoader("../UNSW_NB15_training-set.csv")
    print("the data has been loaded with shape:", data_loader.X_data.shape, "y shape:", data_loader.y_labels.shape)

    cgan = ConditionalGAN(input_shape=37, n_classes=10, latent_dim=100)
    cgan.compile(torch.optim.SGD)

    cgan.fit(data_loader.X_data, data_loader.y_labels, 64, 2000, device)

    cgan.save("./model/model_cgan_sgd_2000_epochs.pth")


if __name__ == "__main__":
    main()