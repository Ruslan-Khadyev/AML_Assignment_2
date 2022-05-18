import torch.nn as nn
from Task_2.subtask_1.routine import *
from Task_2.subtask_1.Discriminator import Discriminator
from Task_2.subtask_1.Generator import Generator
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


class ConditionalGAN(nn.Module):
    def __init__(self, input_shape, n_classes, latent_dim):
        super(ConditionalGAN, self).__init__()
        # define discriminator and generator
        self.discriminator_optim = None
        self.generator_optim = None
        self.criterion = None

        self.discriminator = Discriminator(input_shape, n_classes, latent_dim)
        self.generator = Generator(input_shape, n_classes, latent_dim)

        self.latent_dim = latent_dim
        self.tf_writer = SummaryWriter()

    def compile(self, optimizer, learning_rate=0.0005, criterion=nn.BCELoss()):
        # initialize model weights
        self.discriminator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        self.discriminator_optim = optimizer(self.discriminator.parameters(), learning_rate)
        self.generator_optim = optimizer(self.generator.parameters(), learning_rate)

        self.criterion = criterion

    def save(self, path):
        torch.save(self.state_dict(), path)

    def fit(self, input_data: np.ndarray, labels: np.ndarray, batch_size, num_epochs, device, print_every=1000):
        print("=============== Learning stage ===============")
        # create data loader
        data_loader = DataLoader(TensorDataset(torch.from_numpy(input_data), torch.from_numpy(labels)),
                                 batch_size=batch_size, shuffle=True)

        self.discriminator.to(device)
        self.generator.to(device)

        discriminator_loss_value = 0
        generator_loss_value = 0
        kl_loss_value = 0

        for epoch in range(num_epochs):
            for batch_i, (data, labels) in enumerate(data_loader):

                current_batch_size = data.size(dim=0)

                real_batched_data = data.float().to(device)
                fake_batched_data = generate_latent_points_tensors(self.latent_dim, current_batch_size).to(device)
                batched_labels = labels.to(device)

                # 1. Train the discriminator on real and fake images
                self.discriminator_optim.zero_grad()

                # Get the real loss of discriminator
                discriminator_real_images = self.discriminator.forward_tensor(real_batched_data, batched_labels)
                discriminator_real_loss = real_loss(discriminator_real_images, self.criterion, device)

                # Get the fake loss of dDiscriminator
                generated_fake_images = self.generator.forward_tensor(fake_batched_data, batched_labels)
                discriminator_fake_images = self.discriminator.forward_tensor(generated_fake_images, batched_labels)

                discriminator_loss = discriminator_real_loss + fake_loss(discriminator_fake_images, self.criterion,
                                                                         device)

                discriminator_loss.backward()
                self.discriminator_optim.step()

                # 2. Train the generator with an adversarial loss

                self.generator_optim.zero_grad()
                generated_fake_images = self.generator.forward_tensor(fake_batched_data, batched_labels)
                generator_loss = real_loss(self.discriminator.forward_tensor(generated_fake_images, batched_labels),
                                           self.criterion, device)

                KL_loss = torch.sum(torch.kl_div(generated_fake_images, real_batched_data))

                generator_loss.backward()
                self.generator_optim.step()

                # Print some loss stats
                if batch_i % print_every == 0:
                    # print discriminator and generator loss
                    discriminator_loss_value = discriminator_loss.item()
                    generator_loss_value = generator_loss.item()
                    kl_loss_value = KL_loss

                    # append discriminator loss and generator loss
                    self.tf_writer.add_scalar("discriminator loss", discriminator_loss, epoch)
                    self.tf_writer.add_scalar("generator loss", generator_loss, epoch)
                    self.tf_writer.add_scalar("Kullback–Leibler (KL) divergence", KL_loss, epoch)

            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f} | KL loss: {:6.4f}'.format(
                epoch + 1, num_epochs, discriminator_loss_value, generator_loss_value, kl_loss_value))

        self.tf_writer.flush()
        self.tf_writer.close()


