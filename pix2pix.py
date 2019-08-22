import os
import random
import glob
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


# Patch-GAN
# patch-size = 64
class PatchGAN(nn.Module):
    def __init__(self, d):
        super(PatchGAN, self).__init__()
        self.conv1 = self.conv(6, d)
        self.conv2 = self.conv(d, 2 * d)
        self.batch_norm2 = nn.BatchNorm2d(2 * d)
        self.conv3 = self.conv(2 * d, 4 * d)
        self.batch_norm3 = nn.BatchNorm2d(4 * d)
        self.conv4 = self.conv(4 * d, 8 * d)
        self.batch_norm4 = nn.BatchNorm2d(8 * d)
        self.conv5 = nn.Conv2d(8 * d, 1, kernel_size=4, stride=1, padding=0)

    def conv(self, i, o):
        return nn.Conv2d(i, o, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.batch_norm2(self.conv2(x))
        x = F.leaky_relu(x, 0.2)
        x = self.batch_norm3(self.conv3(x))
        x = F.leaky_relu(x, 0.2)
        x = self.batch_norm4(self.conv4(x))
        x = F.leaky_relu(x, 0.2)
        x = torch.sigmoid(self.conv5(x))
        return torch.mean(torch.mean(x, dim=3), dim=2).squeeze()


# d is the number of features
# layers is the number of layers of downsampling. image_size should be a multiple of 2**layers
# i.e. layers = 5 --> img_size % 32 == 0
class UGenerator(nn.Module):
    def __init__(self, d, layers=3):
        super(UGenerator, self).__init__()
        self.layers = layers
        self.conv_in = nn.Conv2d(3, d, kernel_size=4, stride=2, padding=1)
        self.convs = list()
        self.deconvs = list()
        self.bns = list()
        self.debns = list()
        for i in range(layers):
            c = 2**i
            self.convs.append(nn.Conv2d(c * d, 2 * c * d, kernel_size=4, stride=2, padding=1))
            self.deconvs.append(nn.ConvTranspose2d(4 * c * d, c * d, kernel_size=4, stride=2, padding=1))
            self.bns.append(nn.BatchNorm2d(2 * c * d))
            self.debns.append(nn.BatchNorm2d(c * d))
            self.add_module(f'conv{i}', self.convs[i])
            self.add_module(f'bn{i}', self.bns[i])
            self.add_module(f'deconv{i}', self.deconvs[i])
            self.add_module(f'debn{i}', self.debns[i])
        self.deconv_out = nn.ConvTranspose2d(2 * d, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x0):
        xs = list()
        x_in = self.conv_in(x0)
        x = x_in
        for i in range(self.layers):
            conv = self.convs[i]
            bn = self.bns[i]
            x = conv(F.leaky_relu(x, 0.2))
            if i < self.layers - 1:
                x = bn(x)
            xs.append(x)
        y = x
        for i in range(self.layers,0,-1):
            deconv = self.deconvs[i-1]
            debn = self.debns[i-1]
            y_ = torch.cat((y, xs[i-1]), dim=1)
            y = debn(deconv(F.relu(y_)))
        y_out = self.deconv_out(F.relu(torch.cat((y, x_in), dim=1)))
        return torch.tanh(y_out)

def get_next(dataloader):
    batch = next(dataloader)
    x = batch[0] # batch[1] contains labels
    s = x.size()
    w = s[3] // 2
    left = x[:, :, :, :w]
    right = x[:, :, :, w:]
    return (left, right)

def fake_labels(batch_size):
    labels = []
    for _ in range(batch_size):
        x = random.uniform(0, 0.1)
        labels.append(x)
    return torch.FloatTensor(labels)

def real_labels(batch_size):
    labels = []
    for _ in range(batch_size):
        x = random.uniform(0.9, 1)
        labels.append(x)
    return torch.FloatTensor(labels)

def step(generator, critic, optimizer_g, optimizer_c, dataiter, device, options):
    criterion = nn.BCELoss()
    l1 = nn.L1Loss()

    # Discriminator pass
    # Real datum
    optimizer_c.zero_grad()
    inputs, outputs = get_next(dataiter)
    labels = real_labels(options.batch_size)

    inputs = inputs.to(device)
    outputs = outputs.to(device)
    labels = labels.to(device)

    predictions = critic(torch.cat((inputs, outputs), dim=1))
    critic_loss_real = criterion(predictions, labels)

    # Fake datum
    optimizer_c.zero_grad()
    inputs, _ = get_next(dataiter)
    labels = fake_labels(options.batch_size)

    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = generator(inputs).detach()

    predictions = critic(torch.cat((inputs, outputs), dim=1))
    critic_loss_fake = criterion(predictions, labels)

    critic_loss = critic_loss_real + critic_loss_fake
    critic_loss.backward()
    optimizer_c.step()

    # Generator pass
    optimizer_g.zero_grad()
    inputs, outputs = get_next(dataiter)
    labels = torch.ones(options.batch_size)

    inputs = inputs.to(device)
    labels = labels.to(device)
    generated = generator(inputs)

    predictions = critic(torch.cat((inputs, generated), dim=1))
    generator_loss = criterion(predictions, labels) + options.l1_weight * l1(predictions, labels)
    generator_loss.backward()
    optimizer_g.step()

    return generator_loss.item(), critic_loss.item()

def train(generator, critic, optimizer_g, optimizer_c, dataiter, device, options):

    for epoch in range(options.epochs):
        print(f'{epoch}/{options.epochs}')
        generator_loss, critic_loss = step(generator, critic, optimizer_g, optimizer_c, dataiter, device, options)
        print('Generator loss: %.6f / Discriminator loss: %.6f' % (generator_loss, critic_loss))

        if epoch % options.sample_every == 0:
            inputs, origs = get_next(dataiter)
            input = inputs[0].to(device)
            output = generator(input.unsqueeze(0)).squeeze()
            image_tensor = torch.cat((output.data.to('cpu'), input.data.to('cpu'), origs[0]), dim=2)
            image = transforms.ToPILImage()((image_tensor + 1) / 2)
            image.save(os.path.join(options.output_dir, f'sample_{epoch}.jpg'))

def parse_options():
    parser = argparse.ArgumentParser(description='Pix2pix with Wasserstein GAN')
    parser.add_argument('--data-dir', required=True, help='directory with the images')
    parser.add_argument('--output-dir', required=True, help='directory where to store the generated images')
    parser.add_argument('--model-dir', required=True, help='directory where to store the models')
    parser.add_argument('--restore', action='store_true', help='restart training from the saved models')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--generator-lr', type=float, default=2e-4, help='learning rate for the generator')
    parser.add_argument('--critic-lr', type=float, default=2e-4, help='learning rate for the critic')
    parser.add_argument('--l1-weight', type=float, default=1, help='weight of the L1 loss')
    parser.add_argument('--generator-channels', type=int, default=8, help='number of channels for the generator')
    parser.add_argument('--generator-layers', type=int, default=3, help='number of layers for the generator')
    parser.add_argument('--critic-channels', type=int, default=4, help='number of channels for the critic')
    parser.add_argument('--gradient-penalty', type=float, default=10, help='gradient penalty')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--sample-every', type=int, default=1000, help='how often to sample images')

    return parser.parse_args()

def run():
    options = parse_options()
    print(options)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(options.data_dir, exist_ok=True)
    os.makedirs(options.output_dir, exist_ok=True)
    os.makedirs(options.model_dir, exist_ok=True)

    with open(os.path.join(options.output_dir, 'options.json'), 'w') as f:
        json.dump(vars(options), f, indent=4)

    if options.restore:
        generator = torch.load(os.path.join(options.model_dir, 'generator.pt'))
        critic = torch.load(os.path.join(options.model_dir, 'critic.pt'))
    else:
        generator = UGenerator(options.generator_channels, options.generator_layers)
        critic = PatchGAN(options.critic_channels)
    generator = generator.to(device)
    critic = critic.to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=options.generator_lr, betas=(0.5, 0.999))
    optimizer_c = optim.Adam(critic.parameters(), lr=options.critic_lr, betas=(0.5, 0.999))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = ImageFolder(options.data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    dataiter = (x for _ in range(10000) for x in dataloader)

    train(generator, critic, optimizer_g, optimizer_c, dataiter, device, options)

if __name__ == '__main__':
    run()