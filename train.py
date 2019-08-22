import os
import math
import argparse
import json
import importlib
from timeit import default_timer as timer

import torch
from torch import nn
from torch import autograd
import torchvision
from torchvision import transforms, datasets

log_statistics = importlib.util.find_spec("tensorboardX") is not None
if log_statistics:
    from tensorboardX import SummaryWriter

from models import MyConvo2d, Critic, Generator


def init_weights(m):
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                nn.init.kaiming_uniform_(m.conv.weight)
            else:
                nn.init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            nn.init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def gradient_penalty(critic, real_data, fake_data, penalty, device):
    n_elements = real_data.nelement()
    batch_size = real_data.size()[0]
    colors = real_data.size()[1]
    image_width = real_data.size()[2]
    image_height = real_data.size()[3]
    alpha = torch.rand(batch_size, 1).expand(batch_size, int(n_elements / batch_size)).contiguous()
    alpha = alpha.view(batch_size, colors, image_width, image_height).to(device)

    fake_data = fake_data.view(batch_size, colors, image_width, image_height)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    critic_interpolates = critic(interpolates)

    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty
    return gradient_penalty

def random_noise(batch_size, state_size, device):
    return torch.randn(batch_size, state_size).to(device)

def generate_images(generator, noise):
    batch_size = noise.size()[0]
    side = generator.side
    with torch.no_grad():
    	noisev = noise
    samples = generator(noisev).view(batch_size, 3, side, side)
    samples = samples * 0.5 + 0.5
    return samples

def train(generator, critic, dataloader, device, options):
    fixed_noise = random_noise(options.batch_size, options.state_size, device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=options.generator_lr, betas=(0,0.9))
    optimizer_c = torch.optim.Adam(critic.parameters(), lr=options.critic_lr, betas=(0,0.9))
    generator_path = os.path.join(options.model_dir, 'generator.pt')
    critic_path = os.path.join(options.model_dir, 'critic.pt')
    dataiter = iter(dataloader)
    if log_statistics:
        writer = SummaryWriter()

    minus_one = torch.FloatTensor([-1]).to(device)
    best_distance = math.inf

    for epoch in range(options.epochs):
        print(f'*** Epoch: {epoch + 1}/{options.epochs}')

        # Train the generator
        start = timer()
        for p in critic.parameters():
            p.requires_grad_(False)  # freeze the critic

        generator_cost = None
        for i in range(options.generator_iterations):
            generator.zero_grad()
            noise = random_noise(options.batch_size, options.state_size, device)
            noise.requires_grad_(True)
            fake_data = generator(noise)
            generator_cost = critic(fake_data)
            generator_cost = generator_cost.mean()
            generator_cost.backward(minus_one)
            generator_cost = -generator_cost

            print(f'Generator iteration: {i}, Cost: {generator_cost}')
            optimizer_g.step()

        end = timer()
        print(f'  Generator elapsed time: {end - start}')

        # Train the critic
        start = timer()

        for p in critic.parameters():
            p.requires_grad_(True) # unfreeze the critic

        for i in range(options.critic_iterations):
            critic.zero_grad()
            # Generate fake data and load real data
            noise = random_noise(options.batch_size, options.state_size, device)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training C
            fake_data = generator(noisev).detach()
            batch = next(dataiter, None)
            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()
            batch = batch[0] #batch[1] contains labels
            real_data = batch.to(device)

            critic_real = critic(real_data).mean()
            critic_fake = critic(fake_data).mean()
            penalty = gradient_penalty(critic, real_data, fake_data, options.gradient_penalty, device)
            critic_cost = critic_fake - critic_real + penalty
            critic_cost.backward()

            wasserstein_distance = (critic_real - critic_fake).item()
            print(f'Critic iteration: {i}, Wasserstein distance: {wasserstein_distance}')
            optimizer_c.step()

        end = timer()
        print(f'  Critic elapsed time: {end - start}')

        # Save models
        if wasserstein_distance < best_distance:
            best_distance = wasserstein_distance
            torch.save(generator, generator_path)
            torch.save(critic, critic_path)

        # Generate images
        if epoch % options.sample_every == 0:
            images = generate_images(generator, fixed_noise)
            torchvision.utils.save_image(images, os.path.join(options.output_dir, f'samples_{epoch}.png'), nrow=8, padding=2)

        # Log statistics
        if log_statistics:
            writer.add_scalar('data/generator_cost', generator_cost, epoch)
            writer.add_scalar('data/critic_cost', critic_cost, epoch)
            writer.add_scalar('data/gradient_penalty', penalty, epoch)
            writer.add_scalar('data/W1', critic_real - critic_fake, epoch)
            #writer.add_scalar('data/d_conv_weight_mean', [i for i in critic.children()][0].conv.weight.data.clone().mean(), epoch)
            #writer.add_scalar('data/d_linear_weight_mean', [i for i in critic.children()][-1].weight.data.clone().mean(), epoch)

            if epoch % options.sample_every == 0:
                params = critic.named_parameters()
                for name, param in params:
                    writer.add_histogram('C.' + name, param.clone().data.cpu().numpy(), epoch)
                body_model = [i for i in critic.children()][0]
                conv1 = body_model.conv.weight.data.clone().cpu()
                tensors = torchvision.utils.make_grid(conv1, nrow=8,padding=1)
                writer.add_image('C/conv1', tensors, epoch)

                grid_images = torchvision.utils.make_grid(images, nrow=8, padding=2)
                writer.add_image('images', grid_images, epoch)

def parse_options():
    parser = argparse.ArgumentParser(description='Wasserstein GAN')
    parser.add_argument('--data-dir', required=True, help='directory with the images')
    parser.add_argument('--output-dir', required=True, help='directory where to store the generated images')
    parser.add_argument('--model-dir', required=True, help='directory where to store the models')
    parser.add_argument('--image-class', default='bedroom', help='class to train on, only for LSUN')
    parser.add_argument('--dataset', choices=['raw', 'lsun'], default='raw', help='format of the dataset')
    parser.add_argument('--restore', action='store_true', help='restart training from the saved models')
    parser.add_argument('--image-size', type=int, default=64, help='image dimension')
    parser.add_argument('--state-size', type=int, default=128, help='state size')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--generator-iterations', type=int, default=1, help='number of iterations for the generator')
    parser.add_argument('--critic-iterations', type=int, default=5, help='number of iterations for the critic')
    parser.add_argument('--sample-every', type=int, default=10, help='how often to sample images')
    parser.add_argument('--gradient-penalty', type=float, default=10, help='gradient penalty')
    parser.add_argument('--generator-lr', type=float, default=1e-4, help='learning rate for the generator')
    parser.add_argument('--critic-lr', type=float, default=1e-4, help='learning rate for the critic')

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
        generator = Generator(options.image_size, options.state_size)
        critic = Critic(options.image_size)

        generator.apply(init_weights)
        critic.apply(init_weights)
    generator = generator.to(device)
    critic = critic.to(device)

    transform = transforms.Compose([
        transforms.Resize((options.image_size, options.image_size)),
        transforms.CenterCrop(options.image_size), #redundant?
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    if options.dataset == 'lsun':
        training_class = options.image_class + '_train'
        dataset =  datasets.LSUN(options.data_dir, classes=[training_class], transform=transform)
    else:
        dataset = datasets.ImageFolder(root=options.data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=options.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

    train(generator, critic, dataloader, device, options)

if __name__ == '__main__':
    run()