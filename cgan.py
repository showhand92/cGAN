import numpy as np
import os
import argparse
import torch
import random
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.autograd import Variable

import helper
from models import Discriminator, Generator, init_weights

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--vector_size', type=int, default=10, help='image size input')
parser.add_argument('--epoch', type=int, default=200, help='number of epoch')
parser.add_argument('--lr_rate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta', type=float, default=0.5, help='beta for adam optimizer')
parser.add_argument('--beta1', type=float, default=0.999, help='beta1 for adam optimizer')
parser.add_argument('--output', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--random_seed', type=int, help='seed')
max_area = 4.5

opt = parser.parse_args()

is_cuda = True if torch.cuda.is_available() else False
os.makedirs(opt.output, exist_ok=True)

if opt.randomseed is None:
    opt.random_seed = random.randint(1, 10000)
random.seed(opt.randomseed)
torch.manual_seed(opt.randomseed)

dataset = helper.load_data()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)


# Building generator
generator = Generator()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lrate, betas=(opt.beta, opt.beta1))

# Building discriminator
discriminator = Discriminator()
discriminator.apply(init_weights)
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lrate, betas=(opt.beta, opt.beta1))

# Loss functions
bce_loss = torch.nn.BCELoss()


LT = torch.LongTensor
FT = torch.FloatTensor

if is_cuda:
    generator.cuda()
    discriminator.cuda()
    bce_loss.cuda()
    LT = torch.cuda.LongTensor
    FT = torch.cuda.FloatTensor

# training
for epoch in range(opt.epoch):
    for i, (real_vector, real_area) in enumerate(dataloader):
        batch_size = real_vector.shape[0]

        # convert img, labels into proper form
        real_vector = Variable(real_vector.type(FT))
        real_area = Variable(real_area.type(LT))

        # creating real and fake tensors of labels
        real_label = Variable(FT(batch_size, 1).fill_(1.0))
        fake_label = Variable(FT(batch_size, 1).fill_(0.0))

        #### TRAINING GENERATOR ####
        # initializing gradient
        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()

        # Feeding generator noise and labels
        noise = Variable(FT(np.random.normal(0, 1, batch_size)))
        random_area = Variable(FT(np.random.uniform(0, max_area, batch_size)))
        generated_vector = generator(noise, random_area)
        validity_gen = discriminator(generated_vector, random_area)

        # Generative loss function
        g_loss = bce_loss(validity_gen, real_label)

        # Gradients
        g_loss.backward()
        gen_optimizer.step()


        #### TRAINING DISCRIMINTOR ####
        dis_optimizer.zero_grad()
        # Loss for real images and labels
        validity_real = discriminator(real_vector, real_area)
        d_real_loss = bce_loss(validity_real, real_label)

        # Loss for fake images and labels
        validity_fake = discriminator(generated_vector, random_area)
        d_fake_loss = bce_loss(validity_fake, fake_label)

        # Total discriminator loss
        d_loss = 0.5 * (d_fake_loss + d_real_loss)

        # calculates discriminator gradients
        d_loss.backward()
        dis_optimizer.step()

    print("[Epoch: %d/%d]" "[D loss: %f]" "[G loss: %f]" % (epoch + 1, opt.epoch, d_loss.item(), g_loss.item()))

    # checkpoints
    torch.save(generator.state_dict(), '%s/generator_epoch_%d.pth' % (opt.output, epoch))
    torch.save(discriminator.state_dict(), '%s/generator_epoch_%d.pth' % (opt.output, epoch))
