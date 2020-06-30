# adapted from https://github.com/kmualim/CGAN-Pytorch/blob/master/cgan.py
import numpy as np
import os
import argparse
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import helper
from models import Discriminator, Generator, init_weights

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--vector_size', type=int, default=10, help='image size input')
parser.add_argument('--epoch', type=int, default=200, help='number of epoch')
parser.add_argument('--lr_rate', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta', type=float, default=0.5, help='beta for adam optimizer')
parser.add_argument('--beta1', type=float, default=0.999, help='beta1 for adam optimizer')
parser.add_argument('--output', default='./out', help='folder to output images and model checkpoints')
parser.add_argument('--random_seed', type=int, help='seed')

opt = parser.parse_args()

is_cuda = True if torch.cuda.is_available() else False
os.makedirs(opt.output, exist_ok=True)

if opt.random_seed is None:
    opt.random_seed = random.randint(1, 10000)
random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

dataset = helper.DisVectorData('./GAN-data-10.xlsx')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)


# Building generator
generator = Generator(opt.vector_size)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr_rate, betas=(opt.beta, opt.beta1))

# Building discriminator
discriminator = Discriminator(opt.vector_size)
discriminator.apply(init_weights)
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_rate, betas=(opt.beta, opt.beta1))

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
        #print(real_vector.shape, real_area.shape) [batch_size, 10] and [batch_size]
        real_vector = Variable(real_vector.type(FT))
        real_area = Variable(real_area.type(FT)).unsqueeze(-1)
        #print(real_vector.shape, real_area.shape) [batch_size, 10] and [batch_size, 1]

        # creating real and fake tensors of labels
        real_label = Variable(FT(batch_size, 1).fill_(1.0))
        fake_label = Variable(FT(batch_size, 1).fill_(0.0))

        #### TRAINING GENERATOR ####
        # initializing gradient
        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()

        # Feeding generator noise and labels
        noise = Variable(FT(np.random.normal(0, 1, (batch_size, 1))))
        random_area = Variable(FT(np.random.uniform(0, 1, (batch_size, 1))))
        generated_vector = generator(noise, random_area)
        #print(generated_vector.shape) [batch_size, 10]
        validity_gen = discriminator(generated_vector, random_area)

        # Generative loss function
        g_loss = bce_loss(validity_gen, real_label)

        # Gradients
        g_loss.backward(retain_graph=True)
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

    print("[Epoch: {}/{}]" "[D loss: {}]" "[G loss: {}]".format(epoch + 1, opt.epoch, d_loss.item(), g_loss.item()))

    # checkpoints
    torch.save(generator.state_dict(), '{}/generator_epoch_{}.pth'.format(opt.output, epoch))
    torch.save(discriminator.state_dict(), '{}/generator_epoch_{}.pth'.format(opt.output, epoch))
