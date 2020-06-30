import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, vector_size):
        super(Generator, self).__init__()
        self.depth = 32

        def init(input, output, normalize=True):
            layers = [nn.Linear(input, output)]
            if normalize:
                layers.append(nn.BatchNorm1d(output, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.generator = nn.Sequential(

            *init(2, self.depth),
            *init(self.depth, self.depth * 2),
            *init(self.depth * 2, self.depth * 4),
            *init(self.depth * 4, self.depth * 4),
            nn.Linear(self.depth * 4, vector_size),
            nn.Sigmoid()

        )

    def forward(self, noise, c):
        # noise is a random variable with normal distribution N(0, 1)
        # c is the conditional input
        gen_input = torch.cat((noise, c), -1)
        output = self.generator(gen_input)
        return output


class Discriminator(nn.Module):
    def __init__(self, vector_size):
        super(Discriminator, self).__init__()
        self.dropout = 0.4
        self.depth = 512

        def init(input, output, normalize=True):
            layers = [nn.Linear(input, output)]
            if normalize:
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.discriminator = nn.Sequential(
            *init(vector_size+1, self.depth, normalize=False),
            *init(self.depth, self.depth),
            *init(self.depth, self.depth),
            nn.Linear(self.depth, 1),
            nn.Sigmoid()
        )

    def forward(self, vector, c):
        input = torch.cat((vector, c), -1)
        validity = self.discriminator(input)
        return validity
    # weight initialization


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)