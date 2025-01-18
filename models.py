import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        """
        A residual block with two convolutional layers and skip connection.
        """
        super(ResidualBlock, self).__init__()
        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]
        
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        """
        Forward pass for the residual block.
        """
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        """
        Generator network with initial convolution, downsampling, residual blocks, and upsampling.
        """
        super(Generator, self).__init__()
        model=[nn.ReflectionPad2d(3),
               nn.Conv2d(input_nc, 64, 7),
               nn.InstanceNorm2d(64),
               nn.ReLU(inplace=True)]
        
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model +=[nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                     nn.InstanceNorm2d(out_features),
                     nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        #residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        out_features = in_features//2
        for _ in range(2):
            model += [ nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2
        self.model = nn.Sequential(*model)


    def forward(self, x):
        """
        Forward pass for the generator network.
        """
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        """
        Discriminator network with a series of convolutions and a classification layer.
        """
        super(Discriminator, self).__init__()
        model = [ nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]
        
        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]
        model +=[ nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]
        
        model += [nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        Forward pass for the discriminator network.
        """
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)