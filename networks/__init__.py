# from .discriminators import Discriminator, Discriminator1D, Discriminator3
# from .gan_networks import Generator2, Generator, GeneratorBis
from .ticks_networks import Generator, Discriminator
from .criterion import generator_loss, discriminator_loss, \
    wasserstein_generator_loss, wasserstein_discriminator_loss
from .stft_networks import STFTGenerator, STFTDiscriminator
