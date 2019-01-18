import argparse
import time
import json

import numpy as np
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, x_dim):
        super(Encoder, self).__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.tanh = nn.Tanh()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        #x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.tanh(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, x_dim):
        super(Decoder, self).__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, x_dim)
        # setup the non-linearities
        self.tanh = nn.Tanh()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.tanh(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x x_dim
        loc_img = torch.sigmoid(self.fc21(hidden))
        return loc_img


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim, hidden_dim, x_dim, vectorize, use_cuda=False):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim, x_dim)
        self.decoder = Decoder(z_dim, hidden_dim, x_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.vectorize = vectorize

    # define the model p(x|z)p(z)
    def model(self, x):
        assert x.shape[1] == self.x_dim
        pyro.module("decoder", self.decoder)
        if self.vectorize:
            with pyro.plate("data", x.shape[0]):
                z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
                z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
                loc_img = self.decoder.forward(z)
                pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, self.x_dim))
                return loc_img
        else:
            z_loc = x.new_zeros(self.z_dim)
            z_scale = x.new_ones(self.z_dim)
            for i in pyro.plate('data', x.shape[0]):
                z = pyro.sample("latent%d" % i, dist.Normal(z_loc, z_scale).to_event(1))
                loc_img = self.decoder.forward(z)
                pyro.sample("obs%d" % i, dist.Bernoulli(loc_img).to_event(1), obs=x[i])

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        pyro.module("encoder", self.encoder)
        if self.vectorize:
            with pyro.plate("data", x.shape[0]):
                z_loc, z_scale = self.encoder.forward(x)
                pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
        else:
            for i in pyro.plate('data', x.shape[0]):
                z_loc, z_scale = self.encoder(x[i])
                pyro.sample("latent%d" % i, dist.Normal(z_loc, z_scale).to_event(1))



def dummy_data(N, x_dim):
    return dist.Bernoulli(torch.tensor(0.1).expand(N, x_dim)).sample()

def main(args):
    # clear param store
    pyro.clear_param_store()

    images = dummy_data(args.N, args.x_dim)
    if args.cuda:
        images = images.cuda()

    t0 = time.time()

    vae = VAE(args.z_dim, args.h_dim, args.x_dim, args.vectorize, use_cuda=args.cuda)
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, Adam({"lr": args.step_size}), loss=elbo)

    history = []
    for i in range(args.num_steps):
        loss = svi.step(images)
        history.append(loss)
        print("%d: %f" % (i, loss))

    elapsed = time.time() - t0
    num_params = int(sum([np.product(param.shape) for (name, param) in pyro.get_param_store().items()]))

    # write out results in the same format as written by the webppl variant.
    result = dict(
        condition=dict(
            N=args.N,
            xDim=args.x_dim,
            hDim=args.h_dim,
            zDim=args.z_dim,
            batchSize=args.batch_size,
            numSteps=args.num_steps,
            stepSize=args.step_size,
            adBackend='pyro-vectorized' if args.vectorize else 'pyro-not-vectorized'
        ),
        elapsed=elapsed * 1e3,
        numParams=num_params,
        history=history
    )

    fn = int(time.time() * 1000)
    with open('./benchmark/results/%s.json' % fn, 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    assert pyro.__version__.startswith('0.3.0')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--num-steps', default=10, type=int)
    parser.add_argument('--batch-size', default=10, type=int)
    parser.add_argument('--z-dim', default=10, type=int)
    parser.add_argument('--h-dim', default=10, type=int)
    parser.add_argument('--x-dim', default=10, type=int)
    parser.add_argument('-N',      default=10, type=int)
    parser.add_argument('--step-size', default=1.0e-3, type=float)
    parser.add_argument('--vectorize', action='store_true', default=False)
    # ------------------------------------------------------------
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    args = parser.parse_args()

    model = main(args)
