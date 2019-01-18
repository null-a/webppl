import argparse
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# NOTE: I've not checked that this actually works as expected. I'm
# only interested in benchmarking the against e.g. Pyro, so as long as
# this preform roughly the same amount of computation as a correct
# implementation, which I think it does, then I'm happy.

def dummy_data(N, x_dim):
    return torch.bernoulli(torch.tensor(0.1).expand(N, x_dim))

class Model(nn.Module):
    def __init__(self, z_dim, hidden_dim, x_dim):
        super(Model, self).__init__()
        # encoder
        self.enc_fc1 = nn.Linear(x_dim, hidden_dim)
        self.enc_fc21 = nn.Linear(hidden_dim, z_dim)
        self.enc_fc22 = nn.Linear(hidden_dim, z_dim)
        # decoder
        self.dec_fc1 = nn.Linear(z_dim, hidden_dim)
        self.dec_fc21 = nn.Linear(hidden_dim, x_dim)

        self.tanh = nn.Tanh()

    def forward(self, x):
        enc_hidden = self.tanh(self.enc_fc1(x))
        z_mu = self.enc_fc21(enc_hidden)
        z_log_sigma = self.enc_fc22(enc_hidden)
        z_sigma = torch.exp(z_log_sigma)
        eps = torch.normal(torch.tensor(0.0).expand(z_mu.shape),
                           torch.tensor(1.0).expand(z_mu.shape))
        z = (eps * z_sigma) + z_mu
        dec_hidden = self.tanh(self.dec_fc1(z))
        ps = torch.sigmoid(self.dec_fc21(dec_hidden))
        return ps, z, z_mu, z_sigma, z_log_sigma

log_root_2_pi = np.log(np.sqrt(2 * np.pi))

# ELBO (This doesn't use the analytic KL.)
def loss(x, ps, z, z_mu, z_sigma, z_log_sigma):
    logqz = torch.sum(-0.5 * torch.pow((z - z_mu) / z_sigma, 2) - z_log_sigma - log_root_2_pi, 1)
    logpz = torch.sum(-0.5 * torch.pow(z, 2) - log_root_2_pi, 1)
    logpx = torch.sum(x * torch.log(ps) + (1 - x) * torch.log(1 - ps), 1)
    L = torch.mean(logpz + logpx - logqz)
    return -L

def main(args):
    images = dummy_data(args.N, args.x_dim)

    t0 = time.time()

    model = Model(args.z_dim, args.h_dim, args.x_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.step_size)

    history = []
    for i in range(args.num_steps):
        optimizer.zero_grad()
        output = model(images)
        objective = loss(images, *output)
        objective.backward()
        optimizer.step()
        history.append(objective.item())
        print('%d: %f' % (i , objective))

    elapsed = time.time() - t0
    #print('elapsed: %f' % elapsed)
    num_params = int(sum(np.product(param.shape) for param in model.parameters()))
    #print('num_params: %d' % num_params)

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
            adBackend='pytorch'
        ),
        elapsed=elapsed * 1e3,
        numParams=num_params,
        history=history
    )

    fn = int(time.time() * 1000)
    with open('./benchmark/results/%s.json' % fn, 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--num-steps', default=10, type=int)
    parser.add_argument('--batch-size', default=10, type=int)
    parser.add_argument('--z-dim', default=10, type=int)
    parser.add_argument('--h-dim', default=10, type=int)
    parser.add_argument('--x-dim', default=10, type=int)
    parser.add_argument('-N',      default=10, type=int)
    parser.add_argument('--step-size', default=1.0e-3, type=float)
    args = parser.parse_args()

    main(args)
