import math
import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    """Factorised Gaussian noise; eval() uses learned means."""
    def __init__(self, n_in, n_out, sigma=0.5):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(n_out, n_in))
        self.weight_sigma = nn.Parameter(torch.full((n_out, n_in), sigma / math.sqrt(n_in)))
        self.bias_mu = nn.Parameter(torch.empty(n_out))
        self.bias_sigma = nn.Parameter(torch.full((n_out,), sigma / math.sqrt(n_out)))
        self.register_buffer('weight_eps', torch.empty(n_out, n_in))
        self.register_buffer('bias_eps', torch.empty(n_out))
        self.register_buffer('input_eps', torch.empty(n_in), persistent=False)
        self.register_buffer('output_eps', torch.empty(n_out), persistent=False)
        self._factor_only = False
        bound = 1 / math.sqrt(n_in)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        self.reset_noise()

    def reset_noise(self, factor_only=False):
        def noise(size):
            x = torch.randn(size, device=self.weight_mu.device)
            return x.sign() * x.abs().sqrt()
        a, b = noise(self.weight_mu.shape[1]), noise(self.weight_mu.shape[0])
        self.input_eps = a
        self.output_eps = b
        self._factor_only = factor_only
        if not factor_only:
            self.weight_eps.copy_(b.outer(a))
        self.bias_eps.copy_(b)

    def forward(self, x):
        if self.training and self._factor_only:
            # Algebraically identical factorised noise, without materialising a
            # multi-million-element noisy weight for a single actor observation.
            mean = F.linear(x, self.weight_mu, self.bias_mu)
            noise = F.linear(x * self.input_eps, self.weight_sigma, self.bias_sigma)
            return mean + noise * self.output_eps
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_eps,
                            self.bias_mu + self.bias_sigma * self.bias_eps)
        return F.linear(x, self.weight_mu, self.bias_mu)


class Rainbow(nn.Module):
    def __init__(self, actions, atoms=51, vmin=-10., vmax=10.):
        super().__init__()
        self.actions, self.atoms = actions, atoms
        self.register_buffer('support', torch.linspace(vmin, vmax, atoms))
        self.encoder = nn.Sequential(nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3), nn.ReLU(), nn.Flatten())
        self.value = nn.Sequential(NoisyLinear(3136, 512), nn.ReLU(), NoisyLinear(512, atoms))
        self.advantage = nn.Sequential(NoisyLinear(3136, 512), nn.ReLU(), NoisyLinear(512, actions * atoms))

    def logits(self, x):
        x = self.encoder(x.float() / 255.)
        v = self.value(x).view(-1, 1, self.atoms)
        a = self.advantage(x).view(-1, self.actions, self.atoms)
        return v + a - a.mean(1, keepdim=True)

    def forward(self, x):
        return self.logits(x).softmax(-1)

    def q(self, x):
        return (self(x) * self.support).sum(-1)

    def reset_noise(self, factor_only=False):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise(factor_only=factor_only)


def project_distribution(prob, reward, discount, support):
    """C51 projection, preserving mass for targets exactly on support bins."""
    atoms = len(support)
    target = (reward[:, None] + discount[:, None] * support).clamp(support[0], support[-1])
    b = ((target - support[0]) / (support[-1] - support[0]) * (atoms - 1)).clamp(0, atoms - 1)
    lo, hi = b.floor().long(), b.ceil().long()
    offset = torch.arange(len(prob), device=prob.device)[:, None] * atoms
    result = torch.zeros_like(prob).flatten()
    result.index_add_(0, (lo + offset).flatten(), (prob * (hi - b + (hi == lo))).flatten())
    result.index_add_(0, (hi + offset).flatten(), (prob * (b - lo)).flatten())
    return result.view_as(prob)
