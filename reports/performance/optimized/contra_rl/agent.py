import numpy as np
import torch
from torch.nn import functional as F
from .model import Rainbow, project_distribution
from .performance import span


class Agent:
    def __init__(self, actions, device='cpu', lr=0.0000625, atoms=51, vmin=-10., vmax=10., fast_actor=True, adam_backend='loop'):
        self.device = torch.device(device)
        self.fast_actor = fast_actor
        self.online = Rainbow(actions, atoms, vmin, vmax).to(self.device)
        self.target = Rainbow(actions, atoms, vmin, vmax).to(self.device)
        self.sync()
        options = {'foreach': False} if adam_backend == 'loop' else {adam_backend: True}
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr, eps=1.5e-4, **options)

    def sync(self):
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)

    @torch.no_grad()
    def act(self, obs, evaluate=False):
        self.online.train(not evaluate)
        if not evaluate:
            self.online.reset_noise(factor_only=self.fast_actor)
        return int(self.online.q(torch.as_tensor(obs[None], device=self.device)).argmax(1).item())

    def learn(self, replay, batch, beta):
        with span('replay_sample'):
            obs, actions, rewards, next_obs, discounts, weights, indices = replay.sample(batch, beta)
        with span('tensor_transfer'):
            o, a, r, no, d, w = [torch.as_tensor(x, device=self.device) for x in
                                 (obs, actions, rewards, next_obs, discounts, weights)]
        with span('learn_forward_and_projection'):
            self.online.train()
            self.target.train()
            self.online.reset_noise()
            self.target.reset_noise()
            with torch.no_grad():
                # Double DQN: online selects, target evaluates the distribution.
                next_actions = self.online.q(no).argmax(1)
                prob = self.target(no)[torch.arange(batch, device=self.device), next_actions]
                support = self.online.support
                targets = r[:, None] + d[:, None] * support
                outside_mass = (prob * ((targets < support[0]) | (targets > support[-1]))).sum(1).mean()
                projected = project_distribution(prob, r, d, support)
            log_prob = F.log_softmax(self.online.logits(o), -1)[torch.arange(batch, device=self.device), a]
            per_sample = -(projected * log_prob).sum(1)
            loss = (per_sample * w).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite Rainbow loss')
        with span('backward_and_clip'):
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        with span('optimizer'):
            self.optimizer.step()
        # Original Rainbow prioritises distribution KL; entropy is constant w.r.t. prediction.
        with span('priority_update'):
            entropy = -(projected * projected.clamp_min(1e-8).log()).sum(1)
            priority = (per_sample.detach() - entropy).clamp_min(0).cpu().numpy()
            replay.update(indices, priority)
        return {'loss': float(loss.item()), 'grad_norm': float(grad.item()), 'beta': beta,
                'c51_target_outside_mass': float(outside_mass.item()),
                'c51_projected_edge_mass': float(projected[:, [0, -1]].sum(1).mean().item())}
