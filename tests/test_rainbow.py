import numpy as np
import torch
from contra_rl.model import Rainbow, project_distribution
from contra_rl.replay import NStep, PrioritizedReplay
from contra_rl.agent import Agent

torch.set_num_threads(2)


def test_c51_mass_exact_bins_and_clamping():
    support = torch.linspace(-10, 10, 51)
    p = torch.ones(3, 51) / 51
    out = project_distribution(p, torch.tensor([0., 30., -30.]), torch.tensor([0., 0., 0.]), support)
    assert torch.allclose(out.sum(1), torch.ones(3))
    assert out[0, 25] == 1 and out[1, -1] == 1 and out[2, 0] == 1
    identity = project_distribution(p, torch.zeros(3), torch.ones(3), support)
    assert torch.allclose(identity, p, atol=1e-6)


def test_nstep_terminal_flush_and_time_limit_bootstrap():
    q = NStep(3, .5)
    obs = np.zeros((4, 84, 84), np.uint8)
    assert not q.add(obs, 0, 1, obs, False, False)
    rows = q.add(obs, 0, 2, obs, True, False)
    assert [r[2] for r in rows] == [2, 2]
    assert [r[4] for r in rows] == [0, 0]
    assert not q.queue
    q.add(obs, 0, 1, obs, False, False)
    rows = q.add(obs, 0, 2, obs, False, True)
    assert [r[4] for r in rows] == [.25, .5]
    assert not q.queue


def test_per_bias_weights_wraparound_and_duplicates():
    replay = PrioritizedReplay(4, alpha=1, seed=6)
    obs = np.zeros((4, 84, 84), np.uint8)
    for i in range(6):
        replay.add((obs + i, i, float(i), obs, .99))
    assert replay.size == 4
    replay.update([0, 1, 1, 2, 3], [1, 8, 20, 1, 1])
    assert replay.tree[replay.base + 1] > 20
    sample = replay.sample(1000, 1)
    assert sample[0].shape == (1000, 4, 84, 84)
    assert np.mean(np.array(sample[-1]) == 1) > .8
    assert np.all(sample[-2] > 0) and np.all(sample[-2] <= 1)


def test_noisy_dueling_distribution_and_learning():
    torch.manual_seed(1)
    agent = Agent(3)
    obs = np.zeros((4, 84, 84), np.uint8)
    x = torch.as_tensor(obs[None])
    agent.online.eval()
    first = agent.online(x)
    agent.online.reset_noise()
    assert torch.equal(first, agent.online(x))
    assert torch.allclose(first.sum(-1), torch.ones(1, 3))
    agent.online.train()
    a = agent.online(x)
    agent.online.reset_noise()
    assert not torch.equal(a, agent.online(x))
    replay = PrioritizedReplay(8)
    for _ in range(8):
        replay.add((obs, 0, 1., obs, 0.))
    before = agent.online.value[-1].weight_mu.detach().clone()
    target_before = agent.target.value[-1].weight_mu.detach().clone()
    stats = agent.learn(replay, 4, .4)
    assert np.isfinite(stats['loss'])
    assert not torch.equal(before, agent.online.value[-1].weight_mu)
    assert torch.equal(target_before, agent.target.value[-1].weight_mu)
    agent.sync()
    assert torch.equal(agent.online.value[-1].weight_mu, agent.target.value[-1].weight_mu)


def test_raw_score_c51_support_adaptation_reports_outside_mass_without_reward_clipping():
    obs = np.zeros((4, 84, 84), np.uint8)
    replay = PrioritizedReplay(4)
    for _ in range(4):
        replay.add((obs, 0, 100., obs, 0.))
    narrow = Agent(2, vmin=-10., vmax=10.)
    wide = Agent(2, vmin=-10., vmax=10000.)
    assert narrow.learn(replay, 4, .4)['c51_target_outside_mass'] == 1.
    assert wide.learn(replay, 4, .4)['c51_target_outside_mass'] == 0.
    assert np.all(replay.sample(4, .4)[2] == 100.)  # Raw replay rewards were never clipped.
