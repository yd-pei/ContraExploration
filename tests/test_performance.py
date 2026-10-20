import copy
import numpy as np
import pytest
import torch
from contra_rl.model import NoisyLinear
from contra_rl.replay import PrioritizedReplay


@pytest.mark.parametrize('batch', [1, 32])
def test_factorised_actor_same_noise_outputs_and_gradients(batch):
    torch.manual_seed(14)
    original = NoisyLinear(17, 11)
    factorised = copy.deepcopy(original)
    factorised._factor_only = True
    x = torch.randn(batch, 17)
    expected, actual = original(x), factorised(x)
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for old, new in zip(original.parameters(), factorised.parameters()):
        assert torch.allclose(old.grad, new.grad, atol=1e-6, rtol=1e-5)
    # No new checkpoint keys; existing learned weights remain loadable.
    assert set(original.state_dict()) == {'weight_mu', 'weight_sigma', 'bias_mu', 'bias_sigma', 'weight_eps', 'bias_eps'}


def test_raw_replay_preserves_samples_priorities_wrap_and_input_ownership():
    old = PrioritizedReplay(13, seed=7, storage='zlib')
    new = PrioritizedReplay(13, seed=7, storage='raw')
    for t in range(31):
        obs = np.full((4, 84, 84), t, np.uint8)
        row = (obs, t % 5, t / 3, obs + 1, .99)
        old.add(row)
        new.add(row)
        obs.fill(0)  # Replay must own its contents, not alias the collector.
        if t > 4:
            a, b = old.sample(4, .6), new.sample(4, .6)
            for expected, actual in zip(a, b):
                assert np.array_equal(expected, actual)
            p = np.arange(1, 5, dtype=float)
            old.update(a[-1], p)
            new.update(b[-1], p)
    assert np.array_equal(old.tree, new.tree)
    assert np.array_equal(old.minimum, new.minimum)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS requires normal Metal access')
def test_mps_fused_adam_matches_reference_updates():
    torch.manual_seed(12)
    initial = torch.randn(19, 13, device='mps')
    a, b = torch.nn.Parameter(initial.clone()), torch.nn.Parameter(initial.clone())
    original = torch.optim.Adam([a], lr=.0000625, eps=1.5e-4, foreach=False)
    fused = torch.optim.Adam([b], lr=.0000625, eps=1.5e-4, fused=True)
    for scale in [1e-5, 1., 10.]:
        gradient = torch.randn_like(initial) * scale
        a.grad, b.grad = gradient.clone(), gradient.clone()
        original.step()
        fused.step()
    torch.mps.synchronize()
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)
    for key in ['exp_avg', 'exp_avg_sq']:
        # Metal fused Adam forms (1-beta2) in FP32: .0009999871 vs .001.
        # A float64 reference probe measured 13 ppm in v, and <2.2e-7 in weights.
        tolerance = 2e-5 if key == 'exp_avg_sq' else 1e-5
        assert torch.allclose(original.state[a][key], fused.state[b][key], atol=1e-6, rtol=tolerance)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS requires normal Metal access')
def test_mps_factorised_actor_matches_materialised_noise():
    original = NoisyLinear(3136, 512).to('mps')
    original.reset_noise()
    factorised = copy.deepcopy(original)
    factorised._factor_only = True
    x = torch.randn(1, 3136, device='mps')
    with torch.no_grad():
        assert torch.allclose(original(x), factorised(x), atol=2e-6, rtol=2e-5)
