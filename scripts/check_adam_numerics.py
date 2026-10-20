"""Compare Metal loop/fused FP32 Adam with a float64 formula reference."""
import json
from pathlib import Path
import torch

torch.manual_seed(12)
initial = torch.randn(19, 13, device='mps')
a, b = torch.nn.Parameter(initial.clone()), torch.nn.Parameter(initial.clone())
loop = torch.optim.Adam([a], lr=.0000625, eps=1.5e-4, foreach=False)
fused = torch.optim.Adam([b], lr=.0000625, eps=1.5e-4, fused=True)
reference = initial.cpu().double()
m, v = torch.zeros_like(reference), torch.zeros_like(reference)
for step, scale in enumerate([1e-5, 1., 10.], 1):
    gradient = torch.randn_like(initial) * scale
    g = gradient.cpu().double()
    m = .9 * m + .1 * g
    v = .999 * v + .001 * g.square()
    reference -= .0000625 * (m / (1 - .9 ** step)) / ((v / (1 - .999 ** step)).sqrt() + 1.5e-4)
    a.grad, b.grad = gradient.clone(), gradient.clone()
    loop.step()
    fused.step()
torch.mps.synchronize()
result = {}
for name, p, opt in [('loop', a, loop), ('fused', b, fused)]:
    result[name] = dict(parameter_max_abs_error_vs_float64=float((p.detach().cpu().double() - reference).abs().max()))
    for key, expected in [('exp_avg', m), ('exp_avg_sq', v)]:
        actual = opt.state[p][key].cpu().double()
        result[name][key + '_max_relative_error_vs_float64'] = float(((actual - expected).abs() / expected.abs().clamp_min(1e-12)).max())
result['parameter_max_difference_between_backends'] = float((a.detach()-b.detach()).abs().max())
result['float32_one_minus_beta2'] = float(torch.tensor(1.,dtype=torch.float32)-torch.tensor(.999,dtype=torch.float32))
result['exact_one_minus_beta2'] = .001
Path('reports/performance/adam-numerics.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
