"""Report Metal accessibility in THIS execution context, without private identifiers."""
import json
import platform
import subprocess
import torch

result = dict(platform=platform.platform(), torch=torch.__version__, mps_built=torch.backends.mps.is_built(),
              mps_available=torch.backends.mps.is_available())
for key, name in [('hw.ncpu', 'cpu_cores'), ('hw.memsize', 'physical_memory_bytes')]:
    query = subprocess.run(['sysctl', '-n', key], capture_output=True, text=True)
    result[name] = int(query.stdout.strip()) if query.returncode == 0 else None
if result['mps_available']:
    value = torch.ones((16, 16), device='mps')
    result['metal_operation_correct'] = bool(torch.equal((value @ value).cpu(), torch.full((16, 16), 16.)))
    torch.mps.synchronize()
print(json.dumps(result, indent=2))
