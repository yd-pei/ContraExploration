import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from contra_rl.cli import validate_checkpoint
from contra_rl.env import ENV_VERSION
from contra_rl.experiments import build_plan


def test_comparison_changes_only_reward_and_output_within_each_seed():
    config = json.loads(Path('experiments/reward-comparison.json').read_text())
    plan = build_plan(config)
    assert len(plan) == 9
    def drop(options, keys):
        result = list(options)
        for key in keys:
            i = result.index(key)
            del result[i:i+2]
        return result
    for i in range(0, 9, 3):
        train = [drop(run['train'], ['--reward-mode', '--output']) for run in plan[i:i+3]]
        evaluate = [drop(run['evaluate'], ['--reward-mode', '--checkpoint', '--report']) for run in plan[i:i+3]]
        assert train[0] == train[1] == train[2]
        assert evaluate[0] == evaluate[1] == evaluate[2]
    assert len({run['output'] for run in plan}) == 9


def test_checkpoints_require_explicit_new_semantics_and_same_task():
    args = SimpleNamespace(reward_mode='game_score', all_lives=False, game='Contra-Nes')
    snapshot = dict(env_version=ENV_VERSION, config=vars(args).copy(), actions=[()],
                    model_spec=dict(atoms=51, vmin=-10., vmax=10000.))
    validate_checkpoint(snapshot, args, [()])
    for field, value, message in [('env_version', 'contra-level1-camera-v1', 'Legacy')]:
        changed = copy.deepcopy(snapshot)
        changed[field] = value
        with pytest.raises(ValueError, match=message):
            validate_checkpoint(changed, args, [()])
    for key, value, message in [('reward_mode', 'native', 'reward mode'),
                                ('reward_mode', 'score_clipped', 'reward mode'),
                                ('all_lives', True, 'episode rule')]:
        changed = copy.deepcopy(snapshot)
        changed['config'][key] = value
        with pytest.raises(ValueError, match=message):
            validate_checkpoint(changed, args, [()])
