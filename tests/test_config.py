import json
from pathlib import Path
import pytest
from contra_rl.cli import main, parse_args

ROOT = Path(__file__).resolve().parents[1]


def write_config(tmp_path, **changes):
    config = dict(reward_mode='game_score', output='runs/test', steps=100, all_lives=True,
                  device='cpu', standard_actor=True, learning_rate=.0002)
    config.update(changes)
    path = tmp_path / 'train.json'
    path.write_text(json.dumps(config))
    return path


def test_independent_configs_have_explicit_modes_and_distinct_outputs():
    outputs = []
    for mode in ['score_clipped', 'game_score', 'progress']:
        args = parse_args(['train', '--config', str(ROOT / 'configs' / f'{mode}.json'), '--dry-run'])
        assert args.reward_mode == mode and args.steps == 1000000
        assert args.device == 'mps' and args.adam_backend == 'fused' and args.replay_storage == 'raw'
        outputs.append(args.output)
    assert len(set(outputs)) == 3


def test_cli_overrides_config_including_false_booleans(tmp_path):
    path = write_config(tmp_path)
    args = parse_args(['train', '--config', str(path), '--steps', '32', '--seed', '7',
                       '--no-all-lives', '--no-standard-actor', '--output', 'runs/override'])
    assert args.steps == 32 and args.seed == 7 and not args.all_lives and not args.standard_actor
    assert args.output == 'runs/override' and args.learning_rate == .0002
    assert args.batch == 32  # Absent options still use the existing CLI defaults.
    assert args.source_config_sha256 and args.config == str(path.resolve())


@pytest.mark.parametrize('changes,message', [
    ({'batch_size':32}, 'Unknown training keys'),
    ({'reward_mode':'native'}, 'Invalid reward_mode'),
    ({'batch':True}, 'Invalid type/value'),
    ({'threads':'4'}, 'Invalid type/value'),
    ({'device':'unknown'}, 'Invalid device'),
    ({'output':''}, 'nonempty'),
    ({'learning_rate':-.01}, 'learning_rate'),
    ({'gamma':2}, 'gamma'),
    ({'capacity':8,'batch':32}, 'capacity'),
])
def test_invalid_configs_fail_before_training(tmp_path, capsys, changes, message):
    path = write_config(tmp_path, **changes)
    with pytest.raises(SystemExit) as e:
        parse_args(['train', '--config', str(path), '--dry-run'])
    assert e.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('source', [
    '[]',
    '{"reward_mode":"game_score","output":"runs/test","steps":1,"steps":2}',
    '{"reward_mode":"game_score","output":"runs/test","gamma":NaN}',
    '{"output":"runs/test"}',
])
def test_malformed_or_ambiguous_json_rejected(tmp_path, source):
    path=tmp_path/'bad.json'
    path.write_text(source)
    with pytest.raises(SystemExit):
        parse_args(['train','--config',str(path),'--dry-run'])


def test_dry_run_does_not_create_output_or_call_training(tmp_path, monkeypatch, capsys):
    output=tmp_path/'never-created'
    path=write_config(tmp_path,output=str(output))
    def fail(*args):
        raise AssertionError('Training must not run during preview')
    monkeypatch.setattr('contra_rl.cli.train', fail)
    main(['train','--config',str(path),'--dry-run'])
    assert json.loads(capsys.readouterr().out)['output'] == str(output)
    assert not output.exists()


def test_empty_cli_output_override_rejected(tmp_path):
    path = write_config(tmp_path)
    with pytest.raises(SystemExit):
        parse_args(['train', '--config', str(path), '--output', '', '--dry-run'])
