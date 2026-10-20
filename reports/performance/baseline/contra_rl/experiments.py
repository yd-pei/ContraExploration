"""Plan or explicitly run a controlled three-reward comparison."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import numpy as np
from .env import ENV_VERSION, REWARD_MODES, REWARD_SPECS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_KEYS = {'game', 'steps', 'max_steps', 'all_lives', 'capacity', 'batch', 'warmup',
              'train_every', 'target_every', 'checkpoint_every', 'n_step', 'gamma',
              'atoms', 'vmin', 'vmax', 'log_every', 'device', 'threads'}


def arguments(options):
    result = []
    for key, value in options.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                result.append(flag)
        else:
            result.extend([flag, str(value)])
    return result


def build_plan(config):
    if set(config['reward_modes']) != set(REWARD_MODES) or len(config['reward_modes']) != 3:
        raise ValueError('Comparison must contain each of the three reward modes exactly once')
    if set(config['train']) != TRAIN_KEYS:
        raise ValueError(f'Shared training settings must explicitly specify {sorted(TRAIN_KEYS)}')
    if config['train']['game'] != 'Contra-Nes':
        raise ValueError('Three-score comparison targets Contra-Nes only')
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Seeds must be nonempty and unique')
    plan = []
    for seed in config['seeds']:
        for mode in config['reward_modes']:
            output = Path(config['output_root']) / f'seed-{seed}' / mode
            train = dict(config['train'], seed=seed, reward_mode=mode, output=str(output))
            evaluation = dict(game=train['game'], seed=seed, reward_mode=mode, all_lives=train['all_lives'],
                              device=train['device'], threads=train['threads'], **config['evaluation'],
                              checkpoint=str(output / 'latest.pt'), report=str(output / 'evaluation.json'))
            plan.append(dict(seed=seed, reward_mode=mode, output=str(output),
                             train=['train'] + arguments(train), evaluate=['eval'] + arguments(evaluation)))
    return plan


def summarize(plan):
    rows = []
    for run in plan:
        path = Path(run['output'])
        episodes = json.loads((path / 'evaluation.json').read_text())
        training = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]
        final = next(row for row in reversed(training) if row['type'] == 'summary')
        def mean(field):
            return float(np.mean([e['final'][field] for e in episodes]))
        rows.append(dict(seed=run['seed'], reward_mode=run['reward_mode'],
                         success_rate=mean('episode_success'), mean_score_gain=mean('episode_game_score_gain'),
                         mean_final_game_score=mean('game_score'), mean_progress_max=mean('progress_max'),
                         mean_deaths=mean('episode_deaths'), mean_extra_lives=mean('episode_extra_lives'),
                         mean_decision_steps=mean('decision_steps'), mean_emulator_frames=mean('episode_emulator_frames'),
                         train_updates=final['updates'], c51_target_outside_mass=final['c51_mean_target_outside_mass'],
                         evaluation_episodes=len(episodes)))
    return rows


def main():
    parser = argparse.ArgumentParser(description='Same seeds/budget/task/hyperparameters across three rewards; dry-run by default')
    parser.add_argument('--config', default='experiments/reward-comparison.json')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--output-root')
    parser.add_argument('--run', action='store_true', help='Actually run all planned training and evaluation jobs sequentially')
    args = parser.parse_args()
    config = copy.deepcopy(json.loads(Path(args.config).read_text()))
    if args.steps is not None:
        if args.steps < 1:
            parser.error('--steps must be positive')
        config['train']['steps'] = args.steps
    if args.seeds is not None:
        config['seeds'] = args.seeds
    if args.output_root:
        config['output_root'] = args.output_root
    plan = build_plan(config)
    for run in plan:
        for key in ['train', 'evaluate']:
            print(shlex.join(['uv', 'run', 'contra-rl'] + run[key]), flush=True)
    if not args.run:
        print('Dry run only. Add --run to execute; no training has started.')
        return
    root = Path(config['output_root'])
    if any((Path(run['output']) / 'metrics.jsonl').exists() for run in plan) or (root / 'manifest.json').exists():
        raise ValueError('Comparison output already exists; choose a new --output-root')
    root.mkdir(parents=True, exist_ok=True)
    sources = ['contra_rl/env.py', 'contra_rl/model.py', 'contra_rl/agent.py',
               'contra_rl/replay.py', 'contra_rl/cli.py', 'contra_rl/experiments.py', 'uv.lock']
    manifest = dict(env_version=ENV_VERSION, config=config, reward_definitions=REWARD_SPECS, plan=plan,
                    source_sha256={p: hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest() for p in sources},
                    evaluation_note='Compare common task metrics, not reward returns. Fixed state/greedy policy may repeat identical episodes.')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for run in plan:
        subprocess.run([sys.executable, '-m', 'contra_rl.cli'] + run['train'], check=True)
        subprocess.run([sys.executable, '-m', 'contra_rl.cli'] + run['evaluate'], check=True)
    rows = summarize(plan)
    (root / 'comparison.json').write_text(json.dumps(rows, indent=2))
    with (root / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'comparison': str(root / 'comparison.csv'), 'runs': len(rows)}))


if __name__ == '__main__':
    main()
