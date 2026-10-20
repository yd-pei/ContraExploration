import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import time
import sys
from .config import load_train_config
import numpy as np
import torch
from .agent import Agent
from .env import make_env, ENV_VERSION, REWARD_MODES, REWARD_SPECS
from .replay import NStep, PrioritizedReplay
from .performance import span, TrainingProfile


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def json_line(file, row):
    with span('logging'):
        file.write(json.dumps(row) + '\n')
        file.flush()


def save_checkpoint(path, agent, args, step, actions):
    temporary = path.with_suffix('.tmp')
    torch.save({'model': agent.online.state_dict(), 'optimizer': agent.optimizer.state_dict(),
                'step': step, 'config': vars(args), 'actions': actions, 'env_version': ENV_VERSION,
                'model_spec': {'atoms': args.atoms, 'vmin': args.vmin, 'vmax': args.vmax},
                'note': 'Model/optimizer snapshot; replay and emulator are not saved. No exact resume.'}, temporary)
    temporary.replace(path)


def doctor():
    import stable_retro as retro
    from importlib.metadata import version
    result = {'platform': platform.platform(), 'machine': platform.machine(),
              'versions': {x: version(x) for x in ['torch', 'stable-retro', 'gymnasium']},
              'mps_available': torch.backends.mps.is_available(), 'game': 'Contra-Nes'}
    root = Path(retro.data.get_file_path('Contra-Nes', 'rom.sha', retro.data.Integrations.ALL)).parent
    result.update(integration=str(root), expected_sha1_without_ines_header=(root / 'rom.sha').read_text().strip(),
                  states=[p.name for p in root.glob('*.state')])
    try:
        result['rom'] = retro.data.get_romfile_path('Contra-Nes', retro.data.Integrations.ALL)
        result['rom_valid'] = not retro.data.verify_hash('Contra-Nes', retro.data.Integrations.ALL)
    except FileNotFoundError:
        result.update(rom=None, rom_valid=False)
    print(json.dumps(result, indent=2))


def import_rom(path):
    import stable_retro as retro
    source = Path(path)
    data = source.read_bytes()
    if data[:4] != b'NES\x1a' or len(data) < 16:
        raise ValueError('Expected uncompressed iNES .nes file with a 16-byte header')
    if data[6] & 4:
        raise ValueError('Trainer-bearing ROM unsupported; use the exact integration ROM')
    sha = hashlib.sha1(data[16:]).hexdigest()
    root = Path(retro.data.get_file_path('Contra-Nes', 'rom.sha', retro.data.Integrations.ALL)).parent
    expected = (root / 'rom.sha').read_text().split()
    if sha not in expected:
        raise ValueError(f'Contra-Nes SHA-1 mismatch: got {sha}; expected {expected}')
    shutil.copyfile(source, root / 'rom.nes')
    if retro.data.verify_hash('Contra-Nes', retro.data.Integrations.ALL):
        raise RuntimeError('ROM post-copy validation failed')
    print(json.dumps({'imported': str(root / 'rom.nes'), 'sha1_without_header': sha}))


def train(args):
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.jsonl').exists():
        raise ValueError('Output already contains metrics; choose a new --output to preserve runs')
    env = make_env(args.game, args.max_steps, not args.all_lives, args.reward_mode)
    try:
        agent = Agent(env.action_space.n, args.device, atoms=args.atoms, vmin=args.vmin, vmax=args.vmax,
                      fast_actor=not args.standard_actor, adam_backend=args.adam_backend,
                      lr=args.learning_rate, adam_eps=args.adam_eps)
        replay = PrioritizedReplay(args.capacity, alpha=args.replay_alpha, seed=args.seed, storage=args.replay_storage)
        nstep = NStep(args.n_step, args.gamma)
        config = vars(args).copy()
        config.update(env_version=ENV_VERSION, reward=REWARD_SPECS[args.reward_mode],
                      reward_definitions=REWARD_SPECS, score_ram_point_multiplier=100,
                      actions=env.combos, frame_skip=4, initial_state='Level1',
                      preprocessing='max last 2 frames; grayscale 84x84; stack 4',
                      episode_rule='all_lives' if args.all_lives else 'first_life',
                      algorithm='Rainbow: Double/PER/Dueling/n-step/C51/NoisyNet')
        (output / 'config.json').write_text(json.dumps(config, indent=2))
        obs, _ = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        episode_return, episode_length, updates = 0., 0, 0
        run_totals = dict(game_score_gain=0, integration_scroll_return=0., progress_return=0.,
                          score_clipped_return=0., training_return=0., deaths=0, extra_lives=0,
                          successes=0, emulator_frames=0, episodes_completed=0)
        outside_sum = 0.
        last_info = None
        profiler = TrainingProfile(args.device, args.profile_sync) if args.profile else None
        if profiler:
            profiler.start()
        started = time.perf_counter()
        with (output / 'metrics.jsonl').open('w') as log:
            for step in range(1, args.steps + 1):
                with span('inference'):
                    action = env.action_space.sample() if step <= args.warmup else agent.act(obs)
                with span('env_step_total'):
                    next_obs, reward, terminated, truncated, info = env.step(action)
                with span('replay_add'):
                    for transition in nstep.add(obs, action, reward, next_obs, terminated, truncated):
                        replay.add(transition)
                last_info = info
                for key, field in [('game_score_gain', 'game_score_delta'),
                                   ('integration_scroll_return', 'integration_scroll_reward'),
                                   ('progress_return', 'progress_reward'), ('score_clipped_return', 'score_clipped_reward'),
                                   ('training_return', 'training_reward'), ('deaths', 'deaths'),
                                   ('extra_lives', 'extra_lives'), ('emulator_frames', 'frames_advanced')]:
                    run_totals[key] += info[field]
                if (step % args.log_every == 0 or terminated or truncated or
                    any(info[k] != 0 for k in ['game_score_delta', 'integration_scroll_reward', 'progress_reward', 'extra_lives'])):
                    json_line(log, dict(type='transition', step=step, action=int(action),
                                        terminated=bool(terminated), truncated=bool(truncated), metrics=info))
                obs = next_obs
                episode_return += reward
                episode_length += 1
                if step >= args.warmup and step % args.train_every == 0 and replay.size >= args.batch:
                    beta = min(1., 0.4 + 0.6 * step / args.steps)
                    with span('learn_total'):
                        stats = agent.learn(replay, args.batch, beta)
                    updates += 1
                    outside_sum += stats['c51_target_outside_mass']
                    if updates == 1 or updates % 100 == 0:
                        json_line(log, dict(type='update', step=step, updates=updates, **stats))
                if step % args.target_every == 0:
                    agent.sync()
                if terminated or truncated:
                    run_totals['episodes_completed'] += 1
                    run_totals['successes'] += int(info['episode_success'])
                    row = dict(type='episode', step=step, training_return=episode_return, length=episode_length,
                               terminated=bool(terminated), truncated=bool(truncated), info=info)
                    json_line(log, row)
                    print(json.dumps(row), flush=True)
                    obs, _ = env.reset()
                    episode_return, episode_length = 0., 0
                if step % args.checkpoint_every == 0:
                    with span('checkpoint'):
                        save_checkpoint(output / 'latest.pt', agent, args, step, env.combos)
            with span('checkpoint'):
                save_checkpoint(output / 'latest.pt', agent, args, args.steps, env.combos)
            summary = dict(type='summary', steps=args.steps, emulator_frames_upper_bound=args.steps * 4,
                           updates=updates, seconds=time.perf_counter() - started, replay_size=replay.size,
                           env_version=ENV_VERSION, reward_mode=args.reward_mode, run_totals=run_totals,
                           c51_mean_target_outside_mass=outside_sum / max(1, updates),
                           partial_episode=last_info if episode_length else None)
            json_line(log, summary)
            print(json.dumps(summary), flush=True)
            if profiler:
                profiler.finish(summary, args.profile, extra={'threads': args.threads, 'batch': args.batch})
    finally:
        env.close()


def rollout(args):
    import imageio.v2 as imageio
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    env = make_env(args.game, args.max_steps, not args.all_lives, args.reward_mode)
    writer = trace = None
    try:
        agent = None
        if args.command == 'eval':
            snapshot = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
            validate_checkpoint(snapshot, args, env.combos)
            agent = Agent(env.action_space.n, args.device, **snapshot['model_spec'])
            agent.online.load_state_dict(snapshot['model'])
        if args.video:
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video, fps=15, macro_block_size=1)
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            trace = report_path.with_suffix('.steps.jsonl').open('w')
        results = []
        env.action_space.seed(args.seed)
        for episode in range(args.episodes):
            obs, initial = env.reset(seed=args.seed + episode)
            total = 0.
            if writer:
                writer.append_data(env.render())
            for step in range(1, args.max_steps + 1):
                action = agent.act(obs, evaluate=True) if agent else env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
                assert np.isfinite(reward)
                total += reward
                if trace:
                    json_line(trace, dict(episode=episode, step=step, action=int(action),
                                          terminated=bool(terminated), truncated=bool(truncated), metrics=info))
                if writer:
                    writer.append_data(env.render())
                if terminated or truncated:
                    break
            row = dict(episode=episode, game=args.game, policy='greedy_mean' if agent else 'random',
                       steps=step, training_return=total, reward_mode=args.reward_mode, env_version=ENV_VERSION,
                       task='all_lives' if args.all_lives else 'first_life', terminated=bool(terminated),
                       truncated=bool(truncated), initial=initial, final=info)
            results.append(row)
            print(json.dumps(row), flush=True)
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(results, indent=2))
    finally:
        if writer:
            writer.close()
        if trace:
            trace.close()
        env.close()


def validate_checkpoint(snapshot, args, actions):
    if snapshot.get('env_version') != ENV_VERSION:
        raise ValueError('Legacy/different environment checkpoint rejected. No automatic reward-semantic migration.')
    config = snapshot['config']
    if config.get('reward_mode') not in REWARD_MODES or config['reward_mode'] != args.reward_mode:
        raise ValueError('Checkpoint reward mode differs; pass the exact matching --reward-mode')
    if config.get('all_lives') != args.all_lives:
        raise ValueError('Checkpoint episode rule differs; match --all-lives / first-life setting')
    if config['game'] != args.game or [list(a) for a in snapshot['actions']] != [list(a) for a in actions]:
        raise ValueError('Checkpoint game/action mapping differs from requested environment')
    if 'model_spec' not in snapshot:
        raise ValueError('Checkpoint lacks explicit C51 support parameters')


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description='Full Rainbow DQN with Stable Retro / Gymnasium')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    imp = sub.add_parser('import-rom')
    imp.add_argument('path')
    for command in ['train', 'probe', 'eval']:
        p = sub.add_parser(command)
        p.add_argument('--game', default='Contra-Nes')
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--max-steps', type=int, default=9000)
        p.add_argument('--all-lives', action=argparse.BooleanOptionalAction, default=False)
        p.add_argument('--reward-mode', choices=REWARD_MODES, default='progress',
                       help='score_clipped=sign(game score delta); game_score=raw score delta; progress=existing shaping')
        p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
        p.add_argument('--threads', type=int, default=4)
        if command == 'train':
            train_parser = p
            p.add_argument('--config', help='Independent JSON training config; explicit CLI options override it')
            p.add_argument('--dry-run', action='store_true', help='Print resolved configuration; do not start training or create output')
            p.add_argument('--learning-rate', type=float, default=.0000625)
            p.add_argument('--adam-eps', type=float, default=1.5e-4)
            p.add_argument('--replay-alpha', type=float, default=.5)
            p.add_argument('--steps', type=int, default=1000000)
            p.add_argument('--replay-storage', choices=['zlib', 'raw'], default='zlib')
            p.add_argument('--standard-actor', action=argparse.BooleanOptionalAction, default=False, help='Use the original materialised NoisyNet actor for comparison')
            p.add_argument('--adam-backend', choices=['loop', 'foreach', 'fused'], default='loop')
            p.add_argument('--profile', help='Write timing/resource JSON')
            p.add_argument('--profile-sync', action=argparse.BooleanOptionalAction, default=False, help='Synchronize device at phase boundaries; diagnostic runs only')
            p.add_argument('--atoms', type=int, default=51)
            p.add_argument('--vmin', type=float, default=-10.)
            p.add_argument('--vmax', type=float, default=10000.)
            p.add_argument('--log-every', type=int, default=100)
            p.add_argument('--capacity', type=int, default=20000)
            p.add_argument('--batch', type=int, default=32)
            p.add_argument('--warmup', type=int, default=2000)
            p.add_argument('--train-every', type=int, default=4)
            p.add_argument('--target-every', type=int, default=2000)
            p.add_argument('--checkpoint-every', type=int, default=10000)
            p.add_argument('--n-step', type=int, default=3)
            p.add_argument('--gamma', type=float, default=0.99)
            p.add_argument('--output', default='runs/contra-seed0')
        else:
            p.add_argument('--episodes', type=int, default=3)
            p.add_argument('--video')
            p.add_argument('--report')
            if command == 'eval':
                p.add_argument('--checkpoint', required=True)
    source_hash = None
    if argv and argv[0] == 'train':
        probe = argparse.ArgumentParser(add_help=False)
        probe.add_argument('--config')
        selected, _ = probe.parse_known_args(argv[1:])
        if selected.config:
            try:
                settings, source_hash = load_train_config(selected.config, train_parser)
            except (OSError, ValueError) as e:
                parser.error(f'Invalid config {selected.config}: {e}')
            train_parser.set_defaults(**settings)
    args = parser.parse_args(argv)
    if args.command == 'train' and args.config:
        args.config = str(Path(args.config).resolve())
        args.source_config_sha256 = source_hash
    if hasattr(args, 'seed') and not 0 <= args.seed < 2**32:
        parser.error('seed must be in [0, 2**32)')
    for key in ['max_steps', 'steps', 'capacity', 'batch', 'train_every', 'target_every',
                'checkpoint_every', 'n_step', 'episodes', 'threads', 'log_every']:
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    if args.command == 'train' and (args.atoms < 2 or not np.isfinite([args.vmin, args.vmax]).all() or args.vmin >= args.vmax):
        parser.error('Require atoms >= 2 and finite vmin < vmax')
    if args.command == 'train' and (not np.isfinite([args.learning_rate, args.adam_eps, args.replay_alpha]).all() or args.learning_rate <= 0 or args.adam_eps <= 0 or not 0 <= args.replay_alpha <= 1):
        parser.error('Require finite learning_rate > 0, adam_eps > 0, replay_alpha in [0,1]')
    if args.command == 'train' and not args.output.strip():
        parser.error('output must be a nonempty directory path')
    if args.command == 'train' and args.profile is not None and not args.profile.strip():
        parser.error('profile must be a nonempty file path or null')
    if args.command == 'train' and args.profile_sync and not args.profile:
        parser.error('profile_sync requires a profile output path')
    if args.command == 'train' and (args.capacity < args.batch or not 0 <= args.gamma <= 1 or args.warmup < 0):
        parser.error('Require capacity >= batch, gamma in [0,1], warmup >= 0')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'train' and args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return
    if args.command == 'doctor':
        doctor()
    elif args.command == 'import-rom':
        import_rom(args.path)
    elif args.command == 'train':
        train(args)
    else:
        rollout(args)


if __name__ == '__main__':
    main()
