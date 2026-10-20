"""Real ROM checks; separate from ROM-free pytest tests."""
import json
from pathlib import Path
import numpy as np
from contra_rl.env import make_env


def main():
    results = {}
    env = make_env(max_steps=3000)
    try:
        first, initial = env.reset(seed=0)
        second, repeated = env.reset(seed=0)
        assert np.array_equal(first, second) and initial == repeated
        results['reset_identical'] = True
        total = 0.
        for step in range(1, 3001):
            obs, reward, terminated, truncated, info = env.step(env.combos.index(('RIGHT', 'B')))
            assert obs.dtype == np.uint8 and obs.shape == (4, 84, 84)
            assert np.isfinite(reward)
            total += reward
            if terminated or truncated:
                break
        assert terminated and info['death'] and not info['game_over']
        assert info['progress_max'] > 0
        results['first_life'] = dict(steps=step, return_=total, final=info)
    finally:
        env.close()
    env = make_env(max_steps=3000, first_life=False)
    try:
        env.reset(seed=0)
        events, previous, deaths, total = [], None, 0, 0.
        for step in range(1, 3001):
            _, reward, terminated, truncated, info = env.step(env.combos.index(('RIGHT', 'B')))
            total += reward
            deaths += int(info['death'])
            key = (info['lives'], info['player_state'], info['game_over'])
            if key != previous:
                events.append(dict(step=step, reward=reward, **info))
                previous = key
            if terminated or truncated:
                break
        assert terminated and not truncated and info['game_over'] == 1
        assert deaths == 3
        assert any(e['lives'] == 0 and e['player_state'] == 1 and e['game_over'] == 0 for e in events)
        results['all_lives'] = dict(steps=step, return_=total, deaths=deaths, events=events)
    finally:
        env.close()
    path = Path('reports/contra-all-lives.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps({'reset_identical': True, 'first_life_steps': results['first_life']['steps'],
                      'all_lives_steps': step, 'deaths': deaths, 'game_over_verified': True}))


if __name__ == '__main__':
    main()
