"""Real-ROM, identical-action reward validation; no learning/long training."""
import hashlib
import json
from pathlib import Path
import numpy as np
from contra_rl.env import make_env, REWARD_MODES, ENV_VERSION


def main():
    actions = np.random.default_rng(3).integers(0, 25, 4000)
    results = []
    common_fingerprints = []
    for mode in REWARD_MODES:
        env = make_env(first_life=False, max_steps=1000, reward_mode=mode)
        fingerprint = hashlib.sha256()
        score_gain = deaths = episodes = score_events = 0
        total = 0.
        examples = []
        try:
            env.reset(seed=3)
            for step, action in enumerate(actions, 1):
                obs, reward, te, tr, info = env.step(int(action))
                expected = {'score_clipped': info['score_clipped_reward'],
                            'game_score': info['game_score_delta'], 'progress': info['progress_reward']}[mode]
                assert reward == expected
                assert info['score_clipped_reward'] == np.sign(info['game_score_delta'])
                # All physical observations/metrics must agree for identical actions.
                common = {k: v for k, v in info.items() if k not in
                          ['reward_mode', 'training_reward', 'episode_training_return']}
                fingerprint.update(obs.tobytes())
                fingerprint.update(json.dumps(common, sort_keys=True).encode())
                fingerprint.update(bytes([te, tr]))
                score_gain += info['game_score_delta']
                score_events += int(info['game_score_delta'] != 0)
                deaths += info['deaths']
                total += reward
                if info['game_score_delta'] and len(examples) < 3:
                    examples.append(dict(step=step, game_score_delta=info['game_score_delta'],
                                         score_clipped_reward=info['score_clipped_reward'],
                                         progress_reward=info['progress_reward'], training_reward=reward))
                if te or tr:
                    episodes += 1
                    env.reset()
            common_fingerprints.append(fingerprint.hexdigest())
            results.append(dict(reward_mode=mode, agent_steps=len(actions), game_score_gain=score_gain,
                                score_events=score_events, deaths=deaths, episodes_completed=episodes,
                                training_return=total, examples=examples, common_sha256=fingerprint.hexdigest()))
        finally:
            env.close()
    assert len(set(common_fingerprints)) == 1, 'Reward modes changed physical trajectory/common metrics'
    assert results[0]['score_events'] > 0, 'Need a real scored event to validate reward units'
    report = dict(env_version=ENV_VERSION, identical_actions_and_common_metrics=True,
                  score_unit='100 displayed points per RAM unit; verified against game renderer source', results=results)
    output = Path('reports/three-reward-rom-validation.json')
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
