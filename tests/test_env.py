import gymnasium as gym
import numpy as np
import pytest
from collections import deque
from contra_rl.env import RetroPixels, CONTRA_ACTIONS, REWARD_MODES, select_reward, score_delta_points


class FakeRetro(gym.Env):
    buttons = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
    observation_space = gym.spaces.Box(0, 255, (224, 240, 3), np.uint8)
    action_space = gym.spaces.MultiBinary(9)
    def __init__(self):
        self.data = self
        self.ram = dict(level=0, lives=0, player_state=1, score=0, xscroll=0, game_over=0)
        self.tick = 0
        self.event = None
        self.events = deque()
    def lookup_all(self):
        return self.ram.copy()
    def reset(self, **kwargs):
        return np.zeros((224, 240, 3), np.uint8), {}
    def step(self, mask):
        self.tick += 1
        if self.events:
            self.ram.update(self.events.popleft())
        elif self.event:
            self.ram.update(self.event)
        return np.full((224, 240, 3), self.tick % 256, np.uint8), 1000., False, False, self.ram.copy()


def test_last_life_is_playable_and_death_is_terminal():
    base = FakeRetro()
    env = RetroPixels(base, CONTRA_ACTIONS, contra=True)
    env.reset()
    _, reward, te, _, _ = env.step(0)
    assert reward == 0 and not te  # lives=0 means last playable life, not game over
    base.event = dict(player_state=2)
    _, reward, te, tr, info = env.step(0)
    assert te and not tr and reward == -1 and info['death']


def test_stage_clear_bonus_timeout_and_no_scroll_reward():
    base = FakeRetro()
    env = RetroPixels(base, CONTRA_ACTIONS, contra=True, max_steps=1)
    env.reset()
    base.event = dict(xscroll=9999)
    _, reward, te, tr, _ = env.step(0)
    assert reward == 0 and not te and tr
    env.reset()
    base.event = dict(level=1)
    _, reward, te, tr, info = env.step(0)
    assert reward == 10 and te and not tr and info['success']


def test_all_lives_game_over_and_progress_envelope():
    base = FakeRetro()
    env = RetroPixels(base, CONTRA_ACTIONS, contra=True, first_life=False)
    env.reset()
    base.event = dict(score=200, xscroll=2, player_state=2)
    _, r, te, _, _ = env.step(0)
    assert np.isclose(r, -1 + 2 / 256) and not te
    _, r, te, _, _ = env.step(0)
    assert r == 0 and not te  # repeated dead state / progress cannot farm reward
    base.event = dict(game_over=1)
    assert env.step(0)[2]


@pytest.mark.parametrize('mode,expected', [('score_clipped', 1), ('game_score', 600), ('progress', 0)])
def test_repeat_frames_accumulate_before_sign(mode, expected):
    base = FakeRetro()
    base.events.extend([dict(score=1), dict(score=3), dict(score=6), dict(score=6)])
    env = RetroPixels(base, CONTRA_ACTIONS, contra=True, reward_mode=mode)
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert reward == expected  # sign(100+200+300+0)=1, not sum(sign(...))=3
    assert info['game_score'] == 600 and info['game_score_delta'] == 600
    assert info['score_clipped_reward'] == 1 and info['progress_reward'] == 0
    assert info['integration_scroll_reward'] == 4000 and info['training_reward'] == expected
    assert info['episode_game_score_gain'] == 600
    assert info['frames_advanced'] == 4


def test_sign_including_negative_and_zero_net_is_applied_after_sum():
    assert select_reward('score_clipped', sum([100, -100, 200, -200]), 10) == 0
    assert select_reward('score_clipped', sum([-100, -200, 0, 100]), 10) == -1
    assert select_reward('game_score', -200, 10) == -200


@pytest.mark.parametrize('mode', REWARD_MODES)
def test_reset_score_baseline_and_extra_life_never_imply_death(mode):
    base = FakeRetro()
    base.ram.update(score=299, lives=2)
    base.events.extend([dict(score=300, lives=3), {}, {}, {}])
    env = RetroPixels(base, CONTRA_ACTIONS, contra=True, reward_mode=mode)
    env.reset()
    _, reward, te, _, info = env.step(0)
    assert not te and not info['death'] and not info['lost_life']
    assert info['extra_lives'] == 1 and info['episode_extra_lives'] == 1
    assert info['game_score_delta'] == 100 and info['game_score'] == 30000
    assert reward == {'score_clipped': 1, 'game_score': 100, 'progress': 0}[mode]
    base.ram.update(score=0, lives=2)
    env.reset()  # Explicit reset clears score baseline, totals and death state.
    _, reward, te, _, info = env.step(0)
    assert info['game_score_delta'] == 0 and info['episode_game_score_gain'] == 0 and reward == 0


def test_binary_score_byte_carry_saturation_and_invalid_wrap():
    assert score_delta_points(255, 256) == 100  # binary, not BCD
    assert score_delta_points(65534, 65535) == 100
    assert score_delta_points(65535, 65535) == 0
    with pytest.raises(RuntimeError, match='decrease'):
        score_delta_points(65535, 0)
    with pytest.raises(ValueError, match='uint16'):
        score_delta_points(65535, 65536)
    base = FakeRetro()
    base.ram['score'] = 65534
    base.event = dict(score=65535)
    env = RetroPixels(base, CONTRA_ACTIONS, contra=True, reward_mode='game_score')
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert reward == 100 and info['episode_score_saturations'] == 1
    assert env.step(0)[1] == 0


@pytest.mark.parametrize('first_life', [True, False])
def test_all_reward_modes_share_termination_and_metrics(first_life):
    rows = []
    for mode in REWARD_MODES:
        base = FakeRetro()
        base.events.extend([dict(score=2, xscroll=2, player_state=2), {}, {}, {}])
        env = RetroPixels(base, CONTRA_ACTIONS, contra=True, first_life=first_life, reward_mode=mode)
        env.reset()
        _, _, te, tr, info = env.step(0)
        rows.append((te, tr, info['frames_advanced'], info['game_score_delta'], info['progress_reward'], info['episode_deaths']))
    assert rows[0] == rows[1] == rows[2]
    assert rows[0][0] == first_life


def test_legacy_native_name_rejected():
    with pytest.raises(ValueError, match='removed'):
        RetroPixels(FakeRetro(), CONTRA_ACTIONS, contra=True, reward_mode='native')
