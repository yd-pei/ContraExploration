from collections import deque
from pathlib import Path
import gymnasium as gym
import numpy as np
from PIL import Image
from .performance import span

ENV_VERSION = 'contra-level1-three-rewards-v2'
REWARD_MODES = ('score_clipped', 'game_score', 'progress')
REWARD_SPECS = {
    'score_clipped': 'sign(sum(score RAM deltas over repeated frames) * 100)',
    'game_score': 'sum(score RAM deltas over repeated frames) * 100; no clipping/scaling',
    'progress': 'clip(new camera highwater pixels / 256 - deaths, -1, 1); stage-clear step overrides with +10',
}

# Contra NES: B fires, A jumps. START/SELECT and opposing directions excluded.
CONTRA_ACTIONS = [(), ('B',), ('A',), ('A', 'B')]
for direction in [('LEFT',), ('RIGHT',), ('UP',), ('DOWN',),
                  ('UP', 'LEFT'), ('UP', 'RIGHT'), ('DOWN', 'LEFT'), ('DOWN', 'RIGHT')]:
    CONTRA_ACTIONS.append(direction)
    CONTRA_ACTIONS.append(direction + ('B',))
    if 'DOWN' not in direction:
        CONTRA_ACTIONS.append(direction + ('A', 'B'))


def select_reward(mode, score_delta, progress_reward):
    if mode == 'score_clipped':
        return float(np.sign(score_delta))
    if mode == 'game_score':
        return float(score_delta)
    if mode == 'progress':
        return float(progress_reward)
    raise ValueError(f'Unknown reward mode {mode!r}; use {REWARD_MODES}. '
                     'Old native meant xscroll, not game score, and has been removed.')


def score_delta_points(previous, current):
    """Contra binary uint16 score; renderer appends two zero digits.

    The real game saturates at 0xffff instead of wrapping. A decrease is a
    discontinuity (reset/state change/bad RAM), not a huge reward or death penalty.
    reset() sets a fresh baseline, and first-level completion preserves the score.
    """
    previous, current = int(previous), int(current)
    if not 0 <= previous <= 65535 or not 0 <= current <= 65535:
        raise ValueError('Contra score RAM outside uint16 range')
    if current < previous:
        raise RuntimeError(f'Unexpected Contra score decrease {previous}->{current}; '
                           'reset via env.reset(); uint16 wraparound is not valid for this ROM')
    return (current - previous) * 100


class RetroPixels(gym.Wrapper):
    def __init__(self, env, actions, contra=False, skip=4, max_steps=9000, first_life=True, reward_mode='progress'):
        super().__init__(env)
        select_reward(reward_mode, 0, 0)  # Reject ambiguous legacy names immediately.
        if not contra and reward_mode != 'progress':
            raise ValueError('Game score modes currently support Contra-Nes only')
        self.combos, self.contra, self.reward_mode = actions, contra, reward_mode
        self.skip, self.max_steps, self.first_life = skip, max_steps, first_life
        self.action_space = gym.spaces.Discrete(len(actions))
        self.observation_space = gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
        buttons = env.unwrapped.buttons
        self.masks = [np.array([int(b in combo) for b in buttons], dtype=np.int8) for combo in actions]
        for combo in actions:
            if not set(combo) <= set(buttons):
                raise ValueError(f'Unavailable buttons: {combo}; emulator has {buttons}')
        self.frames = deque(maxlen=4)

    @staticmethod
    def pixel(frame):
        return np.asarray(Image.fromarray(frame).convert('L').resize((84, 84), Image.Resampling.BILINEAR))

    def reset(self, **kwargs):
        frame, info = self.env.reset(**kwargs)
        if self.contra:
            info = dict(self.env.unwrapped.data.lookup_all())
            for key in ['level', 'lives', 'player_state', 'score', 'game_over', 'xscroll']:
                if key not in info:
                    raise RuntimeError(f'Contra integration missing {key}')
            score_delta_points(info['score'], info['score'])
        self.initial_level = info.get('level')
        self.previous_lives = info.get('lives')
        self.previous_state = info.get('player_state')
        self.previous_score = info.get('score', 0)
        self.initial_score = self.previous_score * 100
        self.progress_highwater = info.get('xscroll', 0)
        self.steps = 0
        self.totals = dict(episode_game_score_gain=0, episode_integration_scroll_return=0.,
                           episode_progress_return=0., episode_score_clipped_return=0.,
                           episode_training_return=0., episode_deaths=0, episode_extra_lives=0,
                           episode_success=False, episode_emulator_frames=0, episode_score_saturations=0)
        self.frames.clear()
        self.frames.extend([self.pixel(frame)] * 4)
        return np.stack(self.frames), self._metrics(info)

    def _metrics(self, info):
        return dict(info, **self.totals, game_score=int(info['score']) * 100 if self.contra else None,
                    initial_game_score=self.initial_score if self.contra else None,
                    progress_max=self.progress_highwater, decision_steps=self.steps,
                    reward_mode=self.reward_mode, env_version=ENV_VERSION)

    def step(self, action):
        scroll_reward, shaped, score_delta = 0., 0., 0
        deaths = extra_lives = saturation_events = frames_advanced = 0
        lost_life_any = success_any = False
        recent = deque(maxlen=2)
        terminated = truncated = False
        info = {}
        for _ in range(self.skip):
            with span('emulator_step'):
                frame, reward, te, tr, info = self.env.step(self.masks[int(action)])
            frames_advanced += 1
            recent.append(frame)
            scroll_reward += float(reward)
            if self.contra:
                score_delta += score_delta_points(self.previous_score, info['score'])
                saturation_events += int(self.previous_score < 65535 and int(info['score']) == 65535)
                self.previous_score = int(info['score'])
                # Verified $64/$65 camera screen/offset, not player screen X.
                progress = info['xscroll']
                delta = progress - self.progress_highwater
                if info['level'] == self.initial_level and 0 < delta <= 16 and progress <= 8192:
                    shaped += delta / 256.
                    self.progress_highwater = progress
                lost_life_any = lost_life_any or info['lives'] < self.previous_lives
                extra_lives += max(0, int(info['lives']) - int(self.previous_lives))
                self.previous_lives = info['lives']
                success_any = success_any or info['level'] != self.initial_level
                death = info['player_state'] == 2 and self.previous_state != 2
                self.previous_state = info['player_state']
                deaths += int(death)
                shaped -= float(death)
                te = te or success_any or bool(info['game_over']) or (self.first_life and death)
            else:
                shaped += float(reward)
            terminated, truncated = terminated or te, truncated or tr
            if terminated or truncated:
                break
        self.steps += 1
        truncated = truncated or (self.steps >= self.max_steps and not terminated)
        with span('preprocessing'):
            self.frames.append(self.pixel(np.maximum.reduce(recent)))
        progress_reward = 10. if success_any else float(np.clip(shaped, -1., 1.))
        # Sign is applied ONCE, after accumulating all frames in this agent step.
        clipped_reward = float(np.sign(score_delta))
        training_reward = select_reward(self.reward_mode, score_delta, progress_reward)
        self.totals['episode_game_score_gain'] += score_delta
        self.totals['episode_integration_scroll_return'] += scroll_reward
        self.totals['episode_progress_return'] += progress_reward
        self.totals['episode_score_clipped_return'] += clipped_reward
        self.totals['episode_training_return'] += training_reward
        self.totals['episode_deaths'] += deaths
        self.totals['episode_extra_lives'] += extra_lives
        self.totals['episode_success'] = self.totals['episode_success'] or success_any
        self.totals['episode_emulator_frames'] += frames_advanced
        self.totals['episode_score_saturations'] += saturation_events
        info = self._metrics(dict(info, success=success_any, lost_life=lost_life_any, death=bool(deaths),
                                  deaths=deaths, extra_lives=extra_lives, game_score_delta=score_delta,
                                  integration_scroll_reward=scroll_reward, progress_reward=progress_reward,
                                  score_clipped_reward=clipped_reward, training_reward=training_reward,
                                  frames_advanced=frames_advanced))
        return np.stack(self.frames), training_reward, terminated, truncated, info


def make_env(game='Contra-Nes', max_steps=9000, first_life=True, reward_mode='progress'):
    import stable_retro as retro
    contra = game == 'Contra-Nes'
    select_reward(reward_mode, 0, 0)
    if not contra and reward_mode != 'progress':
        raise ValueError('Game score modes currently support Contra-Nes only')
    if 'Contra' in game and not contra:
        raise ValueError('This project targets Contra-Nes, not Contra Force.')
    try:
        integration = Path(__file__).parent / 'integration'
        custom = {'info': str(integration / 'data.json'), 'scenario': str(integration / 'scenario.json')} if contra else {}
        raw = retro.make(game=game, **custom, state='Level1' if contra else retro.State.DEFAULT,
                         inttype=retro.data.Integrations.ALL, render_mode='rgb_array')
    except FileNotFoundError as e:
        raise RuntimeError('ROM 缺失。运行 uv run contra-rl doctor，再用 import-rom 导入匹配 ROM。') from e
    actions = CONTRA_ACTIONS if contra else [(), ('B',), ('LEFT',), ('RIGHT',), ('LEFT', 'B'), ('RIGHT', 'B')]
    return RetroPixels(raw, actions, contra, max_steps=max_steps, first_life=first_life, reward_mode=reward_mode)
