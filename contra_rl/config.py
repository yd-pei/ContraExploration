"""Small strict JSON loader for one training run; no configuration framework."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON number: {value}')


def load_train_config(path, parser):
    source = Path(path).read_bytes()
    values = json.loads(source, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(values, dict):
        raise ValueError('Expected a flat JSON object of train options')
    actions = {a.dest: a for a in parser._actions if a.dest not in ['help', 'config', 'dry_run']}
    unknown = set(values) - set(actions)
    if unknown:
        raise ValueError(f'Unknown training keys: {", ".join(sorted(unknown))}')
    missing = {'reward_mode', 'output'} - set(values)
    if missing:
        raise ValueError(f'Config must explicitly specify {", ".join(sorted(missing))}')
    for key, value in values.items():
        action = actions[key]
        if isinstance(action, argparse.BooleanOptionalAction):
            valid = type(value) is bool
        elif action.type is int:
            valid = type(value) is int  # bool is not an integer setting.
        elif action.type is float:
            valid = type(value) in [int, float] and math.isfinite(value)
        else:
            valid = isinstance(value, str) or (value is None and action.default is None)
        if not valid:
            raise ValueError(f'Invalid type/value for {key}: {value!r}')
        if action.choices is not None and value not in action.choices:
            raise ValueError(f'Invalid {key}: {value!r}; choose from {list(action.choices)}')
    if not values['output'].strip():
        raise ValueError('output must be a nonempty directory path')
    return values, hashlib.sha256(source).hexdigest()
