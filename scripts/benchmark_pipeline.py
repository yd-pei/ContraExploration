"""Run current or frozen code under the same uv interpreter and CLI arguments."""
import argparse
import importlib
from pathlib import Path
import sys

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--source', help='Directory containing the frozen contra_rl package')
args, rest = p.parse_known_args()
if args.source:
    sys.path.insert(0, str(Path(args.source).resolve()))
sys.argv = ['contra-rl'] + rest
importlib.import_module('contra_rl.cli').main()
