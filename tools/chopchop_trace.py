"""Shared helpers for the ChopChop trace tools: load a run, rebuild outputs, build a checker."""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, 'generators', 'chopchop')]

from generators.chopchop_generator import ChopchopGenerator, replay_trace  # noqa: E402


def load_run(run_dir):
    """The run's per-task JSON files (ChopChop runs with a trace only)."""
    items = []
    for f in sorted(os.listdir(run_dir)):
        if f.startswith('mbpp_') and f.endswith('.json'):
            d = json.load(open(os.path.join(run_dir, f)))
            if d['mode'] != 'chopchop' or 'trace' not in d['stats']:
                sys.exit(f'{f}: not a ChopChop output with a check trace')
            items.append(d)
    return items


def output_at(d, limit):
    """(completion, cut, wall_s) for task d had the run stopped at `limit` s of check CPU.

    The run is identical up to the first check at which the cumulative check CPU time reaches
    the limit; generation stops there, at the last accepted text. limit=None: no cut.
    """
    st = d['stats']
    accepted, cut, wall = st['fixed_prefix'], False, d['wall_s']
    for row, _, accepted in replay_trace(st['trace'], st['fixed_prefix']):
        if limit is not None and row[0] >= limit:
            cut, wall = True, row[1]
            break
    if not cut and limit is not None and st.get('limit_hit'):
        raise ValueError(f"{d['name']}: limit {limit} is above the run's own limit")
    return ChopchopGenerator.finish_output(accepted, st['fixed_prefix'], d.get('stop_tokens') or []), cut, wall


def build_checker_parts():
    """(lexerspec, parser, pruner) exactly as run.py builds them for ChopChop, without a model."""
    from core.lark.from_lark import parse_attribute_grammar  # type: ignore
    from generators.javascript_chopchop import CONSTRUCTORS, JS_START_RULE, make_js_pruner
    src = open(os.path.join(REPO, 'generators', 'grammars', 'javascript_chopchop.lark'), encoding='utf-8').read()
    lexerspec, parser = parse_attribute_grammar(CONSTRUCTORS, src, JS_START_RULE).build_parser()
    return lexerspec, parser, make_js_pruner('basic')
