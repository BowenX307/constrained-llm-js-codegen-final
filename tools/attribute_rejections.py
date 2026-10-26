"""Which part of ChopChop rejected each token: the grammar, or which pruner rule.

usage:
  python tools/attribute_rejections.py RUN_DIR [--workers N]

Runs on CPU, after generation. For every rejected check in a task's trace (see `trace` in
generators/chopchop_generator.py), the same text is checked again
  1. with the grammar alone -- if that rejects it too, the grammar rejected it;
  2. otherwise with each pruner rule alone (javascript_chopchop.PRUNER_RULES).
A rejection the grammar accepts is a pruner rejection; it lists every rule that rejects it
alone (none listed = only the rules together reject it).

Each pass (grammar alone, each rule alone) goes through a task's rejected texts in order with
its own empty memo tables, because the rewriter's memo is keyed on the arguments and not on
which rules are active. Writes RUN_DIR/attribution.jsonl (one line per task) and prints totals.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chopchop_trace import build_checker_parts, load_run, replay_trace  # noqa: E402

_parts = None


def _init():
    global _parts
    _parts = build_checker_parts()


def attribute(d):
    from llm.realizability import RealizabilityChecker  # type: ignore
    from generators.chopchop_generator import clear_chopchop_caches
    lexerspec, parser, pruner = _parts
    t0 = time.perf_counter()
    rejected = [(text, bool(row[3])) for row, text, _ in replay_trace(d['stats']['trace'], d['stats']['fixed_prefix'])
                if not row[2]]

    def run_pass(constraint, texts):
        clear_chopchop_caches(lexerspec)
        chk = RealizabilityChecker(constraint, parser, lexerspec)
        res = []
        for text, final in texts:
            try:
                res.append(chk.realizable(text, final))
            except Exception as e:                     # same guard as the generation run
                res.append(f'{type(e).__name__}')
        return res

    grammar = run_pass(lambda asts: asts, rejected)
    by_pruner = [i for i, ok in enumerate(grammar) if ok is True]
    rules_of = {i: [] for i in by_pruner}
    for rule in pruner.rules:
        with pruner.only(rule):
            res = run_pass(pruner, [rejected[i] for i in by_pruner])
        for i, ok in zip(by_pruner, res):
            if ok is False:
                rules_of[i].append(rule)

    rows = []
    for i, (text, final) in enumerate(rejected):
        g = grammar[i]
        by = 'grammar' if g is False else 'pruner' if g is True else f'unknown ({g})'
        rows.append(dict(by=by, rules=rules_of.get(i, []), final=final, tail=text[-80:]))
    by_rule = Counter(r for x in rows for r in x['rules'])
    return dict(name=d['name'], rejected=len(rows),
                by_grammar=sum(x['by'] == 'grammar' for x in rows),
                by_pruner=sum(x['by'] == 'pruner' for x in rows),
                by_pruner_no_single_rule=sum(x['by'] == 'pruner' and not x['rules'] for x in rows),
                unknown=sum(x['by'].startswith('unknown') for x in rows),
                by_rule={r: by_rule.get(r, 0) for r in pruner.rules},
                seconds=round(time.perf_counter() - t0, 1), rejections=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir')
    p.add_argument('--workers', type=int, default=os.cpu_count())
    a = p.parse_args()
    items = [d for d in load_run(a.run_dir) if d['stats']['rejected']]
    items.sort(key=lambda d: -d['stats']['rejected'])          # longest first
    out_path = os.path.join(a.run_dir, 'attribution.jsonl')
    t0, done = time.time(), []
    with Pool(a.workers, initializer=_init) as pool, open(out_path + '.tmp', 'w') as f:
        for k, r in enumerate(pool.imap_unordered(attribute, items)):
            f.write(json.dumps(r) + '\n'); f.flush()
            done.append(r)
            print(f"[{k + 1}/{len(items)}] {r['name'][:40]:<41} {r['rejected']:4d} rejections  {r['seconds']:7.1f}s",
                  flush=True)
    os.replace(out_path + '.tmp', out_path)
    tot = Counter()
    for r in done:
        tot.update(dict(rejected=r['rejected'], grammar=r['by_grammar'], pruner=r['by_pruner'],
                        pruner_no_single_rule=r['by_pruner_no_single_rule'], unknown=r['unknown']))
        tot.update({'rule ' + k: v for k, v in r['by_rule'].items()})
    print(f'\n{a.run_dir}: {len(done)} tasks with rejections, {(time.time() - t0) / 60:.1f} min')
    for k, v in tot.items():
        print(f'  {k:<24}{v:6d}')
    print(f"  tasks with a pruner rejection: {sum(1 for r in done if r['by_pruner'])}")
