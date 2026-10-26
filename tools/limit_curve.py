"""How a ChopChop run's results depend on the per-task check-time limit.

usage:
  python tools/limit_curve.py RUN_DIR [RUN_DIR ...] [--limits 60,120,150,300,600,900,1200,1800,2700,3600]

One run with a high limit holds every lower limit: the run is identical up to the first check
at which the task's cumulative check CPU time reaches the limit, and stops there (see
`trace` in generators/chopchop_generator.py). For each limit this rebuilds every task's
completion, runs it under node (evaluate.judge), and prints how many tasks the limit cuts
short, pass / syntax / semantic / other, and the generation time the whole run would take
(summed over tasks, i.e. on one worker).
Limits above the run's own are skipped. Writes RUN_DIR/limit_curve.json.

First it checks that the rebuild with no cut reproduces every stored completion.
"""
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chopchop_trace import REPO, load_run, output_at  # noqa: E402

sys.path.insert(0, REPO)
from evaluate import judge, node_version  # noqa: E402


def curve(run_dir, limits):
    items = load_run(run_dir)
    meta = json.load(open(os.path.join(run_dir, 'run_meta.json')))
    own = meta['settings'].get('check_cpu_limit_s')
    bad = [d['name'] for d in items if output_at(d, None)[0] != d['completion']]
    if bad:
        sys.exit(f'{run_dir}: the trace does not reproduce {len(bad)} stored completions, e.g. {bad[:3]}')
    limits = [x for x in limits if own is None or x <= own]
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for lim in limits:
            outs = [output_at(d, lim) for d in items]
            recs = list(ex.map(lambda a: judge(a[0], a[1][0]), zip(items, outs)))
            cat = Counter(r['category'] for r in recs)
            rows.append(dict(limit_s=lim, tasks=len(items), cut=sum(o[1] for o in outs),
                             passed=cat['pass'], syntax=cat['syntax'], semantic=cat['semantic'],
                             other=cat['other'], gen_hours=round(sum(o[2] for o in outs) / 3600, 2),
                             passed_names=sorted(r['name'] for r in recs if r['category'] == 'pass')))
    out = dict(run_dir=run_dir, run_limit_s=own, node=node_version(), rows=rows)
    json.dump(out, open(os.path.join(run_dir, 'limit_curve.json'), 'w'), indent=1)
    return out


def show(out):
    print(f"{out['run_dir']}  (run limit {out['run_limit_s']} s, node {out['node']})")
    print(f"  {'limit s':>8} {'cut':>5} {'pass':>5} {'syntax':>7} {'semantic':>9} {'other':>6} {'1-worker h':>10}")
    for r in out['rows']:
        print(f"  {r['limit_s']:>8g} {r['cut']:>5} {r['passed']:>5} {r['syntax']:>7} {r['semantic']:>9} "
              f"{r['other']:>6} {r['gen_hours']:>10}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dirs', nargs='+')
    p.add_argument('--limits', default='60,120,150,300,600,900,1200,1800,2700,3600')
    a = p.parse_args()
    for d in a.run_dirs:
        show(curve(d, [float(x) for x in a.limits.split(',')]))
        print()
