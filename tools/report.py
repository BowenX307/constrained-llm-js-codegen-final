"""Every table of the final report, from the run folders.

usage:
  python tools/report.py ROOT [--commit 2c14eef] [--out reports/final]

ROOT is laid out like the Drive folder: ROOT/<model>/<run>/ with run_meta.json, mbpp_*.json and
eval.jsonl (written by evaluate.py; run here if it is missing), and for ChopChop also
limit_curve.json and attribution.jsonl. Only full runs (no --sample) made at --commit are used.

Writes OUT/tables.md (all tables), OUT/tables.json (the same numbers) and OUT/tasks.csv (one row
per model x mode x task). Every error class and message comes from node (evaluate.py).

Tables
  1 controlled variables     what is the same and what differs between runs
  2 pass@1                   pass / syntax / semantic / other, with a 95% Wilson interval
  3 error subcategories      node's error class, and the most common messages per class
  4 time                     generation time per task; time inside the checks
  5 IterGen semantic checks  how often each rule fired, and whether node still reports the
                             error that rule is about in the final program
  6 ChopChop checks          checks, rejections, limit and wall-cap hits; grammar vs pruner rule (sampled)
  7 limit curve              ChopChop results under every lower check-CPU limit
  8 Qwen3.5 vs Qwen2.5       paired per task, with an exact McNemar test
  9 IterGen on vs off        paired per task
"""
import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODES = ('unconstrained', 'itergen_off', 'itergen_on', 'chopchop')
CATS = ('pass', 'syntax', 'semantic', 'other')
PROMPT = {'unconstrained': 'raw MultiPL-E prompt', 'itergen_off': 'raw MultiPL-E prompt',
          'itergen_on': 'raw MultiPL-E prompt', 'chopchop': 'chat template + rewritten prompt'}
# The node error each IterGen rule is about (generators/semantic_checks.py).
RULE_NODE_ERROR = {
    'undeclared': ('ReferenceError', 'X is not defined'),
    'redeclaration': ('SyntaxError', 'Identifier X has already been declared'),
    'const_reassign': ('TypeError', 'Assignment to constant variable.'),
    'orphan_break': ('SyntaxError', 'Illegal break statement'),
}


# ---------------------------------------------------------------- loading

def load_runs(root, commit):
    runs = {}
    for slug in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, slug)):
            continue
        for run in sorted(os.listdir(os.path.join(root, slug))):
            d = os.path.join(root, slug, run)
            meta_p = os.path.join(d, 'run_meta.json')
            if not os.path.exists(meta_p):
                continue
            meta = json.load(open(meta_p))
            s = meta['settings']
            if s.get('sample') or (commit and not meta['env'].get('repo_commit', '').startswith(commit)):
                continue
            key = (s['model'], s['mode'])
            if key in runs:
                sys.exit(f'two runs for {key}: {runs[key]["dir"]} and {d}')
            runs[key] = dict(dir=d, meta=meta, tasks=load_tasks(d))
    return runs


def load_tasks(d):
    eval_p = os.path.join(d, 'eval.jsonl')
    if not os.path.exists(eval_p):
        subprocess.run([sys.executable, os.path.join(REPO, 'evaluate.py'), d], check=True,
                       stdout=subprocess.DEVNULL)
    ev = {r['name']: r for r in map(json.loads, open(eval_p))}
    tasks = {}
    for f in sorted(os.listdir(d)):
        if f.startswith('mbpp_') and f.endswith('.json'):
            t = json.load(open(os.path.join(d, f)))
            t['stats'].pop('trace', None)                 # large; the limit curve has what we need
            t['eval'] = ev[t['name']]
            tasks[t['name']] = t
    if set(tasks) != set(ev):
        sys.exit(f'{d}: eval.jsonl does not match the task files; rerun evaluate.py')
    return tasks


def load_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


# ---------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pct(k, n):
    return f'{100 * k / n:.1f}%' if n else '-'


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def md_table(header, rows):
    out = ['| ' + ' | '.join(map(str, header)) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join('' if x is None else str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(out)


def short(model):
    return model.split('/')[-1]


# ---------------------------------------------------------------- tables

def t_controlled(runs):
    rows = []
    for (model, mode), r in sorted(runs.items()):
        s, e = r['meta']['settings'], r['meta']['env']
        limit = (f"{s['check_cpu_limit_s']:g} s check CPU" if mode == 'chopchop'
                 else {'itergen_off': '60 s wall', 'itergen_on': '60 s wall'}.get(mode, 'none'))
        rows.append([short(model), mode, len(r['tasks']), s['decoding'], s['max_new_tokens'], PROMPT[mode],
                     limit,
                     e.get('gpu'), e.get('cpu'), e.get('n_cpus'), e.get('torch'), e.get('transformers'),
                     e.get('repo_commit', '')[:7] + (' (dirty)' if e.get('repo_dirty') else '')])
    return md_table(['model', 'mode', 'tasks', 'decoding', 'max new tokens', 'prompt', 'time limit', 'GPU',
                     'CPU', 'CPUs', 'torch', 'transformers', 'commit'], rows), rows


def t_pass(runs):
    rows, data = [], {}
    for (model, mode), r in sorted(runs.items(), key=lambda kv: (kv[0][0], MODES.index(kv[0][1]))):
        c = Counter(t['eval']['category'] for t in r['tasks'].values())
        n = len(r['tasks'])
        lo, hi = wilson(c['pass'], n)
        rows.append([short(model), mode, n, c['pass'], pct(c['pass'], n), f'{100 * lo:.1f}-{100 * hi:.1f}%',
                     c['syntax'], c['semantic'], c['other']])
        data[f'{short(model)} {mode}'] = dict(n=n, **{k: c[k] for k in CATS})
    return md_table(['model', 'mode', 'tasks', 'pass', 'pass@1', '95% CI', 'syntax', 'semantic', 'other'], rows), data


def t_errors(runs, top=5):
    statuses = Counter(t['eval']['status'] for r in runs.values() for t in r['tasks'].values()
                       if t['eval']['category'] != 'pass')
    order = [s for s, _ in statuses.most_common()]
    keys = sorted(runs, key=lambda k: (k[0], MODES.index(k[1])))
    rows = []
    for st in order:
        rows.append([st] + [sum(t['eval']['status'] == st for t in runs[k]['tasks'].values()) for k in keys])
    by_class = md_table(['node error class'] + [f'{short(m)} {mode}' for m, mode in keys], rows)

    parts = [by_class]
    # where node points, for syntax errors: in the completion or in the tests (code after an
    # unfinished function); and parse vs runtime
    rows = []
    for k in keys:
        ts = [t['eval'] for t in runs[k]['tasks'].values() if t['eval']['status'] == 'SyntaxError']
        rows.append([short(k[0]), k[1], len(ts),
                     sum(e['region'] == 'completion' for e in ts), sum(e['region'] == 'tests' for e in ts),
                     sum(e['region'] == 'prompt' for e in ts), sum(e['phase'] == 'runtime' for e in ts)])
    parts.append('SyntaxError: where node points\n\n' + md_table(
        ['model', 'mode', 'SyntaxError', 'in completion', 'in tests', 'in prompt', 'thrown at runtime'], rows))
    # most common messages per class, per run
    for st in order:
        if st in ('Timeout', 'GenerationFailed', 'NonzeroExit', 'AssertionError'):
            continue
        rows = []
        for k in keys:
            c = Counter(t['eval']['message_template'] for t in runs[k]['tasks'].values()
                        if t['eval']['status'] == st)
            for msg, n in c.most_common(top):
                rows.append([short(k[0]), k[1], n, f'`{msg}`'])
        if rows:
            parts.append(f'{st}: most common messages (names and numbers replaced by X / N)\n\n'
                         + md_table(['model', 'mode', 'tasks', 'message'], rows))
    return '\n\n'.join(parts)


def t_time(runs):
    rows = []
    for (model, mode), r in sorted(runs.items(), key=lambda kv: (kv[0][0], MODES.index(kv[0][1]))):
        ts = list(r['tasks'].values())
        w = [t['wall_s'] for t in ts]
        toks = [t['stats'].get('tokens') for t in ts if t['stats'].get('tokens') is not None]
        if mode.startswith('itergen'):
            chk = sum(t['stats'].get('check_s', 0) for t in ts)
            check = f'{chk:.1f} s wall in view() + rules ({pct(chk, sum(w))} of total)'
        elif mode == 'chopchop':
            cpu = sum(t['stats'].get('check_cpu_s', 0) for t in ts)
            wall = sum(t['stats'].get('check_wall_s', 0) for t in ts)
            check = (f'{cpu / 3600:.2f} h CPU / {wall / 3600:.2f} h wall in the checker '
                     f'({pct(wall, sum(w))} of total)')
        else:
            check = '-'
        rows.append([short(model), mode, f'{sum(w) / 3600:.2f}', f'{statistics.median(w):.1f}',
                     f'{quantile(w, 0.9):.1f}', f'{max(w):.0f}',
                     statistics.median(toks) if toks else '-', check])
    return md_table(['model', 'mode', 'total h (1 worker)', 'median s/task', 'p90 s/task', 'max s',
                     'median tokens', 'time in checks'], rows)


def node_error_of(t, rule):
    cls, msg = RULE_NODE_ERROR[rule]
    e = t['eval']
    # node quotes the name in some messages but not in "flip is not defined", so match the raw message
    pattern = '^' + re.escape(msg).replace('X', r'\S+') + '$'
    return e['status'] == cls and bool(re.match(pattern, e['message'] or ''))


def t_itergen(runs):
    parts = []
    for model in sorted({m for m, _ in runs}):
        on, off = runs.get((model, 'itergen_on')), runs.get((model, 'itergen_off'))
        if not on:
            continue
        ts = on['tasks']
        v = [(n, x) for n, t in ts.items() for x in t['stats'].get('violations', [])]
        rows = []
        for rule in RULE_NODE_ERROR:
            fired = {n for n, x in v if x['rule'] == rule}
            still = sum(node_error_of(ts[n], rule) for n in fired)
            gave_up = sum(1 for n, x in v if x['rule'] == rule and x.get('gave_up'))
            off_n = sum(node_error_of(t, rule) for t in off['tasks'].values()) if off else None
            on_n = sum(node_error_of(t, rule) for t in ts.values())
            rows.append([rule, f'{RULE_NODE_ERROR[rule][0]}: {RULE_NODE_ERROR[rule][1]}',
                         sum(1 for _, x in v if x['rule'] == rule), len(fired), gave_up,
                         f'{len(fired) - still}/{len(fired)}' if fired else '-', off_n, on_n])
        stop = Counter(t['stats'].get('stop_reason') for t in ts.values())
        fb = sum(t['stats'].get('fallback_used', False) for t in ts.values())
        parts.append(f'**{short(model)}**, itergen_on. "Error gone" = the task where the rule fired does not '
                     f'end with the node error that rule is about. The last two columns count that node '
                     f'error over all tasks, with the checks off and on.\n\n' + md_table(
                         ['rule', 'node error it is about', 'fired', 'tasks', 'gave up', 'error gone',
                          'tasks with that error: off', 'on'], rows)
                     + f'\n\nstop reasons: {dict(stop)}; unconstrained fallback used on {fb} tasks')
    return '\n\n'.join(parts) or '(no itergen_on run)'


def t_chopchop(runs):
    rows, rule_rows = [], []
    for model in sorted({m for m, _ in runs}):
        r = runs.get((model, 'chopchop'))
        if not r:
            continue
        s = [t['stats'] for t in r['tasks'].values()]
        rej = sum(x.get('rejected', 0) for x in s)
        rows.append([short(model), sum(x.get('checks', 0) for x in s), rej,
                     sum(1 for x in s if x.get('rejected')), sum(1 for x in s if x.get('limit_hit')),
                     sum(1 for x in s if x.get('wall_cap_hit')),
                     sum(x.get('check_errors', 0) for x in s),
                     sum(1 for t in r['tasks'].values() if t.get('generation_error'))])
        att = load_jsonl(os.path.join(r['dir'], 'attribution.jsonl'))
        if att:
            # each task's sampled shares, scaled to its total number of rejections
            est = Counter()
            for a in att:
                if not a['attributed']:
                    continue
                w = a['rejected'] / a['attributed']
                est['grammar'] += w * a['by_grammar']
                est['pruner'] += w * a['by_pruner']
                est['pruner (no single rule)'] += w * a['by_pruner_no_single_rule']
                est['unknown'] += w * a['unknown']
                for k, n in a['by_rule'].items():
                    est['rule ' + k] += w * n
            tot = sum(a['rejected'] for a in att)
            sampled = sum(a['attributed'] for a in att)
            for k, n in est.items():
                rule_rows.append([short(model), k, f'{n:.0f}', pct(n, tot)])
            rule_rows.append([short(model), f'(re-checked {sampled} of {tot} rejections)', '', ''])
    out = md_table(['model', 'checks', 'rejections', 'tasks with a rejection', 'tasks cut by the limit',
                    'tasks cut by the wall cap', 'check errors', 'generation errors'], rows)
    if rule_rows:
        out += ('\n\nWho rejected (estimated: each task\'s re-checked sample scaled to its rejections; a '
                'pruner rejection can be listed under several rules)\n\n'
                + md_table(['model', 'rejected by', 'rejections (est.)', 'share'], rule_rows))
    return out


def t_limit_curve(runs):
    parts = []
    for model in sorted({m for m, _ in runs}):
        r = runs.get((model, 'chopchop'))
        lc = r and os.path.exists(os.path.join(r['dir'], 'limit_curve.json'))
        if not lc:
            continue
        c = json.load(open(os.path.join(r['dir'], 'limit_curve.json')))
        parts.append(f'**{short(model)}**\n\n' + md_table(
            ['limit s', 'tasks cut', 'pass', 'syntax', 'semantic', 'other', 'total h (1 worker)'],
            [[f"{x['limit_s']:g}", x['cut'], x['passed'], x['syntax'], x['semantic'], x['other'], x['gen_hours']]
             for x in c['rows']]))
    return '\n\n'.join(parts) or '(no limit_curve.json)'


def paired(a, b):
    names = sorted(set(a) & set(b))
    pa = {n for n in names if a[n]['eval']['category'] == 'pass'}
    pb = {n for n in names if b[n]['eval']['category'] == 'pass'}
    return names, pa, pb


def t_models(runs):
    models = sorted({m for m, _ in runs})
    old = [m for m in models if '2.5' in m]
    new = [m for m in models if '3.5' in m]
    if not old or not new:
        return '(needs both models)', None
    m1, m2 = old[0], new[0]
    rows, trans = [], []
    for mode in MODES:
        if (m1, mode) not in runs or (m2, mode) not in runs:
            continue
        a, b = runs[(m1, mode)]['tasks'], runs[(m2, mode)]['tasks']
        names, pa, pb = paired(a, b)
        only_a, only_b = len(pa - pb), len(pb - pa)
        rows.append([mode, len(names), len(pa), len(pb), len(pa & pb), only_a, only_b,
                     f'{mcnemar_exact(only_a, only_b):.3f}'])
        c = Counter((a[n]['eval']['status'] if a[n]['eval']['category'] != 'pass' else 'pass',
                     b[n]['eval']['status'] if b[n]['eval']['category'] != 'pass' else 'pass') for n in names)
        for (x, y), n in sorted(c.items(), key=lambda kv: -kv[1]):
            if x != y:
                trans.append([mode, x, y, n])
    out = md_table(['mode', 'tasks', f'{short(m1)} pass', f'{short(m2)} pass', 'both', f'only {short(m1)}',
                    f'only {short(m2)}', 'McNemar p (exact)'], rows)
    out += (f'\n\nWhere the two models differ: node verdict of {short(m1)} -> {short(m2)} on the same task\n\n'
            + md_table(['mode', short(m1), short(m2), 'tasks'], trans))
    return out, rows


def t_on_off(runs):
    rows = []
    for model in sorted({m for m, _ in runs}):
        if (model, 'itergen_off') not in runs or (model, 'itergen_on') not in runs:
            continue
        a, b = runs[(model, 'itergen_off')]['tasks'], runs[(model, 'itergen_on')]['tasks']
        names, pa, pb = paired(a, b)
        same_text = sum(a[n]['completion'] == b[n]['completion'] for n in names)
        rows.append([short(model), len(names), len(pa), len(pb), len(pa - pb), len(pb - pa),
                     f'{mcnemar_exact(len(pa - pb), len(pb - pa)):.3f}', same_text])
    return md_table(['model', 'tasks', 'off pass', 'on pass', 'only off', 'only on', 'McNemar p (exact)',
                     'identical completion'], rows)


def write_csv(runs, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'mode', 'task', 'category', 'status', 'message', 'region', 'phase', 'wall_s',
                    'tokens', 'generation_error'])
        for (model, mode), r in sorted(runs.items()):
            for n, t in sorted(r['tasks'].items()):
                e = t['eval']
                w.writerow([short(model), mode, n, e['category'], e['status'], e['message'], e['region'],
                            e['phase'], t['wall_s'], t['stats'].get('tokens'), t.get('generation_error')])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root')
    p.add_argument('--commit', default='2c14eef', help="only runs made at this commit ('' = any)")
    p.add_argument('--out', default=os.path.join(REPO, 'reports', 'final'))
    a = p.parse_args()
    runs = load_runs(a.root, a.commit)
    if not runs:
        sys.exit(f'no full runs at commit {a.commit!r} under {a.root}')
    os.makedirs(a.out, exist_ok=True)
    ctrl_md, ctrl = t_controlled(runs)
    pass_md, pass_data = t_pass(runs)
    models_md, models_rows = t_models(runs)
    sections = [
        ('1. Controlled variables', ctrl_md),
        ('2. pass@1 (node)', pass_md),
        ('3. Error subcategories (node error class and message)', t_errors(runs)),
        ('4. Time', t_time(runs)),
        ('5. IterGen semantic checks', t_itergen(runs)),
        ('6. ChopChop checks and pruner', t_chopchop(runs)),
        ('7. ChopChop: results under lower check-CPU limits', t_limit_curve(runs)),
        ('8. Qwen3.5-4B vs Qwen2.5-3B-Instruct (paired)', models_md),
        ('9. IterGen semantic checks on vs off (paired)', t_on_off(runs)),
    ]
    text = '# Final report tables\n\nRuns: ' + ', '.join(
        f'{short(m)}/{mode}' for m, mode in sorted(runs)) + '\n\n' + '\n\n'.join(
        f'## {h}\n\n{b}' for h, b in sections) + '\n'
    open(os.path.join(a.out, 'tables.md'), 'w').write(text)
    json.dump(dict(runs={f'{short(m)} {mode}': r['dir'] for (m, mode), r in runs.items()}, pass_at_1=pass_data,
                   controlled=ctrl, models=models_rows), open(os.path.join(a.out, 'tables.json'), 'w'), indent=1)
    write_csv(runs, os.path.join(a.out, 'tasks.csv'))
    print(text)
