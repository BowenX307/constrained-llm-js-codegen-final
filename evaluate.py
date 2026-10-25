"""Run every generated program under node and record node's own verdict.

usage:
  python evaluate.py RUN_DIR [RUN_DIR ...]      # writes RUN_DIR/eval.jsonl and prints a summary

The program is prompt + completion + "\\n" + tests, as one .js file. For each task, eval.jsonl
holds:
  status    'OK', node's error class ('SyntaxError', 'AssertionError', 'TypeError', ...),
            'Timeout' (no exit within 15 s), 'GenerationFailed' (the generator raised; there is
            no program), or 'NonzeroExit' (failed without printing an error class)
  category  'pass' | 'syntax' (SyntaxError) | 'semantic' (any other node error, including
            AssertionError = wrong answer) | 'other' (Timeout, GenerationFailed, NonzeroExit)
  message   node's error message (first line), and `message_template` with the quoted names
            and numbers replaced by X, for grouping
  line      the line node points at, and `region`: where that line is -- 'prompt',
            'completion' or 'tests'
  phase     for SyntaxError: 'parse' (the file does not compile) or 'runtime' (thrown while
            running, e.g. by JSON.parse or new RegExp)
Nothing here is our own judgement of the code: every field is read off node's output.
Also reads the older output format ({'completions': [...]}).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

TIMEOUT_S = 15
_ERR = re.compile(r'^(\w*Error)(?: \[[A-Z_]+\])?: ?(.*)$')


def node_version():
    try:
        return subprocess.run(['node', '--version'], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        sys.exit('node is not installed')


def load(path):
    d = json.load(open(path))
    completion = d['completion'] if 'completion' in d else (d.get('completions') or [''])[0]
    return d, completion


def template(msg):
    msg = re.sub(r"'[^']*'|\"[^\"]*\"|`[^`]*`", 'X', msg)
    return re.sub(r'\b\d+(\.\d+)?\b', 'N', msg)


def judge(d, completion):
    rec = dict(name=d['name'], status=None, category=None, message=None, message_template=None,
               line=None, region=None, phase=None)
    if d.get('generation_error'):
        rec.update(status='GenerationFailed', category='other', message=d['generation_error'][:300])
        return rec
    prompt, tests = d['prompt'], d['tests']
    program = prompt + completion + '\n' + tests
    n_prompt = prompt.count('\n')                      # the prompt ends with a newline
    n_code = n_prompt + completion.count('\n') + 1     # last line of prompt + completion
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run(['node', path], capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        rec.update(status='Timeout', category='other')
        return rec
    finally:
        os.remove(path)
    if r.returncode == 0:
        rec.update(status='OK', category='pass')
        return rec
    err = r.stderr
    for l in err.splitlines():
        m = _ERR.match(l.strip())
        if m:
            rec.update(status=m.group(1), message=m.group(2)[:300])
            break
    if rec['status'] is None:
        rec.update(status='NonzeroExit', category='other', message=err.strip().splitlines()[-1][:300] if err.strip() else '')
        return rec
    rec['category'] = 'syntax' if rec['status'] == 'SyntaxError' else 'semantic'
    rec['message_template'] = template(rec['message'])
    # node names the file once per location it reports; the first one is where it points
    m = re.search(re.escape(path) + r':(\d+)', err)
    if m:
        line = int(m.group(1))
        rec['line'] = line
        rec['region'] = 'prompt' if line <= n_prompt else 'completion' if line <= n_code else 'tests'
    if rec['status'] == 'SyntaxError':
        rec['phase'] = 'runtime' if re.search(r'\bat .*' + re.escape(path), err) else 'parse'
    return rec


def evaluate(run_dir):
    files = sorted(f for f in os.listdir(run_dir) if f.endswith('.json') and f != 'run_meta.json')
    items = [load(os.path.join(run_dir, f)) for f in files]
    with ThreadPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(lambda x: judge(*x), items))
    for (d, _), rec in zip(items, recs):
        rec['wall_s'] = d.get('wall_s')
    nv = node_version()
    with open(os.path.join(run_dir, 'eval.jsonl'), 'w') as f:
        for rec in recs:
            f.write(json.dumps(dict(rec, node=nv)) + '\n')
    return recs, nv


def summary(run_dir, recs, nv):
    n = len(recs)
    cat = Counter(r['category'] for r in recs)
    lines = [f'{run_dir}  ({n} tasks, node {nv})',
             f"  pass {cat['pass']}   syntax {cat['syntax']}   semantic {cat['semantic']}   other {cat['other']}"]
    for c in ('syntax', 'semantic', 'other'):
        rs = [r for r in recs if r['category'] == c]
        if not rs:
            continue
        lines.append(f'  {c}:')
        for st, k in Counter(r['status'] for r in rs).most_common():
            lines.append(f'    {st:<18}{k:4d}')
            sub = Counter((r['message_template'] or '') + (f"  [{r['region']}]" if r['region'] else '')
                          for r in rs if r['status'] == st)
            for msg, kk in sub.most_common(6):
                if msg:
                    lines.append(f'        {kk:4d}  {msg[:90]}')
    return '\n'.join(lines)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        recs, nv = evaluate(d)
        print(summary(d, recs, nv), '\n')
