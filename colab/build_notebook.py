"""Build final/colab/run_final.ipynb. usage: python build_nb.py OUT.ipynb REPO_COMMIT"""
import json, sys

OUT, COMMIT = sys.argv[1], sys.argv[2]
cells = []
def md(s): cells.append(dict(cell_type='markdown', metadata={}, source=s.strip('\n')))
def code(s): cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s.strip('\n')))

md(r"""
# Final experiments: one model, one mode per run

**Runtime: G4 GPU** (NVIDIA RTX PRO 6000; Runtime > Change runtime type). Then:

1. Set `MODEL` and `MODE` in section 1.
2. **Runtime > Run all**.

Notes:
- Every final run uses the same runtime type. For ChopChop this matters for the outputs, not
  only the times: its limit is in CPU seconds, so a faster CPU checks more tokens within it.
- `N_TASKS = 20` is a smoke test. Set it to `None` and Run all again for all 397 tasks; the 20 finished tasks are kept.
- ChopChop pilot: `SAMPLE = 60`, `CHECK_CPU_LIMIT = 3600`. Section 5 then shows how the results
  depend on the limit (tools/limit_curve.py), which is how the final limit is chosen. On the
  Qwen2.5 pilot pass@1 stopped changing at 600 s (27/60 from 600 s to 3600 s); the final runs use
  1200 s, and section 5 shows every lower limit for all 397 tasks.
- Outputs are copied to Google Drive as they finish. After a disconnect, Run all again to continue.
- The code is pinned to one commit of
  [constrained-llm-js-codegen-final](https://github.com/BowenX307/constrained-llm-js-codegen-final),
  and outputs go to a folder named after that commit, so two code versions never mix.

| MODE | workers | time limit |
|---|---|---|
| `unconstrained` | 1 | none |
| `itergen_off`, `itergen_on` | 1: IterGen's 60 s per-task limit is wall-clock, so parallel runs would change outputs | 60 s wall |
| `chopchop` | as many as fit: its limit is check CPU time, so outputs do not depend on the worker count | `CHECK_CPU_LIMIT` s of check CPU time |
""")

md("## 1. Settings")
code(rf"""
MODEL      = 'Qwen/Qwen2.5-3B-Instruct'   # or 'Qwen/Qwen3.5-4B'
MODE       = 'unconstrained'              # unconstrained | itergen_off | itergen_on | chopchop
N_TASKS    = 20                           # 20 = smoke test; None = all 397
SAMPLE     = None                         # chopchop pilot: 60 = 60 tasks drawn at random (same for both models)
SEED       = 0
N_WORKERS  = 'auto'                       # 'auto': 1, or for chopchop as many as fit
CHECK_CPU_LIMIT = 1200                    # chopchop: per-task check CPU seconds (from the pilot; pilot used 3600)
EXPECT_GPU = 'RTX PRO 6000'               # G4; every final run uses the same GPU type
REPO_URL    = 'https://github.com/BowenX307/constrained-llm-js-codegen-final.git'
REPO_COMMIT = '{COMMIT}'
""")

md("## 2. Runtime, Drive, code, packages")
code(r"""
import os, sys, json, time, shutil, glob, subprocess, threading
import psutil

assert MODE in ('unconstrained', 'itergen_off', 'itergen_on', 'chopchop'), MODE
REPO   = '/content/final'
CACHE  = '/content/cache'
WORK   = '/content/work'
SLUG   = MODEL.split('/')[-1]
RUN    = (MODE + (f'_lim{CHECK_CPU_LIMIT:g}' if MODE == 'chopchop' else '')
          + (f'_sample{SAMPLE}seed{SEED}' if SAMPLE else '') + '_' + REPO_COMMIT[:7])
OUT    = f'{WORK}/{SLUG}/{RUN}'
DRIVE_ROOT = '/content/drive/MyDrive/constrained-llm-js-codegen'     # same folder as the earlier runs
DRIVE_OUT  = f'{DRIVE_ROOT}/final/{SLUG}/{RUN}'
MS_DRIVE   = f'{DRIVE_ROOT}/mask_store_zst'                          # compressed mask stores, reused
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

smi = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                     capture_output=True, text=True)
if smi.returncode != 0 or not smi.stdout.strip():
    raise RuntimeError('No GPU. Runtime > Change runtime type > G4 GPU, then Run all.')
GPU_NAME, _mb = [x.strip() for x in smi.stdout.strip().splitlines()[0].split(',')]
GPU_GB, RAM_GB, CPUS = int(float(_mb)) / 1024, psutil.virtual_memory().total / 2**30, os.cpu_count()
print(f'GPU {GPU_NAME} ({GPU_GB:.0f} GB) | RAM {RAM_GB:.0f} GB | {CPUS} CPUs')
if EXPECT_GPU not in GPU_NAME:
    raise RuntimeError(f'This runtime has {GPU_NAME}; every final run uses {EXPECT_GPU}. '
                       'Change the runtime type (or EXPECT_GPU, if that is really intended).')

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as e:
    print('Drive not mounted:', type(e).__name__)
DRIVE_OK = os.path.isdir('/content/drive/MyDrive')
if DRIVE_OK:
    os.makedirs(DRIVE_OUT, exist_ok=True)
    restored = 0
    for f in os.listdir(DRIVE_OUT):
        if (f.endswith('.json') or f == 'run_meta.json') and not os.path.exists(f'{OUT}/{f}'):
            shutil.copy(f'{DRIVE_OUT}/{f}', f'{OUT}/{f}'); restored += 1
    print(f'Drive: {restored} files restored from {DRIVE_OUT}')
else:
    print('WARNING: no Drive -- a disconnect that recycles the VM loses the outputs.')

if os.path.isdir(REPO):
    shutil.rmtree(REPO)
for cmd in (['git', 'init', '-q', REPO], ['git', '-C', REPO, 'remote', 'add', 'origin', REPO_URL],
            ['git', '-C', REPO, 'fetch', '-q', '--depth', '1', 'origin', REPO_COMMIT],
            ['git', '-C', REPO, 'checkout', '-q', 'FETCH_HEAD']):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(' '.join(cmd) + ' failed:\n' + r.stderr)
print('code at', REPO_COMMIT[:7])

def pip(*a):
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *a], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('pip install failed: ' + ' '.join(a) + '\n' + r.stderr[-2000:])
def importable(stmt):
    return subprocess.run([sys.executable, '-c', stmt], cwd=REPO, capture_output=True).returncode == 0

import transformers
if transformers.__version__ != '5.20.1':
    pip('transformers==5.20.1')
    raise RuntimeError(f'transformers was {transformers.__version__}; installed 5.20.1. '
                       'Runtime > Restart session, then Run all.')
pip('lark', 'regex', 'networkx', 'fire', 'interegular', 'tqdm', 'datasets', 'jsonschema', 'psutil')
if not importable('from syncode.parsers.itergen_parser import IGParser'):
    # GitHub, pinned, --no-deps (so it cannot replace torch / transformers)
    pip('--no-deps', '--ignore-requires-python', '--force-reinstall',
        'git+https://github.com/structuredllm/syncode.git@4d6c110a028d3fb9cbb5f1a6ac6446b696f3eac4')
for tool, pkg in (('zstd', 'zstd'), ('node', 'nodejs')):
    if not shutil.which(tool):
        subprocess.run(['apt-get', 'install', '-y', '-qq', pkg], capture_output=True)
import torch
print('transformers', transformers.__version__, '| torch', torch.__version__,
      '| node', subprocess.run(['node', '--version'], capture_output=True, text=True).stdout.strip())
ENV = dict(os.environ, SYNCODE_CACHE=CACHE + '/syncode/', HF_HOME=CACHE + '/hf_home',
           TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', PYTHONPATH=REPO,
           HF_HUB_DISABLE_PROGRESS_BARS='1')
""")

md(r"""
## 3. Grammar mask store (IterGen modes only)

IterGen needs a mask store for the grammar and the model's tokenizer. Two are already on Drive
from the earlier runs: Qwen2.5 (its tokenizer is identical to Qwen2.5-Coder's) and Qwen3.5.
This cell restores the right one in about a minute. If none is there, it builds one, which takes
about 30 min, and saves a compressed copy to Drive.
""")
code(r"""
if MODE.startswith('itergen'):
    GRAMMAR = f'{REPO}/generators/grammars/javascript.lark'
    probe = subprocess.run([sys.executable, '-c',
        'import syncode.common as c; from syncode.parsers.grammars import Grammar\n'
        f'tok = c.load_tokenizer({MODEL!r}); g = Grammar({GRAMMAR!r})\n'
        'print("MS_PATH=" + c.SYNCODE_CACHE + "mask_stores/" + type(tok).__name__ + '
        '"/grammar_strict_" + g.hash() + "_" + str(tok.vocab_size) + ".pkl")'],
        env=ENV, cwd=REPO, capture_output=True, text=True)
    lines = [l for l in probe.stdout.splitlines() if l.startswith('MS_PATH=')]
    if probe.returncode != 0 or not lines:
        raise RuntimeError('could not resolve the mask store path:\n' + probe.stderr[-2000:])
    MS_PATH = lines[-1][len('MS_PATH='):]
    MS_ZST = f'{MS_DRIVE}/{os.path.basename(MS_PATH)}.zst'
    os.makedirs(os.path.dirname(MS_PATH), exist_ok=True)
    print('mask store:', os.path.basename(MS_PATH))
    t0 = time.time()
    if not (os.path.exists(MS_PATH) and os.path.getsize(MS_PATH) > 100 * 2**20):
        if DRIVE_OK and os.path.exists(MS_ZST):
            r = subprocess.run(['zstd', '-q', '-d', '-f', '-T0', MS_ZST, '-o', MS_PATH + '.part'],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError('the copy on Drive is unreadable: ' + r.stderr[-300:])
            os.replace(MS_PATH + '.part', MS_PATH)
            print(f'restored from Drive in {time.time() - t0:.0f}s')
        else:
            print('not on Drive: building it (~30 min) ...')
            r = subprocess.run([sys.executable, '-c',
                'import syncode.common as c\nfrom syncode.parsers.grammars import Grammar\n'
                'from syncode.mask_store.mask_store import MaskStore\n'
                f'tok = c.load_tokenizer({MODEL!r})\n'
                f"MaskStore.init_mask_store(grammar=Grammar({GRAMMAR!r}), tokenizer=tok, use_cache=True, mode='grammar_strict')\n"],
                env=ENV, cwd=REPO, capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(MS_PATH):
                raise RuntimeError(f'build failed (exit {r.returncode}; -9 = out of RAM):\n' + r.stderr[-2000:])
            print(f'built in {(time.time() - t0) / 60:.0f} min')
            if DRIVE_OK:
                os.makedirs(MS_DRIVE, exist_ok=True)
                z = subprocess.run(['zstd', '-q', '-f', '-T0', '-3', MS_PATH, '-o', MS_ZST + '.part'],
                                   capture_output=True, text=True)
                if z.returncode == 0:
                    os.replace(MS_ZST + '.part', MS_ZST)
                    print(f'saved to Drive: {os.path.getsize(MS_ZST) / 2**30:.2f} GB')
                else:
                    print('*** could not save it to Drive (full?):', z.stderr[-200:])
    print(f'mask store ready: {os.path.getsize(MS_PATH) / 2**30:.2f} GB')
else:
    print(MODE, 'does not use a mask store')
""")

md(r"""
## 4. Generate

Each worker is `python run.py --shard i/N`. A worker that crashes is restarted; `run.py` skips
the tasks that are already done. Finished outputs are copied to Drive every minute.
""")
code(r"""
import selectors

if N_WORKERS != 'auto':
    NW = int(N_WORKERS)
elif MODE == 'chopchop':
    # each worker holds its own copy of the model (~7-9 GB of GPU memory) and needs ~1 CPU
    NW = max(1, min(int(GPU_GB // 10), CPUS - 1, int(RAM_GB // 8)))
else:
    NW = 1
THREADS = max(1, CPUS // NW)
STALL_S = 3 * 3600 if MODE == 'chopchop' else 1800   # a worker silent this long is stuck
print(f'{MODE}: {NW} worker(s), {THREADS} CPU threads each')

def n_done():
    return len([f for f in os.listdir(OUT) if f.startswith('mbpp_') and f.endswith('.json')])
N_TOTAL = min(x for x in (SAMPLE, N_TASKS, 397) if x)

stop_sync = threading.Event()
def copy_to_drive():
    if not DRIVE_OK:
        return
    for f in os.listdir(OUT):
        if f.endswith('.json') or f.endswith('.jsonl'):
            src, dst = f'{OUT}/{f}', f'{DRIVE_OUT}/{f}'
            try:
                if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                    shutil.copyfile(src, dst)
            except Exception as e:
                print('Drive copy failed:', type(e).__name__, e)
def sync_to_drive():
    while not stop_sync.wait(60):
        copy_to_drive()

def launch(i):
    cmd = [sys.executable, '-u', f'{REPO}/run.py', '--model', MODEL, '--mode', MODE, '--out', OUT,
           '--shard', f'{i}/{NW}', '--threads', str(THREADS), '--check_cpu_limit', str(CHECK_CPU_LIMIT)]
    if N_TASKS:
        cmd += ['--n_tasks', str(N_TASKS)]
    if SAMPLE:
        cmd += ['--sample', str(SAMPLE), '--seed', str(SEED)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env=ENV, cwd=REPO)
    sel.register(p.stdout, selectors.EVENT_READ, i)
    return p

sync = threading.Thread(target=sync_to_drive, daemon=True); sync.start()
sel = selectors.DefaultSelector()
procs, last, restarts, fatal = {}, {}, {i: 0 for i in range(NW)}, None
t0, done0 = time.time(), n_done()
for i in range(NW):
    procs[i] = launch(i); last[i] = time.time()
try:
    while procs:
        for key, _ in sel.select(timeout=10):
            i, line = key.data, key.fileobj.readline()
            if not line:
                sel.unregister(key.fileobj); continue
            last[i] = time.time()
            if line.startswith('[') and '/' in line.split(']')[0]:
                d = n_done(); rate = (time.time() - t0) / max(1, d - done0)
                print(f'w{i} {line.rstrip()}   | {d}/{N_TOTAL}, ETA {rate * (N_TOTAL - d) / 60:.0f} min', flush=True)
            elif not any(s in line for s in ('Loading weights', 'Warning', 'warn(', 'torch_dtype')):
                print(f'w{i}: {line.rstrip()}', flush=True)
        for i, p in list(procs.items()):
            if p.poll() is None and time.time() - last[i] > STALL_S:
                print(f'!! worker {i} silent for {STALL_S} s: killing it'); p.kill(); p.wait()
            rc = p.poll()
            if rc is None:
                continue
            del procs[i]
            if rc == 0:
                continue
            restarts[i] += 1
            print(f'!! worker {i} exited with {rc} (restart {restarts[i]})', flush=True)
            if restarts[i] > 5:
                fatal = f'worker {i} failed more than 5 times'; continue
            procs[i] = launch(i); last[i] = time.time()
finally:
    for p in procs.values():
        p.terminate()
    stop_sync.set(); sync.join()
    copy_to_drive()                                  # whatever finished since the last minute
print(f'\n{n_done()}/{N_TOTAL} done ({(time.time() - t0) / 60:.0f} min this session)')
if fatal:
    raise RuntimeError(fatal)
""")

md("## 5. Evaluate with node")
code(r"""
r = subprocess.run([sys.executable, f'{REPO}/evaluate.py', OUT], capture_output=True, text=True, cwd=REPO)
print(r.stdout[-6000:], r.stderr[-2000:])
recs = [json.loads(l) for l in open(f'{OUT}/eval.jsonl')]
stats = {}
for f in glob.glob(f'{OUT}/mbpp_*.json'):
    d = json.load(open(f)); stats[d['name']] = d
import statistics
walls = [d['wall_s'] for d in stats.values()]
print(f'time: total {sum(walls) / 3600:.2f} h, median {statistics.median(walls):.1f} s per task')
if MODE == 'itergen_on':
    v = [x for d in stats.values() for x in d['stats'].get('violations', [])]
    print(f"checks fired {len(v)} times on {sum(1 for d in stats.values() if d['stats'].get('violations'))} tasks:",
          dict(__import__('collections').Counter(x['rule'] for x in v)))
if MODE.startswith('itergen'):
    print('stop reasons:', dict(__import__('collections').Counter(d['stats'].get('stop_reason') for d in stats.values())),
          '| fallback used:', sum(d['stats'].get('fallback_used', False) for d in stats.values()))
if MODE == 'chopchop':
    s = [d['stats'] for d in stats.values()]
    print('checks', sum(x['checks'] for x in s), '| rejected', sum(x['rejected'] for x in s),
          '| CPU limit hit', sum(x.get('limit_hit', False) for x in s),
          '| check errors', sum(x.get('check_errors', 0) for x in s))
    # results under every lower limit, rebuilt from the per-check trace
    r = subprocess.run([sys.executable, f'{REPO}/tools/limit_curve.py', OUT], capture_output=True, text=True, cwd=REPO)
    print(r.stdout[-4000:], r.stderr[-2000:])
copy_to_drive()
""")

md(r"""
## 5b. ChopChop: who rejected each token (CPU only)

For every rejected token, re-checks the same text with the grammar alone and with each pruner
rule alone (tools/attribute_rejections.py). Runs on all CPUs after generation; it does not
change any output. At most 30 evenly spaced rejections per task are re-checked (all of them
would take hours for a task with thousands). Skip it with `ATTRIBUTE = False` -- it can be
run later on any machine.
""")
code(r"""
ATTRIBUTE = True
if MODE == 'chopchop' and ATTRIBUTE:
    p = subprocess.Popen([sys.executable, '-u', f'{REPO}/tools/attribute_rejections.py', OUT],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO, env=ENV)
    for line in p.stdout:
        print(line.rstrip(), flush=True)
    p.wait()
    copy_to_drive()
""")

md("## 6. Download (optional; everything is also on Drive)")
code(r"""
z = shutil.make_archive(f'/content/{SLUG}_{MODE}_{REPO_COMMIT[:7]}', 'zip', OUT)
print(z, f'{os.path.getsize(z) / 2**20:.1f} MB')
try:
    from google.colab import files; files.download(z)
except Exception as e:
    print('download did not start:', type(e).__name__, '-- the files are on Drive:', DRIVE_OUT)
""")

nb = dict(cells=cells, metadata=dict(accelerator='GPU', colab=dict(provenance=[]),
                                     kernelspec=dict(display_name='Python 3', name='python3'),
                                     language_info=dict(name='python')), nbformat=4, nbformat_minor=0)
for c in nb['cells']:
    src = c['source']
    c['source'] = [l + '\n' for l in src.split('\n')[:-1]] + [src.split('\n')[-1]]
json.dump(nb, open(OUT, 'w'), indent=1)
print('wrote', OUT)
