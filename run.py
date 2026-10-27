"""Generate MBPP-JS completions with one model and one decoding mode.

usage:
  python run.py --model Qwen/Qwen2.5-Coder-3B-Instruct --mode itergen_on --out results/q25/itergen_on
  python run.py ... --shard 0/4        # this process does tasks 0, 4, 8, ... (run 4 of them)
  python run.py ... --sample 60        # 60 tasks drawn at random with a fixed seed (pilot)
  python run.py ... --list_todo        # print the names of the tasks not done yet, and exit
  python run.py ... --tasks_file F     # do only the tasks named in F (one per line)

modes:
  unconstrained   plain greedy decoding (raw prompt)
  itergen_off     IterGen, grammar only (raw prompt)
  itergen_on      IterGen + the semantic checks in generators/semantic_checks.py (raw prompt)
  chopchop        ChopChop + the "basic" JS pruner (chat template + rewritten prompt, as
                  the ChopChop integration was designed)

One JSON file per task in --out, written atomically; tasks that already have one are skipped,
so an interrupted run is resumed by running the same command again. --out/run_meta.json
records the settings and the environment; a run refuses to continue into a directory whose
settings differ.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
DATASET = os.path.join(REPO, 'datasets', 'js_prompts_mbpp.jsonl')
ITERGEN_GRAMMAR = os.path.join(REPO, 'generators', 'grammars', 'javascript.lark')
CHOPCHOP_GRAMMAR = os.path.join(REPO, 'generators', 'grammars', 'javascript_chopchop.lark')
MODES = ('unconstrained', 'itergen_off', 'itergen_on', 'chopchop')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--mode', required=True, choices=MODES)
    p.add_argument('--out', required=True)
    p.add_argument('--max_new_tokens', type=int, default=512)
    p.add_argument('--n_tasks', type=int, default=None, help='first N tasks only (smoke test)')
    p.add_argument('--sample', type=int, default=None,
                   help='N tasks drawn at random (seed --seed); the same N tasks for every model')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--shard', default='0/1', help='i/N: do every N-th task starting at i')
    p.add_argument('--check_cpu_limit', type=float, default=1800.0,
                   help='chopchop: per-task limit on check CPU seconds (0 = none)')
    p.add_argument('--wall_cap', type=float, default=0,
                   help='chopchop: per-task wall-clock cap in seconds, a safety net for tasks that '
                        'run for hours with almost no check time (0 = none); recorded per task')
    p.add_argument('--list_todo', action='store_true', help='print the tasks not done yet and exit')
    p.add_argument('--tasks_file', default=None, help='do only the tasks named in this file')
    p.add_argument('--threads', type=int, default=None, help='torch CPU threads')
    return p.parse_args()


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ''


def environment():
    import torch
    import transformers
    cpu = ''
    if os.path.exists('/proc/cpuinfo'):
        cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '')
    try:
        from importlib.metadata import version
        syncode = version('syncode')
    except Exception:
        syncode = None
    return dict(
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cpu=cpu or platform.processor(), n_cpus=os.cpu_count(),
        python=platform.python_version(), torch=torch.__version__,
        transformers=transformers.__version__, syncode=syncode,
        repo_commit=sh('git', '-C', REPO, 'rev-parse', 'HEAD'),
        repo_dirty=bool(sh('git', '-C', REPO, 'status', '--porcelain', '--untracked-files=no')),
    )


def build(args):
    if args.mode == 'unconstrained':
        from generators.hf_generator import HFGenerator
        return HFGenerator(args.model)
    if args.mode.startswith('itergen'):
        from generators.itergen_generator import ItergenGenerator
        return ItergenGenerator(args.model, ITERGEN_GRAMMAR, temperature=0.0,
                                max_new_tokens=args.max_new_tokens)
    from generators.chopchop_generator import ChopchopGenerator
    from generators.javascript_chopchop import (CONSTRUCTORS, JS_START_RULE, JS_CONTEXT,
                                                make_js_pruner, extract_js_prefix, build_js_prompt)
    return ChopchopGenerator(args.model, CHOPCHOP_GRAMMAR, constructors=CONSTRUCTORS,
                             start_rule=JS_START_RULE, pruner_fn=make_js_pruner('basic'),
                             extract_prefix_fn=extract_js_prefix, build_prompt_fn=build_js_prompt,
                             context=JS_CONTEXT,
                             check_cpu_limit_s=args.check_cpu_limit or None,
                             wall_cap_s=args.wall_cap or None)


def generate(gen, mode, task, max_new_tokens):
    """(completion, stats) for one task."""
    stops = task.get('stop_tokens')
    if mode == 'unconstrained':
        out = gen.generate(task['prompt'], stops, temperature=0.0, max_new_tokens=max_new_tokens)
        return out, dict(gen.last_stats)
    if mode.startswith('itergen'):
        out = gen.generate(task['prompt'], stops, temperature=0.0, max_new_tokens=max_new_tokens,
                           semantic_checks=(mode == 'itergen_on'))
        return out, dict(gen.stats, violations=list(gen.violations))
    out = gen.generate(task['prompt'], stop_tokens=stops, temperature=0.0, max_new_tokens=max_new_tokens)
    return out, dict(gen.last_stats or {})


def main():
    args = parse_args()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    shard_i, shard_n = (int(x) for x in args.shard.split('/'))
    os.makedirs(args.out, exist_ok=True)

    tasks = [json.loads(l) for l in open(DATASET) if l.strip()]
    if args.sample:
        import random
        keep = set(random.Random(args.seed).sample(range(len(tasks)), args.sample))
        tasks = [t for i, t in enumerate(tasks) if i in keep]
    if args.n_tasks:
        tasks = tasks[:args.n_tasks]
    if args.list_todo:
        for t in tasks:
            if not os.path.exists(os.path.join(args.out, t['name'] + '.json')):
                print(t['name'])
        return

    settings = dict(model=args.model, mode=args.mode, max_new_tokens=args.max_new_tokens,
                    decoding='greedy', dataset=os.path.relpath(DATASET, REPO),
                    check_cpu_limit_s=(args.check_cpu_limit or None) if args.mode == 'chopchop' else None,
                    sample=args.sample, seed=args.seed if args.sample else None)
    meta_path = os.path.join(args.out, 'run_meta.json')
    env = environment()
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        if old['settings'] != settings:
            sys.exit(f'{args.out} holds outputs made with different settings:\n  {old["settings"]}\n'
                     f'  now {settings}\nUse a new --out directory.')
        if old['env'] != env:
            print(f'NOTE: environment differs from the one recorded in {meta_path}:\n'
                  f'  then {old["env"]}\n  now  {env}', flush=True)
    else:
        # atomic: the other workers may read it at the same moment
        tmp = f'{meta_path}.{shard_i}.tmp'
        json.dump(dict(settings=settings, env=env), open(tmp, 'w'), indent=1)
        os.replace(tmp, meta_path)

    if args.tasks_file:
        mine = set(open(args.tasks_file).read().split())
        tasks = [t for t in tasks if t['name'] in mine]
    todo = [t for i, t in enumerate(tasks) if i % shard_n == shard_i
            and not os.path.exists(os.path.join(args.out, t['name'] + '.json'))]
    print(f'{args.mode} | {args.model} | shard {args.shard}: {len(todo)} tasks to do', flush=True)
    if not todo:
        return

    t0 = time.time()
    gen = build(args)
    print(f'loaded in {time.time() - t0:.0f}s', flush=True)

    import torch
    for k, t in enumerate(todo):
        t1 = time.perf_counter()
        err, stats = None, {}
        try:
            out, stats = generate(gen, args.mode, t, args.max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            traceback.print_exc()
            sys.exit(5)                   # not a property of the task: rerun the command
        except Exception as e:
            out, err = '', f'{type(e).__name__}: {e}'
            traceback.print_exc()
        wall = time.perf_counter() - t1
        if stats.get('generation_error'):
            err = err or stats['generation_error']

        item = dict(name=t['name'], model=args.model, mode=args.mode, prompt=t['prompt'],
                    tests=t['tests'], stop_tokens=t.get('stop_tokens'), completion=out,
                    generation_error=err, wall_s=round(wall, 3), stats=stats)
        path = os.path.join(args.out, t['name'] + '.json')
        with open(path + '.tmp', 'w') as f:
            json.dump(item, f, indent=1)
        os.replace(path + '.tmp', path)
        print(f"[{k + 1}/{len(todo)}] {t['name'][:40]:<41} {wall:7.1f}s"
              + (f"  ERROR {err[:80]}" if err else ''), flush=True)
    print(f'done: {len(todo)} tasks in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
