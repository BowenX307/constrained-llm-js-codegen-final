"""Give a finished ChopChop run the unconstrained completion step that the IterGen integration has.

usage:
  python tools/chopchop_fallback.py --model Qwen/Qwen2.5-3B-Instruct --run RUN_DIR --out OUT_DIR
  python tools/chopchop_fallback.py ... --dry_run      # list the tasks it would complete; no model

The IterGen integration (generators/itergen_generator.py, added by Cindy) finishes a function body
that is still open when IterGen stops -- time limit, no progress, or the token budget: greedy
decoding without constraints, at most 128 new tokens, cut where the prompt's brace closes. The
ChopChop integration has no such step, so a task cut by its check limit ends with an incomplete
program. This applies the same step to a finished ChopChop run.

For each task whose body is open, the text ChopChop had accepted when it stopped is rebuilt from
stats.trace (and must reproduce the saved completion), the model continues it from ChopChop's
own input (chat template + rewritten prompt), and the result is cut and stop-trimmed as before.
Decoding is greedy, so this is what a run with the step built in would produce. Like IterGen's
step, the accepted text is re-tokenized as text.

OUT_DIR gets every task of RUN_DIR (tasks with a closed body unchanged) with stats.fallback_used,
and OUT_DIR/fallback.jsonl lists the completed tasks: name, completion, fallback_text, tokens.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from generators.base import BaseGenerator
from generators.chopchop_generator import ChopchopGenerator, replay_trace
from generators.itergen_generator import ItergenGenerator


def accepted_text(t):
    """The text ChopChop had accepted when it stopped, rebuilt from the trace."""
    st = t['stats']
    accepted = st['fixed_prefix']
    for _, _, accepted in replay_trace(st['trace'], st['fixed_prefix']):
        pass
    return accepted


def body_is_open(t):
    depth = t['prompt'].count('{') - t['prompt'].count('}')    # as itergen_generator.py
    return depth > 0 and depth + ItergenGenerator._brace_depth(t['completion']) > 0, depth


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--run', required=True, help='a finished chopchop run folder')
    p.add_argument('--out', required=True)
    p.add_argument('--max_new_tokens', type=int, default=128, help="as IterGen's step")
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()

    tasks = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.run, 'mbpp_*.json')))]
    todo, mismatch = [], []
    for t in tasks:
        is_open, depth = body_is_open(t)
        if not is_open:
            continue
        acc = accepted_text(t)
        if ChopchopGenerator.finish_output(acc, t['stats']['fixed_prefix'], t['stop_tokens']) != t['completion']:
            mismatch.append(t['name'])          # cannot rebuild what ChopChop accepted: leave it
            continue
        todo.append((t, acc, depth))
    print(f'{len(tasks)} tasks, {len(todo)} with an open body to complete'
          + (f', {len(mismatch)} left unchanged (trace does not reproduce the completion): {mismatch}'
             if mismatch else ''), flush=True)
    if a.dry_run:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from generators.javascript_chopchop import JS_CONTEXT, build_js_prompt
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map='auto', torch_dtype=torch.bfloat16)

    os.makedirs(a.out, exist_ok=True)
    for f in ('run_meta.json', 'limit_curve.json'):
        if os.path.exists(os.path.join(a.run, f)):
            shutil.copy(os.path.join(a.run, f), os.path.join(a.out, f))
    done = {}
    for k, (t, acc, depth) in enumerate(todo):
        t0 = time.perf_counter()
        # ChopChop's input (_LocalLanguageModelRunner._tokenize_prompt), then the accepted text
        ids = tok.apply_chat_template(
            [{'role': 'system', 'content': JS_CONTEXT}, {'role': 'user', 'content': build_js_prompt(t['prompt'])}],
            tokenize=True, add_generation_prompt=True, add_special_tokens=False, return_tensors='pt',
            enable_thinking=False)
        if not isinstance(ids, torch.Tensor):
            ids = ids['input_ids']
        ids = torch.cat([ids, tok(acc, add_special_tokens=False, return_tensors='pt')['input_ids']], dim=-1)
        ids = ids.to(model.device)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        new = out[0][ids.shape[-1]:]
        tail = tok.decode(new, skip_special_tokens=True)
        body = ChopchopGenerator.finish_output(acc + tail, t['stats']['fixed_prefix'], None)
        body = ItergenGenerator._trim_at_balanced_brace(None, body, depth)
        completion = BaseGenerator._post_process_stop(None, body, t['stop_tokens'])
        done[t['name']] = dict(name=t['name'], completion=completion, fallback_text=tail,
                               fallback_tokens=len(new), seconds=round(time.perf_counter() - t0, 2))
        print(f'[{k + 1}/{len(todo)}] {t["name"][:40]:<41} {len(new):4d} tokens', flush=True)

    for t in tasks:
        d = done.get(t['name'])
        t['stats']['fallback_used'] = d is not None
        if d:
            t['completion_before_fallback'] = t['completion']
            t['completion'] = d['completion']
            t['stats'].update(fallback_text=d['fallback_text'], fallback_tokens=d['fallback_tokens'],
                              fallback_s=d['seconds'])
        json.dump(t, open(os.path.join(a.out, t['name'] + '.json'), 'w'), indent=1)
    with open(os.path.join(a.out, 'fallback.jsonl'), 'w') as f:
        for d in done.values():
            f.write(json.dumps(d) + '\n')
    print(f'done: {len(done)} completed, written to {a.out}', flush=True)


if __name__ == '__main__':
    main()
