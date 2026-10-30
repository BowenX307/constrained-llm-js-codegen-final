"""Replay IterGen (checks off) on a few tasks and record every token its grammar step rejects.

usage:
  python tools/itergen_rejections.py --model Qwen/Qwen2.5-3B-Instruct --run RUN_DIR --out OUT.jsonl \
      mbpp_230_replace_blank mbpp_285_text_match_two_three ...

IterGen's grammar step (generators/itergen/itergen/main.py:498) takes the model's top token,
asks _is_valid (main.py:526) whether the code is still grammatical with it, and only if not
applies the grammar_strict mask (main.py:92) and picks again. This replays the run with
_is_valid wrapped -- same logic as main.py:539-551, plus a record -- so each rejection shows
the token, where the code was, why it was judged invalid, and the token picked instead.
RUN_DIR is the run the tasks come from; each replayed completion is compared with the saved one
(greedy decoding: they should match).

OUT.jsonl: one line per task: name, same_as_saved, completion, rejections [{token, tail, reason,
chosen}].
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--run', required=True, help='the itergen_off run the tasks come from')
    p.add_argument('--out', required=True)
    p.add_argument('names', nargs='+')
    a = p.parse_args()

    import torch
    from syncode.parse_result import RemainderState
    from generators.itergen_generator import ItergenGenerator
    gen = ItergenGenerator(a.model, os.path.join(REPO, 'generators', 'grammars', 'javascript.lark'),
                           temperature=0.0, max_new_tokens=512)
    ig, tok = gen.itergen, gen.itergen.tokenizer
    log = []

    def is_valid(idx, input_ids, next_token):          # main.py:526-551, recording rejections
        input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=-1)
        partial_code = tok.decode(input_ids[ig.start_from:], skip_special_tokens=True)
        try:
            r = ig.inc_parsers[idx].get_acceptable_next_terminals(partial_code, ig.symbol_pos_maps[idx])
        except Exception as e:
            ok, reason = False, f'parser exception: {type(e).__name__}: {str(e).splitlines()[0][:150] if str(e) else ""}'
        else:
            if r.remainder_state in (RemainderState.COMPLETE, RemainderState.MAYBE_COMPLETE):
                return True
            ok = ig.dfa_mask_store.is_valid_prefix(r)
            reason = (f'unfinished terminal {getattr(r, "remainder", None)!r} '
                      f'({r.remainder_state.name}); is_valid_prefix -> {ok}')
        if not ok:
            log.append(dict(token=tok.decode([int(next_token)]), tail=partial_code[-70:], reason=reason))
        return ok
    ig._is_valid = is_valid

    orig_grammar_step = ig._get_next_token_grammar
    def grammar_step(*args, **kw):
        n = len(log)
        next_token, probs = orig_grammar_step(*args, **kw)
        for e in log[n:]:
            e['chosen'] = tok.decode([int(next_token[0])])
        return next_token, probs
    ig._get_next_token_grammar = grammar_step

    with open(a.out, 'w') as f:
        for name in a.names:
            t = json.load(open(os.path.join(a.run, name + '.json')))
            del log[:]
            out = gen.generate(t['prompt'], t.get('stop_tokens'), temperature=0.0, max_new_tokens=512,
                               semantic_checks=False)
            rec = dict(name=name, same_as_saved=(out == t['completion']), completion=out, rejections=list(log))
            f.write(json.dumps(rec) + '\n')
            print(f'{name}: {len(log)} rejections, same as saved: {rec["same_as_saved"]}', flush=True)
            for e in log:
                print(f'   {e["token"]!r:>12} -> {e.get("chosen")!r:<12} after {e["tail"][-40:]!r} | {e["reason"]}',
                      flush=True)


if __name__ == '__main__':
    main()
