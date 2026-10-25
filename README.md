# Constrained decoding for JavaScript code generation: final experiments

This repo compares four ways to generate JavaScript with a code LLM on MBPP-JS (397 tasks from
MultiPL-E). Every run uses greedy decoding with `max_new_tokens = 512`.

| mode | what it is | prompt |
|---|---|---|
| `unconstrained` | plain greedy decoding | raw (comment + function header, the model continues) |
| `itergen_off` | IterGen, grammar constraint only | raw |
| `itergen_on` | IterGen + semantic checks with backtracking (`generators/semantic_checks.py`) | raw |
| `chopchop` | ChopChop + the `basic` JS pruner | chat template + rewritten task, as the ChopChop integration was designed |

`datasets/js_prompts_mbpp.jsonl` is the MBPP-JS prompt set from
[MultiPL-E](https://github.com/nuprl/MultiPL-E), which is built on MBPP (CC BY 4.0).

## Running

```
pip install -r requirements.txt
python run.py --model Qwen/Qwen2.5-Coder-3B-Instruct --mode itergen_on --out results/q25/itergen_on
```

The run writes one JSON file per task to `--out`, and re-running the same command resumes it.
`--shard i/N` splits a run over N processes. `--n_tasks 20` runs a smoke test.
`--out/run_meta.json` records the settings and the environment (GPU, CPU, versions, commit).

## What each output records

Every task file has `completion`, `generation_error`, `wall_s`, and a `stats` dict that depends on
the mode:

- **unconstrained**: `tokens`.
- **IterGen**:
  - `check_s` is the time spent in the semantic checks.
  - `violations` lists every check that fired: the rule, the reason, the position, and whether
    IterGen gave up.
  - `backward_calls` and `removed` show what each backtrack cut.
  - `stop_reason` is one of `function_closed`, `stop_token`, `no_progress`, `time_limit`,
    `gave_up`, `forward_error`, or `max_steps`.
  - `fallback_used` and `fallback_text` record the unconstrained 128-token completion that runs
    when the constrained loop stops with the function still open.
  - `mask_applied` and `mask_empty` count grammar masks. An empty mask means the constraint was
    off for that token.
  - `forward_errors`, `reprefills`, and `tokens`.
- **ChopChop**:
  - `checks`, `check_cpu_s`, and `check_wall_s`.
  - `rejected`, split into `rejected_by_grammar` and `rejected_by_pruner`, with
    `rejected_by_rule` per pruner rule (`undeclared`, `break_outside_loop`, `no_eval`).
  - `rejection_samples`.
  - `limit_hit`, `check_errors`, `llm_finished`, and `tokens`.

Rejection attribution re-checks a rejected prefix with the grammar alone, then with one pruner
rule at a time. It runs only on rejections and never changes a decision. Its time is reported
separately as `attribution_wall_s`.

## Time limits

- **IterGen**: a 60 s wall-clock limit per task. After it, an unconstrained 128-token completion
  closes the function. This is unchanged from the original integration.
- **ChopChop**: a per-task limit on check time, measured as CPU time (`--check_cpu_limit`,
  default 1800 s). When it is reached, generation stops at the last accepted token, so every
  emitted token was checked.
  - This replaces an earlier 120 s budget after which tokens were accepted unchecked.
  - Because the limit is CPU time, parallel workers give the same outputs as a single worker.

## Where the code comes from, and what was changed

- **`generators/itergen/`**: IterGen ([uiuc-arc/itergen](https://github.com/uiuc-arc/itergen),
  MIT), ported by the original project to the current `syncode` API. One fix was made in
  `_backtrack_till_char_pos`: `backward()` no longer drops the prompt from the generation.
- **`generators/chopchop/`**: ChopChop ([timothytmzhou/chopchop](https://github.com/timothytmzhou/chopchop),
  MIT), core library only (`core/`, `llm/`). Two small changes came from the original project:
  - `lru_cache.py` uses a Python < 3.12 compatible generic syntax. Behaviour is unchanged.
  - `run_llm.py` adds a `max_new_tokens` cap.
- **The JS grammars, the JS pruner (`javascript_chopchop.py`), and the three generator wrappers**:
  these come from the original project (CindyZhou2003/constrained-llm-js-codegen). Changes here:
  - `javascript.lark` accepts more than one top-level item (`top_level+`).
  - `semantic_checks.py` is the IterGen checks, moved out of the generator. Two rules were fixed:
    - `orphan_break` never fired.
    - `const_reassign` missed `++`/`--`.
  - `kv_cache.py` provides a rewindable KV cache, used by both tools, for transformers 5.x and
    hybrid models.
  - The ChopChop wrapper uses the CPU-time limit and rejection attribution described above.
    - The JS pruner gained per-rule switches for the attribution. With all rules on, which is
      always the case outside attribution, its decisions are unchanged.
    - The pruner's known false rejections were deliberately **not** fixed: parenthesised arrow
      parameters, `new X()` declarations, hoisting, block `var`, and globals missing from its
      list.
  - The unconstrained generator stops as soon as the post-processing would cut the text. The
    kept output is identical; this only avoids wasted tokens, which matters for timing.
  - All the instrumentation listed above.
