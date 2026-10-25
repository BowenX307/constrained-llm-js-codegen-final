import re
import time
import torch
from transformers import StoppingCriteria
from .itergen.itergen.main import IterGen
from .base import BaseGenerator
from . import semantic_checks
from .kv_cache import RewindableCache, text_config


class _BraceDepthStop(StoppingCriteria):
    """Stop forward() as soon as the prompt's outer `{` is balanced.

    Without this, after the model emits the function's closing `}`,
    forward(unit="statement", num=1) keeps generating: the `}` ends a
    compound_statement, not a new top-level statement, and comments are
    %ignore'd -- so the unit never fires and the whole max_new_tokens
    budget is burned on garbage that post-processing throws away (49s
    wasted on the 0.5B model, minutes on the 3B).
    """
    def __init__(self, itergen_ref, prompt_len, depth_offset):
        self.itergen = itergen_ref
        self.prompt_len = prompt_len
        self.depth_offset = depth_offset

    def __call__(self, input_ids, scores, **kwargs):
        gen = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
        body = gen[self.prompt_len:] if len(gen) > self.prompt_len else ""
        depth = self.depth_offset + body.count('{') - body.count('}')
        stop = depth <= 0
        return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)

class ItergenGenerator(BaseGenerator):
    def __init__(self, model_name: str, grammar: str, **kwargs):

        temp= kwargs.get("temperature")
        itergen_params = {
            "model_id": model_name,
            "grammar": grammar,
            "parse_output_only": False,
            "recurrence_penalty": 0.0,
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "do_sample": temp > 0  # if temperature > 0, enable sampling; otherwise, use greedy decoding
        }
        if temp > 0:
            itergen_params["temperature"] = temp
            
        self.itergen = IterGen(
            **itergen_params
        )
        # dev_mode=True (the IterGen default) causes grammar parsing exceptions
        # to be re-raised from forward(), which makes the outer loop break on any
        # incremental parse failure (e.g. partial regex literals, complex
        # expressions). Disable it so failures are handled gracefully instead.
        self.itergen.dev_mode = False

        # IterGen targets transformers 4.53.2, which defaulted these to ints; newer versions
        # leave them None and IterGen compares them as ints. Greedy, one output.
        for f in ('num_return_sequences', 'num_beams', 'num_beam_groups'):
            if getattr(self.itergen.generation_config, f, None) is None:
                setattr(self.itergen.generation_config, f, 1)
        self._text_config = text_config(self.itergen.model)

        # Instrumentation only -- none of these wrappers changes what IterGen does.
        # An EMPTY grammar mask means "no token is grammatical here"; IterGen then masks
        # nothing, so the constraint is off for that token. Count both outcomes.
        orig_apply_mask = self.itergen._apply_mask
        def counting_apply_mask(idx, accept_mask, scores):
            self.stats['mask_applied' if bool(accept_mask.any()) else 'mask_empty'] += 1
            return orig_apply_mask(idx, accept_mask, scores)
        self.itergen._apply_mask = counting_apply_mask

        # generate() wraps forward() in `except Exception: break`, which silently ends the
        # constrained part of the generation. Record each failure.
        orig_forward = self.itergen.forward
        def recording_forward(*a, **k):
            try:
                return orig_forward(*a, **k)
            except Exception as e:
                self.stats['forward_errors'].append(f'{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ""}')
                raise
        self.itergen.forward = recording_forward
        self.stats = self._new_stats()

    @staticmethod
    def _new_stats():
        """Per-generation record, read by the caller after generate()."""
        return {
            'check_s': 0.0,            # time in view() + the semantic rules
            'backward_calls': 0,
            'removed': [],             # the last 80 chars before each backward(): what was cut
            'stop_reason': None,       # why the constrained loop ended (see generate())
            'fallback_used': False,    # the unconstrained 128-token completion ran
            'fallback_text': '',       # what it appended
            'forward_errors': [],
            'mask_applied': 0,
            'mask_empty': 0,
            'reprefills': 0,           # hybrid-cache rewinds (see kv_cache.py)
            'tokens': None,            # tokens IterGen generated (its own counter)
        }

    def generate(self, prompt: str, stop_tokens, **kwargs) -> str:
        self.stats = self._new_stats()
        RewindableCache.reprefills = 0
        try:
            return self._generate(prompt, stop_tokens, **kwargs)
        finally:
            self.stats['reprefills'] = RewindableCache.reprefills
            self.stats['tokens'] = (getattr(self.itergen, '_metadata', None) or {}).get('total_tokens')
            self.stats['check_s'] = round(self.stats['check_s'], 3)

    def _generate(self, prompt: str, stop_tokens, **kwargs) -> str:

        temp = kwargs.get("temperature")

        # Do NOT pass max_new_tokens to forward(): start() already set max_length as an
        # absolute token-count ceiling (prompt_len + max_new_tokens). Re-passing it each
        # call would trigger generation_config.update(max_new_tokens=N) which does NOT
        # recalculate max_length, so the ceiling never slides and long functions get
        # truncated once the early statements consume the budget.
        forward_params = {
            "do_sample": temp > 0,
        }
        if temp > 0:
            forward_params["temperature"] = temp

        self.itergen.start(prompt=prompt)

        # IterGen.start() resets the incremental parsers and the decoding trace, but it
        # never resets symbol_pos_maps (main.py:99 creates them once in __init__). Since
        # one generator is reused for every task, symbol positions from earlier prompts
        # stay in the map, and view() then slices the *new* text with *old* offsets --
        # which is where the garbage fragments come from. Start each task with a clean map.
        self.itergen.symbol_pos_maps = [
            type(spm)() for spm in self.itergen.symbol_pos_maps
        ]
        # start() builds a plain DynamicCache(), which transformers 5.x cannot crop the way
        # IterGen asks and a hybrid model cannot use at all (see kv_cache.py).
        self.itergen.model_kwargs['past_key_values'] = RewindableCache(self._text_config)

        # The old default ran hand-written semantic checks after every generated
        # statement and backtracked on failures. In practice the symbol map can
        # expose partial/stale identifier slices while IterGen is recovering from
        # an incremental parse failure, so the checks falsely rejected fragments
        # such as "fo", "ction", or "ing" as undeclared variables. That caused
        # short completions, repeated backtracking, wall-clock timeouts, and lower
        # pass rates than the unconstrained baseline. Keep the experimental checks
        # available for debugging, but make grammar-only IterGen the default.
        enable_semantic_checks = bool(kwargs.get("semantic_checks", False))
        tracking_categories = list(semantic_checks.TRACKED_CATEGORIES) if enable_semantic_checks else []
        # Which rules to run, in order; the first one that fires wins.
        semantic_rules = tuple(kwargs.get("semantic_rules", semantic_checks.DEFAULT_RULES))
        # One record per backtrack: which rule fired and why. Read it after generate()
        # to attribute what the checks did to individual rules.
        self.violations = []

        # Names always in scope, plus the prompted function's own name and parameters
        # (e.g. "function pancake_sort(nums){" -> pancake_sort, nums).
        base_identifiers = set(semantic_checks.BASE_IDENTIFIERS) | semantic_checks.prompt_identifiers(prompt)

        # Count opening braces in prompt to track when the function body is complete
        prompt_brace_depth = prompt.count('{') - prompt.count('}')

        # Add a stopping criterion that cuts forward() short the moment brace depth
        # hits 0 -- otherwise the model wastes the rest of its token budget on
        # %ignore'd comments after the function close (see _BraceDepthStop docstring).
        if prompt_brace_depth > 0:
            self.itergen.stopping_criteria.append(
                _BraceDepthStop(self.itergen, len(prompt), prompt_brace_depth)
            )

        # Loop until the model hits its own token budget (enforced by start()'s max_length)
        # or our early-termination checks fire. Using a large fixed bound avoids the old
        # bug where max_steps == max_new_tokens caused premature exit: each forward() call
        # consumes *multiple* tokens (a whole statement), so 512 steps ≠ 512 tokens.
        max_steps = 4096

        prev_gen_len = len(prompt)  # track generation length to detect stalls

        # Detect infinite backtrack loops: if backward() returns us to the same
        # position repeatedly, the next forward() is regenerating the same invalid
        # statement (typical under greedy decoding with recurrence_penalty=1.0).
        last_backtrack_pos = -1
        consecutive_backtracks = 0
        # 5 retries gives recurrence_penalty=0.0 enough chances to diverge to a
        # different token path after backward(); 3 was too tight for cases where
        # the alternative paths also briefly look invalid before resolving.
        MAX_CONSECUTIVE_BACKTRACKS = 5

        # Hard wall-clock cap per task. A normal task is ~2s; anything past 60s
        # almost certainly means we're churning on slow validation or grammar masks
        # with no real progress. Bail with whatever has been generated so far.
        MAX_GENERATE_SECONDS = 60.0
        start_time = time.perf_counter()

        for step in range(max_steps):

            if time.perf_counter() - start_time > MAX_GENERATE_SECONDS:
                print(f"[itergen] wall-clock timeout at step {step} ({MAX_GENERATE_SECONDS:.0f}s)")
                self.stats['stop_reason'] = 'time_limit'
                break

            pre_counts = {}

            if enable_semantic_checks:
                t_check = time.perf_counter()
                for cat in tracking_categories:
                    try:
                        res = self.itergen.view(unit=cat)
                        pre_counts[cat] = len(res[0]) if res and res[0] else 0
                    except Exception:
                        pre_counts[cat] = 0
                self.stats['check_s'] += time.perf_counter() - t_check

            # forward 1 step
            try:
                self.itergen.forward(unit="statement", num=1, **forward_params)
            except Exception:
                self.stats['stop_reason'] = 'forward_error'
                break

            current_code = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""

            # If nothing was generated this step (EOS/max_length already reached), stop.
            # This prevents an infinite loop when the model is done but 'statement' never
            # completes (e.g. the model only emitted comments, which %ignore discards).
            current_len = len(current_code) if current_code.startswith(prompt) else len(prompt) + len(current_code)
            if current_len <= prev_gen_len:
                self.stats['stop_reason'] = 'no_progress'
                break
            prev_gen_len = current_len

            # --- Early termination checks ---
            generated_so_far = current_code[len(prompt):] if current_code.startswith(prompt) else current_code
            
            # Check 1: Stop tokens — if any stop token appears in generated text, stop immediately.
            # Skip comment-style stop tokens (\n// and \n/*) when inside a function body
            # (brace depth > 0), since the grammar already prevents new function declarations
            # there and IterGen allows comments anywhere via %ignore.
            if stop_tokens and generated_so_far:
                should_stop = False
                gen_depth = prompt_brace_depth + generated_so_far.count('{') - generated_so_far.count('}')
                for stop in stop_tokens:
                    if stop in generated_so_far:
                        if stop in ('\n//', '\n/*') and gen_depth > 0:
                            continue  # Comments inside function body are fine for grammar-constrained gen
                        should_stop = True
                        break
                if should_stop:
                    self.stats['stop_reason'] = 'stop_token'
                    break

            # Check 2: Brace depth — if the function body is complete (all braces balanced), stop
            if generated_so_far and prompt_brace_depth > 0:
                gen_open = generated_so_far.count('{')
                gen_close = generated_so_far.count('}')
                current_depth = prompt_brace_depth + gen_open - gen_close
                if current_depth <= 0:
                    self.stats['stop_reason'] = 'function_closed'
                    break

            if not enable_semantic_checks:
                continue

            t_check = time.perf_counter()
            post_items = {}
            for cat in tracking_categories:
                try:
                    res = self.itergen.view(unit=cat)
                    post_items[cat] = res[0] if res and res[0] else []
                except Exception:
                    post_items[cat] = []
            
            # semantic validation rules (generators/semantic_checks.py)
            full_gen_so_far = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
            spm = self.itergen.symbol_pos_maps[0]
            post_positions = {cat: list(spm.get_symbol_pos_all(cat)) for cat in tracking_categories}
            ctx = semantic_checks.build_context(generated_so_far, full_gen_so_far, post_items,
                                                pre_counts, base_identifiers, post_positions)
            violation = semantic_checks.first_violation(ctx, semantic_rules)
            self.stats['check_s'] += time.perf_counter() - t_check
            is_valid = violation is None
            violation_reason = violation[1] if violation else ""
            if violation:
                self.violations.append({"rule": violation[0], "reason": violation[1],
                                        "at": len(generated_so_far), "gave_up": False})

            # backtrack if any violation is found
            if not is_valid:
                self.stats['backward_calls'] += 1
                self.stats['removed'].append(full_gen_so_far[-80:])
                self.itergen.backward(unit="statement", num=1)
                # Reset prev_gen_len after backtracking so the stall detector stays accurate.
                backtracked_code = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
                new_pos = len(backtracked_code)
                # If we keep landing at the same backtrack position, the model is
                # regenerating the identical invalid statement -- abort instead of spinning.
                if new_pos == last_backtrack_pos:
                    consecutive_backtracks += 1
                    if consecutive_backtracks >= MAX_CONSECUTIVE_BACKTRACKS:
                        tail = backtracked_code[-80:] if backtracked_code else ""
                        print(f"[itergen] abort: {consecutive_backtracks}x stuck @pos={new_pos} | reason={violation_reason!r} | tail={tail!r}")
                        self.violations[-1]["gave_up"] = True
                        self.stats['stop_reason'] = 'gave_up'
                        break
                else:
                    last_backtrack_pos = new_pos
                    consecutive_backtracks = 1
                prev_gen_len = new_pos
                continue

            # Successful step -- reset the backtrack-loop tracker.
            last_backtrack_pos = -1
            consecutive_backtracks = 0
        else:
            self.stats['stop_reason'] = 'max_steps'

        full_text = self.itergen.structured_gen[0] if self.itergen.structured_gen else ""
        if full_text.startswith(prompt):
            generated_for_depth = full_text[len(prompt):]
        else:
            generated_for_depth = full_text

        # If IterGen stops on a timeout, parser exception, or model budget before
        # closing the prompted function body, the benchmark sees a SyntaxError
        # even when the prefix is very close to the unconstrained greedy answer.
        # Use the already-loaded model to greedily finish only those incomplete
        # bodies, then trim as soon as the original prompt brace is balanced.
        if prompt_brace_depth > 0 and prompt_brace_depth + self._brace_depth(generated_for_depth) > 0:
            self.stats['fallback_used'] = True
            before_fallback = full_text
            full_text = self._complete_unconstrained(
                full_text=full_text,
                prompt=prompt,
                stop_tokens=stop_tokens,
                prompt_brace_depth=prompt_brace_depth,
                temperature=temp,
                max_new_tokens=kwargs.get("itergen_fallback_max_new_tokens", 128),
            )
            self.stats['fallback_text'] = (full_text[len(before_fallback):]
                                           if full_text.startswith(before_fallback) else full_text)

        generated_only = ""
        if full_text.startswith(prompt):
            generated_only = full_text[len(prompt):]
        else:
            # If prompt is not a prefix for some reason, return empty or full text depending on logic.
            # Usually safe to return full_text if it doesn't align, or just empty.
            # But let's try to return full text if prompt checking fails to be safe, though unexpected.
            generated_only = full_text 
            
        return self._post_process_stop(generated_only, stop_tokens, prompt_brace_depth)

    def _post_process_stop(self, text: str, stop_tokens, prompt_brace_depth: int = 0) -> str:
        # Skip \n// and \n/* when still inside a function body (brace depth > 0):
        # grammar-constrained gen allows comments anywhere via %ignore, so these
        # don't signal a new top-level declaration.
        if not stop_tokens:
            return text
        min_stop_index = len(text)
        found = False
        for stop in stop_tokens:
            idx = text.find(stop)
            if idx == -1:
                continue
            if stop in ('\n//', '\n/*'):
                depth = prompt_brace_depth + text[:idx].count('{') - text[:idx].count('}')
                if depth > 0:
                    continue
            min_stop_index = min(min_stop_index, idx)
            found = True
        return text[:min_stop_index] if found else text

    def _complete_unconstrained(
        self,
        full_text: str,
        prompt: str,
        stop_tokens,
        prompt_brace_depth: int,
        temperature: float,
        max_new_tokens: int = 128,
    ) -> str:
        if max_new_tokens <= 0:
            return full_text

        tokenizer = self.itergen.tokenizer
        model = self.itergen.model
        inputs = tokenizer([full_text], return_tensors="pt").to(self.itergen.device)
        input_len = inputs["input_ids"].shape[-1]
        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        tail = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        if not tail:
            return full_text

        completed = full_text + tail
        if completed.startswith(prompt):
            generated = completed[len(prompt):]
            trimmed = self._trim_at_balanced_brace(generated, prompt_brace_depth)
            return prompt + self._post_process_stop(trimmed, stop_tokens, prompt_brace_depth)

        return completed

    def _trim_at_balanced_brace(self, text: str, prompt_brace_depth: int) -> str:
        depth = prompt_brace_depth
        state = "code"
        escape = False
        i = 0

        while i < len(text):
            ch = text[i]
            nxt = text[i + 1] if i + 1 < len(text) else ""

            if state == "line_comment":
                if ch == "\n":
                    state = "code"
            elif state == "block_comment":
                if ch == "*" and nxt == "/":
                    state = "code"
                    i += 1
            elif state in {"single_quote", "double_quote", "template"}:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif (
                    (state == "single_quote" and ch == "'")
                    or (state == "double_quote" and ch == '"')
                    or (state == "template" and ch == "`")
                ):
                    state = "code"
            else:
                if ch == "/" and nxt == "/":
                    state = "line_comment"
                    i += 1
                elif ch == "/" and nxt == "*":
                    state = "block_comment"
                    i += 1
                elif ch == "'":
                    state = "single_quote"
                elif ch == '"':
                    state = "double_quote"
                elif ch == "`":
                    state = "template"
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth <= 0:
                        return text[:i + 1]

            i += 1

        return text

    @staticmethod
    def _brace_depth(text: str) -> int:
        depth = 0
        state = "code"
        escape = False
        i = 0

        while i < len(text):
            ch = text[i]
            nxt = text[i + 1] if i + 1 < len(text) else ""

            if state == "line_comment":
                if ch == "\n":
                    state = "code"
            elif state == "block_comment":
                if ch == "*" and nxt == "/":
                    state = "code"
                    i += 1
            elif state in {"single_quote", "double_quote", "template"}:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif (
                    (state == "single_quote" and ch == "'")
                    or (state == "double_quote" and ch == '"')
                    or (state == "template" and ch == "`")
                ):
                    state = "code"
            else:
                if ch == "/" and nxt == "/":
                    state = "line_comment"
                    i += 1
                elif ch == "/" and nxt == "*":
                    state = "block_comment"
                    i += 1
                elif ch == "'":
                    state = "single_quote"
                elif ch == '"':
                    state = "double_quote"
                elif ch == "`":
                    state = "template"
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1

            i += 1

        return depth
