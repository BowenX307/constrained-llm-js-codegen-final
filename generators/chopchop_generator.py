"""
Generic ChopChop generation framework.

Language-specific components (AST nodes, constructors, pruners, prompt helpers)
are provided by language plugins, e.g. generators/javascript_chopchop.py.

Typical usage:

    from generators.javascript_chopchop import (
        CONSTRUCTORS, JS_START_RULE, JS_CONTEXT,
        make_js_pruner, extract_js_prefix, build_js_prompt,
    )
    gen = ChopchopGenerator(
        model_name="microsoft/phi-2",
        grammar="generators/grammars/javascript_chopchop.lark",
        constructors=CONSTRUCTORS,
        start_rule=JS_START_RULE,
        pruner_fn=make_js_pruner("basic"),
        extract_prefix_fn=extract_js_prefix,
        build_prompt_fn=build_js_prompt,
        context=JS_CONTEXT,
    )
    output = gen.generate(prompt, stop_tokens=[...], temperature=0.2)
"""

import gc
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.utils import logging as hf_logging

from .base import BaseGenerator


# Make ChopChop package imports available when running from repository root.
CHOPCHOP_ROOT = Path(__file__).resolve().parent / "chopchop"
if str(CHOPCHOP_ROOT) not in sys.path:
    sys.path.insert(0, str(CHOPCHOP_ROOT))

from core.rewrite import rewriter  # type: ignore  # noqa: E402
from llm.realizability import RealizabilityChecker  # type: ignore  # noqa: E402
from llm.run_llm import Config, LanguageModelRunner, ModelConfig  # type: ignore  # noqa: E402
import llm.run_llm as _run_llm  # type: ignore  # noqa: E402
from core.lark.from_lark import parse_attribute_grammar  # type: ignore  # noqa: E402

from .kv_cache import RewindableCache, text_config  # noqa: E402


class CheckLimitReached(Exception):
    """The task used up its check-time limit; generation stops here."""


class _SafeRealizabilityChecker(RealizabilityChecker):
    """
    Wraps the upstream RealizabilityChecker with:

      - RecursionError / NetworkXError guard (treat as realizable on error, counted)
      - A per-task limit on CHECK time, measured as this thread's CPU time
        (time.thread_time) spent inside realizable(). When it is used up, generation
        stops: the output ends at the last accepted token and the task is marked
        `limit_hit`. Every token before that point was checked.

        This replaces the earlier 120 s "cumulative budget", after which every further
        token was accepted UNCHECKED (204/397 tasks in the A100 run lost their checks that
        way, at a median of 183 characters). CPU time instead of wall-clock time makes the
        cut-off independent of what else runs on the machine, so several workers in
        parallel give the same outputs as one.
      - Rejection attribution: when a token is rejected, the same prefix is re-checked
        with the grammar alone, and -- if the grammar accepts it -- with each pruner rule
        alone, to record WHO rejected it. These re-checks run only on rejections, do not
        change the decision, and are timed separately (not counted toward the limit).

    The rewriter's global state is cleared around the attribution re-checks: its memo
    cache is keyed on the arguments, not on which pruner rules are active.
    """

    def __init__(self, constraint, parser, lexerspec, *, check_cpu_limit_s: Optional[float] = None,
                 attribute_rejections: bool = True, max_samples: int = 30):
        super().__init__(constraint, parser, lexerspec)
        self.check_cpu_limit_s = check_cpu_limit_s
        self.attribute_rejections = attribute_rejections
        self.max_samples = max_samples
        self._grammar_only = RealizabilityChecker(lambda asts: asts, parser, lexerspec)
        # A pruner may expose its rules (see javascript_chopchop.make_js_pruner) so that a
        # rejection can be traced to the rule(s) that caused it.
        self.rules = tuple(getattr(constraint, 'rules', ()))
        self.reset_budget()

    def reset_budget(self):
        """Reset the per-task counters (call before each generation run)."""
        self.stats = {
            'checks': 0, 'check_cpu_s': 0.0, 'check_wall_s': 0.0,
            'rejected': 0, 'rejected_by_grammar': 0, 'rejected_by_pruner': 0,
            'rejected_by_rule': {r: 0 for r in self.rules},
            'rejection_samples': [],          # [{'tail', 'by', 'rules'}], first max_samples
            'attribution_wall_s': 0.0,
            'check_errors': 0, 'check_error_msg': None,
            'limit_hit': False,
        }
        self.last_accepted = None

    def realizable(self, prefix: str, final: bool = False) -> bool:
        st = self.stats
        c0, w0 = time.thread_time(), time.perf_counter()
        try:
            result = super().realizable(prefix, final=final)
        except Exception as e:                # RecursionError, NetworkXError, ...
            rewriter.clear()
            result = True
            st['check_errors'] += 1
            if st['check_error_msg'] is None:
                st['check_error_msg'] = f'{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ""}'
        st['check_cpu_s'] += time.thread_time() - c0
        st['check_wall_s'] += time.perf_counter() - w0
        st['checks'] += 1

        if result:
            self.last_accepted = prefix
        else:
            st['rejected'] += 1
            if self.attribute_rejections:
                self._attribute(prefix, final)

        if self.check_cpu_limit_s is not None and st['check_cpu_s'] >= self.check_cpu_limit_s:
            st['limit_hit'] = True
            raise CheckLimitReached()
        return result

    def _attribute(self, prefix: str, final: bool):
        st = self.stats
        w0 = time.perf_counter()
        try:
            rewriter.clear()
            if not self._grammar_only.realizable(prefix, final=final):
                by, rules = 'grammar', []
            else:
                by, rules = 'pruner', []
                for rule in self.rules:
                    with self.constraint.only(rule):
                        rewriter.clear()
                        if not RealizabilityChecker.realizable(self, prefix, final=final):
                            rules.append(rule)
                for rule in rules:
                    st['rejected_by_rule'][rule] += 1
            st['rejected_by_' + by] += 1
        except Exception as e:
            by, rules = f'unknown ({type(e).__name__})', []
        finally:
            rewriter.clear()
            st['attribution_wall_s'] += time.perf_counter() - w0
        if len(st['rejection_samples']) < self.max_samples:
            st['rejection_samples'].append({'tail': prefix[-80:], 'by': by, 'rules': rules})


class _LocalLanguageModelRunner(LanguageModelRunner):
    """
    Local subclass of the upstream LanguageModelRunner that overrides the
    model-loading, tokenization, and token-generation methods.

    Supports both chat-template models (e.g. Qwen Instruct) and plain
    causal-LM models (e.g. Phi-2) without modifying upstream code.
    """

    def _load_model_and_tokenizer(self):
        hf_logging.set_verbosity_error()
        model_id = self.model_config.model_id.strip()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=self.model_config.dtype,
        )

        vocab_size = model.get_input_embeddings().num_embeddings
        if len(tokenizer) > vocab_size:
            model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def _tokenize_prompt(
        self, prompt: str, context: str, fixed_prefix: str = ""
    ) -> torch.Tensor:
        """
        Convert the task prompt into input token IDs.

        Chat-template models (e.g. Qwen2.5-Coder-Instruct): formats prompt as
        system+user message pair via apply_chat_template.

        Plain causal-LM models (e.g. Phi-2): concatenates context and prompt
        as a single string.

        If fixed_prefix is provided (used during retry after a partial timeout),
        its tokens are appended so that generation continues from that point.
        """
        if getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            )
            # Newer transformers return a BatchEncoding here (apply_chat_template's
            # return_dict now defaults to True); older ones returned a bare tensor.
            if not isinstance(input_ids, torch.Tensor):
                input_ids = input_ids["input_ids"]
        else:
            text = f"{context}\n\n{prompt}\n"
            input_ids = self.tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]

        if fixed_prefix:
            prefix_tokens = self.tokenizer(
                fixed_prefix,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            input_ids = torch.cat([input_ids, prefix_tokens], dim=-1)

        return input_ids.to(self.model.device)

    def _generate_next_token(
        self,
        input_ids: torch.Tensor,
        config: Config,
        generated_tokens: List[int],
        forbidden_tokens: set[int],
        cache: DynamicCache,
    ):
        """
        Generate the next single token, enforcing grammar-driven token masking
        via bad_words_ids (tokens that would make the prefix unrealizable are
        set to -inf before sampling).
        """
        bad_words = [[tok] for tok in forbidden_tokens] if forbidden_tokens else None

        inp = torch.tensor([list(input_ids[0]) + generated_tokens])
        inp = inp.to(self.model_config.device)
        attention_mask = torch.ones_like(inp, dtype=torch.long)

        if self.tokenizer.eos_token_id in forbidden_tokens:
            eos_token_id = None
        else:
            eos_token_id = self.tokenizer.eos_token_id

        do_sample = config.temperature > 0
        gen_kwargs = dict(
            attention_mask=attention_mask,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=1,
            bad_words_ids=bad_words,
            repetition_penalty=config.repetition_penalty,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            past_key_values=cache,
        )
        if do_sample:
            gen_kwargs.update(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )

        return self.model.generate(inp, **gen_kwargs)

    def __del__(self):
        try:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class ChopchopGenerator(BaseGenerator):
    """
    Grammar-constrained code generator using the ChopChop framework.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    grammar : str
        Path to a .lark attribute grammar file.
    constructors : list
        Language-specific AST node classes for parse_attribute_grammar().
    start_rule : str
        Grammar entry point (e.g. "function_declaration").
    pruner_fn : callable, optional
        Callable(asts) -> asts applied inside the realizability checker.
        Defaults to the identity function (no extra pruning).
    extract_prefix_fn : callable, optional
        Callable(prompt: str) -> str that extracts a fixed_prefix from the raw
        prompt (e.g. the function signature without the opening brace).
        Defaults to returning an empty string.
    build_prompt_fn : callable, optional
        Callable(prompt: str) -> str that builds the instruction prompt for the
        model from the raw prompt text.
        Defaults to the identity function.
    context : str, optional
        System context string for the model. Defaults to a generic assistant
        prompt.
    """

    def __init__(
        self,
        model_name: str,
        grammar: Optional[str],
        *,
        constructors: list,
        start_rule: str,
        pruner_fn: Optional[Callable] = None,
        extract_prefix_fn: Optional[Callable[[str], str]] = None,
        build_prompt_fn: Optional[Callable[[str], str]] = None,
        context: str = "You are a helpful coding assistant.",
        **kwargs,
    ):
        self.model_name = model_name
        self.context = context
        self.fixed_prefix = kwargs.get("fixed_prefix", "")
        self.extract_prefix_fn = extract_prefix_fn or (lambda p: "")
        self.build_prompt_fn = build_prompt_fn or (lambda p: p)

        grammar_source = self._load_grammar_source(grammar)

        effective_pruner = pruner_fn if pruner_fn is not None else (lambda asts: asts)

        lexer_spec, parser = parse_attribute_grammar(
            constructors, grammar_source, start_rule
        ).build_parser()
        self.checker = _SafeRealizabilityChecker(
            effective_pruner, parser, lexer_spec,
            check_cpu_limit_s=kwargs.get("check_cpu_limit_s"),
            attribute_rejections=kwargs.get("attribute_rejections", True),
        )

        self.runner = _LocalLanguageModelRunner(ModelConfig(model_id=model_name))

        # The upstream run loop builds `DynamicCache()` and takes rejected tokens back with
        # crop(-1); use the cache that handles that on every model (see kv_cache.py).
        cfg = text_config(self.runner.model)
        _run_llm.DynamicCache = lambda: RewindableCache(cfg)
        self.last_stats = None

    @staticmethod
    def _load_grammar_source(grammar: Optional[str]) -> str:
        if not grammar:
            raise FileNotFoundError(
                "ChopChop requires an explicit grammar file path via --grammar"
            )
        grammar_path = Path(grammar)
        if grammar_path.exists():
            return grammar_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Grammar file not found: {grammar}")

    def generate(self, prompt: str, stop_tokens=None, **kwargs) -> str:
        """
        Generate a single completion for the given prompt.

        Flow:
        1. Build the instruction prompt via build_prompt_fn.
        2. Extract the fixed_prefix via extract_prefix_fn (unless overridden).
        3. Run ChopChop's grammar-constrained generation loop.
        4. Strip the fixed_prefix and opening brace from the output.
        5. Apply stop-token truncation.
        """
        temperature = float(kwargs.get("temperature"))
        max_new_tokens = kwargs.get("max_new_tokens")
        context = kwargs.get("context", self.context)
        fixed_prefix = kwargs.get("fixed_prefix", self.fixed_prefix)
        # Wall-clock limit of the upstream loop. None = upstream's default (no limit): the
        # per-task limit is the checker's CPU-time limit instead.
        wall_timeout = kwargs.get("wall_timeout_s")
        stop_tokens = kwargs.get("stop_tokens", stop_tokens) or []

        if not fixed_prefix:
            fixed_prefix = self.extract_prefix_fn(prompt)

        task_prompt = kwargs.get("task_prompt")
        if not task_prompt:
            task_prompt = self.build_prompt_fn(prompt)

        self.checker.reset_budget()
        RewindableCache.reprefills = 0
        cfg_kwargs = dict(temperature=temperature, max_new_tokens=max_new_tokens)
        if wall_timeout is not None:
            cfg_kwargs['timeout'] = wall_timeout

        run_info = None
        try:
            run_info = self.runner.run(
                Config(**cfg_kwargs),
                prompt=task_prompt,
                context=context,
                fixed_prefix=fixed_prefix,
                realizability_checker=self.checker,
            )
            output = run_info.output
        except CheckLimitReached:
            # Stop at the last token the checker accepted; everything up to it was checked.
            output = self.checker.last_accepted or fixed_prefix
        except RecursionError:
            self.last_run_info = None
            self.last_stats = dict(self.checker.stats, generation_error='RecursionError inside ChopChop')
            return ""

        self.last_run_info = run_info
        n_prefix = len(self.runner.tokenizer(fixed_prefix, add_special_tokens=False)['input_ids'])
        self.last_stats = dict(
            self.checker.stats,
            check_cpu_s=round(self.checker.stats['check_cpu_s'], 3),
            check_wall_s=round(self.checker.stats['check_wall_s'], 3),
            attribution_wall_s=round(self.checker.stats['attribution_wall_s'], 3),
            llm_finished=run_info.llm_finished if run_info else False,
            wall_timed_out=run_info.timed_out if run_info else False,
            tokens=(run_info.num_tokens_generated - n_prefix) if run_info else
                   len(self.runner.tokenizer(output, add_special_tokens=False)['input_ids']) - n_prefix,
            tokens_tried=run_info.num_tokens_guessed if run_info else None,
            reprefills=RewindableCache.reprefills,
        )

        # The grammar start rule produces "function foo(…) { body }".
        # Strip fixed_prefix and the opening "{" since the caller's prompt
        # already contains them, leaving only "body }" to append.
        if fixed_prefix and output.startswith(fixed_prefix):
            output = output[len(fixed_prefix):].lstrip()
            if output.startswith("{"):
                output = output[1:]

        output = self._post_process_stop(output, stop_tokens)

        return output
