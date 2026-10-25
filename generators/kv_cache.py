"""A KV cache that IterGen and ChopChop can rewind, for any model -- including hybrid ones.

Both tools rewind the cache when they take tokens back:
  IterGen   main.py  cache.crop(max_length=L)   backward(): keep the first L positions
            main.py  cache.crop(-1)             drop the last position
  ChopChop  run_llm  cache.crop(-1)             a rejected token is taken back
Two things break that on current transformers (5.x):
  * crop() takes a negative "tokens to remove" and rejects a positive absolute length;
  * a linear-attention layer (e.g. Qwen3.5) folds the whole past into one fixed-size state,
    which cannot be cut back to an earlier position at all.
This class takes both tools' calls as they are. A standard-attention cache is cropped
exactly. A cache with layers that cannot be cut back is emptied instead; it is then falsy,
so the next step re-reads the kept tokens from scratch. That is exact -- the same
computation as the first time -- and costs one prefill per rewind (counted in `reprefills`).
"""
from transformers.cache_utils import DynamicCache


def text_config(model):
    """The decoder's config (multimodal models keep it under `text_config`)."""
    return getattr(model.config, 'text_config', model.config)


def is_hybrid(config):
    return any(t != 'full_attention' for t in (getattr(config, 'layer_types', None) or []))


class RewindableCache(DynamicCache):
    reprefills = 0                       # class-wide counter; callers reset it per generation

    def __init__(self, config):
        super().__init__(config=config)
        self._config = config
        self.hybrid = is_hybrid(config)

    def __bool__(self):
        # With a config, DynamicCache builds all its layers up front, so len() is never 0;
        # the tools only need to know whether anything has been cached yet.
        return self.get_seq_length() > 0

    def crop(self, max_length):
        cur = self.get_seq_length()
        keep = cur + max_length if max_length < 0 else max_length
        if keep >= cur:
            return
        if self.hybrid:
            DynamicCache.__init__(self, config=self._config)    # empty: the tool re-prefills
            RewindableCache.reprefills += 1
        else:
            super().crop(keep - cur)                             # negative = tokens to remove
