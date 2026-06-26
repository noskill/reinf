import torch
from transformer import LlamaConfig, LlamaModel, LlamaRMSNorm
from recurrent_cache import CacheModuleMixin
from tr_cache import PositionBasedDynamicCache, WindowedPositionBasedDynamicCache


class CachedTransformer(torch.nn.Module, CacheModuleMixin):
    def __init__(self, config):
        super().__init__()
        self.backbone = LlamaModel(config)
        self.attention_window = config.attention_window
        self._cache_position: Optional[torch.Tensor] = None
        self._cache = None

    def reset_cache(self, reset_mask: torch.Tensor):
        if reset_mask is None:
            return
        if self._cache_position is None:
            return
        reset_mask = reset_mask.to(torch.bool).view(-1)
        if reset_mask.numel() != self._cache_position.numel():
            raise ValueError("reset_mask size must match num_envs for cache reset")
        self._cache.reset(reset_mask)
        self._cache_position[reset_mask] = 0

    def clear_cache(self):
        self._cache = None
        self._cache_position = None
        
    @property
    def device(self):
        return next(next(self.backbone.modules()).parameters()).device

    def init_cache(self, num_entries):
        if self.attention_window is not None and self.attention_window > 0:
            self._cache = WindowedPositionBasedDynamicCache(self.attention_window)
        else:
            self._cache = PositionBasedDynamicCache()
        self._cache_position = torch.zeros(num_entries, dtype=torch.long, device=self.device)

    def forward(self, x, key_padding_mask, reset_mask):
        using_internal_cache = reset_mask is not None
        if using_internal_cache:
            assert x.shape[1] == 1, f"CachedTransformer internal cache expects online T=1 calls, got sequence length {x.shape[1]}"
            if self._cache is None or self._cache_position is None:
                self.init_cache(len(reset_mask))
            else:
                self.reset_cache(reset_mask)

        result = self.backbone(
            x,
            past_key_values=self._cache if using_internal_cache else None,
            cache_position=self._cache_position.unsqueeze(1) if using_internal_cache else None,
            key_padding_mask=key_padding_mask,
            attention_window=self.attention_window,)

        if using_internal_cache:
            self._cache_position.add_(1)
        return result

        
        
        


