import torch
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class LlamaConfig:
    input_size: int = 64
    hidden_size: int = 128
    intermediate_size: int = 512
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = True
    head_dim: int = 32
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    use_rope: bool = True
    attention_window: Optional[int] = None
    _attn_implementation: str = "eager"

# --- Helper Functions ---

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# --- Core Components ---
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute frequencies for the whole max_len
        t = torch.arange(max_position_embeddings).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) # [max_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, position_ids):
        # position_ids: [batch, seq_len]
        # We index the precomputed cos/sin using the position_ids
        cos = self.cos_cached[position_ids] # [batch, seq_len, dim]
        sin = self.sin_cached[position_ids] # [batch, seq_len, dim]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if config.hidden_size != config.num_attention_heads * config.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim "
                f"(got {config.hidden_size} vs {config.num_attention_heads} * {config.head_dim})"
            )
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads "
                f"(got {config.num_attention_heads} vs {config.num_key_value_heads})"
            )
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self, 
        hidden_states, 
        position_embeddings=None, 
        attention_mask=None, 
        past_key_value=None, 
        cache_position=None
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # Create cache_kwargs
            cache_kwargs = {"cache_position": cache_position} if cache_position is not None else None
            
            # Pass cache_kwargs to the update method
            key_states, value_states, pad_mask = past_key_value.update(
                key_states, 
                value_states, 
                self.layer_idx, 
                cache_kwargs
            )
            # Combine provided attention_mask (if any) with padding mask
            if attention_mask is None:
                attention_mask_combined = pad_mask
            else:
                # Both masks are additive; shapes should broadcast to [B, 1, Q, K]
                attention_mask_combined = attention_mask + pad_mask
        else:
            attention_mask_combined = attention_mask

        attn_output, _ = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask_combined, self.scaling, self.attention_dropout
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Pass cache_position down to Attention
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position, # <--- Passed Down
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Linear(config.input_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Only initialize RoPE if requested
        if config.use_rope:
            self.rotary_emb = LlamaRotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
        else:
            self.rotary_emb = None

    def forward(self, data, position_ids=None, past_key_values=None, cache_position=None, key_padding_mask=None, attention_window=None):
        # data: [batch, seq_len, input_size]
        hidden_states = self.embed_tokens(data)

        position_embeddings = None

        # --- ROPE LOGIC ---
        if self.rotary_emb is not None:
            if position_ids is None:
                if cache_position is not None:
                    # Use provided cache position (e.g. for step-by-step decoding)
                    position_ids = cache_position
                else:
                    # Default: linear sequence 0 to L
                    seq_len = data.shape[1]
                    position_ids = torch.arange(0, seq_len, device=data.device).unsqueeze(0)

            # Compute cos/sin
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)

        # --- ATTENTION MASK (training causal mask when no cache) ---
        attn_mask = None
        if past_key_values is None:
            # Build a standard causal mask for [B, T, H] sequences
            B, T, _ = hidden_states.shape
            # 0 on and below diagonal, -inf above
            causal = torch.full((T, T), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype)
            causal = torch.triu(causal, diagonal=1)
            window = attention_window if attention_window is not None else self.config.attention_window
            if window is not None and window > 0:
                i = torch.arange(T, device=hidden_states.device).view(T, 1)
                j = torch.arange(T, device=hidden_states.device).view(1, T)
                too_old = (i - j) >= window
                window_mask = torch.where(too_old, torch.full_like(causal, float('-inf')), torch.zeros_like(causal))
                causal = causal + window_mask
            # Shape to [1, 1, T, T] so it broadcasts over batch and heads
            attn_mask = causal.view(1, 1, T, T)
            # Add optional key padding mask: [B, T] where True means PADDED
            if key_padding_mask is not None:
                kpm = key_padding_mask
                if kpm.dim() == 3 and kpm.shape[-1] == 1:
                    kpm = kpm.squeeze(-1)
                if kpm.dim() != 2 or kpm.shape[0] != B or kpm.shape[1] != T:
                    raise ValueError("key_padding_mask must have shape [B, T]")
                kpm = kpm.to(torch.bool)
                zeros = torch.zeros((B, 1, 1, T), device=hidden_states.device, dtype=hidden_states.dtype)
                neg_inf = torch.full_like(zeros, float('-inf'))
                # kpm=True indicates padded positions -> disallow attending to them (add -inf)
                pad_mask = torch.where(kpm.view(B, 1, 1, T), neg_inf, zeros)
                attn_mask = attn_mask + pad_mask

        # --- LAYER LOOP ---
        for layer in self.layers:
            # We pass position_embeddings (which might be None)
            hidden_states = layer(
                hidden_states, 
                position_embeddings,
                attention_mask=attn_mask,
                past_key_value=past_key_values,
                cache_position=cache_position
            )

        return self.norm(hidden_states)


small = LlamaConfig(
    input_size = None,
    hidden_size=128,
    intermediate_size=512,     # 4x hidden is typical
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,     # use MHA; keep equal
    # hidden_act="silu",
    # initializer_range=0.02,
    rms_norm_eps=1e-5,

    attention_bias=False,
    attention_dropout=0.0,     # RL often prefers 0
    mlp_bias=True,             # I’d enable biases for small nets
    head_dim=32,                # hidden_size / n_heads
    # use_positional_encoding=False,
)


big = LlamaConfig(
    input_size = None,
    hidden_size=256,
    intermediate_size=1024,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=8,
    # hidden_act="silu",
    # initializer_range=0.02,
    rms_norm_eps=1e-5,

    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=True,
    head_dim=32,
    # use_positional_encoding=False,
)
