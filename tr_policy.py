import torch
import torch.nn as nn
from typing import Dict, Optional

from transformer import LlamaConfig, LlamaModel
from tr_cache import PositionBasedDynamicCache


class IsaacLabEncoders(nn.Module):
    """Shared spatial and proprio encoders used by policy and value."""

    def __init__(self, config, n_cubes=3):
        super().__init__()
        self.config = config
        self.n_cubes = n_cubes

        # Spatial encoder (set-like; no RoPE)
        self.spatial_config = LlamaConfig(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            use_rope=False,
        )
        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.cube_proj = nn.Linear(7, config.hidden_size)
        self.cube_id_emb = nn.Embedding(n_cubes, config.hidden_size)
        self.spatial_encoder = LlamaModel(self.spatial_config)

        # Proprio encoder (ordered features; use RoPE)
        self.proprio_config = LlamaConfig(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            use_rope=True,
        )
        self.proprio_cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.action_proj = nn.Linear(8, config.hidden_size)
        self.eef_pos_proj = nn.Linear(3, config.hidden_size)
        self.eef_quat_proj = nn.Linear(4, config.hidden_size)
        self.gripper_proj = nn.Linear(2, config.hidden_size)
        self.joint_proj = nn.Linear(1, config.hidden_size)
        self.proprio_encoder = LlamaModel(self.proprio_config)

    def _process_spatial(self, obs: Dict[str, torch.Tensor]):
        batch_size = obs['cube_positions'].shape[0]
        pos = obs['cube_positions'].view(batch_size, self.n_cubes, 3)
        orient = obs['cube_orientations'].view(batch_size, self.n_cubes, 4)
        cubes = self.cube_proj(torch.cat([pos, orient], dim=-1))
        cubes = cubes + self.cube_id_emb(torch.arange(self.n_cubes, device=cubes.device))
        cls_token = self.spatial_cls_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat([cls_token, cubes], dim=1)
        out = self.spatial_encoder(transformer_input)
        return out[:, 0, :]

    def _process_proprio(self, obs: Dict[str, torch.Tensor]):
        batch_size = obs['actions'].shape[0]
        emb_actions = self.action_proj(obs['actions']).unsqueeze(1)
        emb_eef_pos = self.eef_pos_proj(obs['eef_pos']).unsqueeze(1)
        emb_eef_quat = self.eef_quat_proj(obs['eef_quat']).unsqueeze(1)
        emb_gripper = self.gripper_proj(obs['gripper_pos']).unsqueeze(1)
        emb_j_pos = self.joint_proj(obs['joint_pos'].unsqueeze(-1))
        emb_j_vel = self.joint_proj(obs['joint_vel'].unsqueeze(-1))
        token_list = [
            self.proprio_cls_token.expand(batch_size, -1, -1),
            emb_actions,
            emb_eef_pos,
            emb_eef_quat,
            emb_gripper,
            emb_j_pos,
            emb_j_vel,
        ]
        transformer_input = torch.cat(token_list, dim=1)
        out = self.proprio_encoder(transformer_input)
        return out[:, 0, :]


class IsaacLabPolicy(IsaacLabEncoders):
    def __init__(self, config, n_cubes=3, action_dim=8):
        super().__init__(config, n_cubes)
        self.action_dim = action_dim
        self._num_envs: Optional[int] = None
        self._cache_position: Optional[torch.Tensor] = None  # [num_envs]
        self._cache = None


        # --- 3. TEMPORAL DECODER (Sequence Transformer) ---
        # Input: Spatial_CLS + Proprio_CLS -> Output: Action
        self.temporal_config = LlamaConfig(
            input_size=config.hidden_size * 2, # Concatenated features
            hidden_size=config.hidden_size,
            use_rope=True # Sequential history matters
        )
        self.temporal_decoder = LlamaModel(self.temporal_config)
        
        # Final Action Head (e.g., Gaussian Mean)
        # Final Action Head (mu and log_sigma for each action dim)
        self.action_head = nn.Linear(config.hidden_size, 2 * action_dim)

    # encoder methods are inherited from IsaacLabEncoders

    def forward(self, obs, episode_start, past_key_values=None, cache_position=None):
        """
        Auto-detects Inference vs Training based on input shape.
        """
        # 1. Detect Mode
        # If 'actions' is 3D (Batch, Time, Features), we are training on sequences.
        # If 'actions' is 2D (Batch, Features), we are doing inference.
        is_sequence_training = obs['actions'].ndim == 3
        
        if episode_start is None:
            assert is_sequence_training
        if is_sequence_training:
            batch_size, seq_len = obs['actions'].shape[:2]
            # FLATTEN: Combine Batch and Time -> (Batch * Time)
            # We treat every timestep as an independent sample for the encoders
            obs_flat = {k: v.view(-1, *v.shape[2:]) for k, v in obs.items()}
        else:
            obs_flat = obs
            batch_size = obs['actions'].shape[0]
            seq_len = 1
            if self._cache is None:
                assert len(episode_start.shape) == 1
                self.init_cache(num_envs=episode_start.numel(), device=self.action_head.weight.device)
            else:
                self.reset_cache(episode_start)

        # 2. Run Encoders (Parallel)
        # They process (Batch*Time) items at once. Much faster than looping.
        # Output: (Batch * Time, Hidden)
        spatial_summary = self._process_spatial(obs_flat) 
        proprio_summary = self._process_proprio(obs_flat)
        
        # 3. Reshape for Temporal Llama
        # Unflatten: (Batch * Time, Hidden) -> (Batch, Time, Hidden)
        spatial_seq = spatial_summary.view(batch_size, seq_len, -1)
        proprio_seq = proprio_summary.view(batch_size, seq_len, -1)
        
        # Concatenate: (Batch, Time, 2 * Hidden)
        temporal_input = torch.cat([spatial_seq, proprio_seq], dim=-1)

        # 4. Temporal Processing
        if is_sequence_training:
            # TRAINING: use full sequence (no cache) and return per-step logits flattened
            key_padding_mask = None
            # If provided by the data pipeline, use a per-step validity mask [B, T]
            if isinstance(obs, dict) and 'key_padding_mask' in obs:
                kpm = obs['key_padding_mask']
                # Expect shape [B, T] with True indicating PADDED positions
                assert kpm.dim() == 2 and kpm.shape[0] == batch_size and kpm.shape[1] == seq_len, (
                    f"key_padding_mask must be [B,T]={batch_size, seq_len}, got {tuple(kpm.shape)}"
                )
                key_padding_mask = kpm.to(torch.bool)
            temporal_out = self.temporal_decoder(
                temporal_input,
                past_key_values=None,
                key_padding_mask=key_padding_mask,
            )
            # Training: usually want per-step logits [B, T, 2*A] for sequence losses
            logits = self.action_head(temporal_out)
            return logits
   
        else:
            # INFERENCE: single step + optional cache
            temporal_out = self.temporal_decoder(
                temporal_input,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
            last_step = temporal_out[:, -1, :]
            self._cache_position.add_(1)
            return self.action_head(last_step)

    # --- Cache management helpers ---
    def init_cache(self, num_envs: int, device: torch.device):
        self._num_envs = int(num_envs)
        self._cache = PositionBasedDynamicCache().to(device=device)
        self._cache_position = torch.zeros(self._num_envs, dtype=torch.long, device=device)

    def reset_cache(self, reset_mask: torch.Tensor):
        """Reset cache rows where reset_mask is True and zero their positions."""
        if self._cache_position is None:
            return
        reset_mask = reset_mask.to(torch.bool).view(-1)
        if reset_mask.numel() != self._cache_position.numel():
            raise ValueError("reset_mask size must match num_envs for cache reset")
        self._cache.reset(reset_mask)
        self._cache_position[reset_mask] = 0


class IsaacLabSkillPolicy(IsaacLabPolicy):
    """IsaacLab policy that embeds a DIAYN skill and fuses it into proprio.

    - Discrete skills: nn.Embedding(num_skills -> H)
    - Continuous skills: nn.Linear(skill_dim -> H)
    - Fusion: concat(proprio_summary, skill_embed) -> Linear(2H -> H), then proceed
      with spatial-proprio concat into temporal decoder.
    """

    def __init__(self, config, n_cubes=3, action_dim=8, *, skill_dim: int, discrete: bool = True):
        super().__init__(config, n_cubes=n_cubes, action_dim=action_dim)
        H = config.hidden_size
        self.skill_dim = skill_dim
        self.discrete = discrete
        if discrete:
            self.skill_emb = nn.Embedding(skill_dim, H)
        else:
            self.skill_emb = nn.Linear(skill_dim, H)
        self.proprio_skill_fuse = nn.Linear(2 * H, H)

    def _embed_skills(self, skills):
        if self.discrete:
            return self.skill_emb(skills.long())
        else:
            return self.skill_emb(skills)

    def forward(self, obs, episode_start, past_key_values=None, cache_position=None):
        # Expect obs to contain 'skills' with shape [B,T] or [B,T,D] (training)
        # or [B] / [B,D] (inference). We mirror parent flatten/unflatten logic.
        assert 'skills' in obs, "IsaacLabSkillPolicy expects 'skills' in obs"

        is_sequence_training = obs['actions'].ndim == 3

        if episode_start is None:
            assert is_sequence_training
        if is_sequence_training:
            batch_size, seq_len = obs['actions'].shape[:2]
            obs_flat = {k: v.view(-1, *v.shape[2:]) for k, v in obs.items() if k != 'key_padding_mask'}
            skills = obs_flat['skills']  # [B*T] or [B*T,D]
        else:
            obs_flat = obs
            batch_size = obs['actions'].shape[0]
            seq_len = 1
            skills = obs_flat['skills']  # [B] or [B,D]
            if self._cache is None:
                assert len(episode_start.shape) == 1
                self.init_cache(num_envs=episode_start.numel(), device=self.action_head.weight.device)
            else:
                self.reset_cache(episode_start)

        # Encoders on flat batch
        spatial_summary = self._process_spatial(obs_flat)  # [B*T,H] or [B,H]
        proprio_summary = self._process_proprio(obs_flat)  # [B*T,H] or [B,H]

        # Skill embedding on flat batch
        skill_embed = self._embed_skills(skills)
        if skill_embed.dim() == 1:
            skill_embed = skill_embed.unsqueeze(-1)

        # Reshape to sequences
        spatial_seq = spatial_summary.view(batch_size, seq_len, -1)
        proprio_seq = proprio_summary.view(batch_size, seq_len, -1)
        skill_seq = skill_embed.view(batch_size, seq_len, -1)

        # Fuse skill into proprio
        proprio_fused = torch.cat([proprio_seq, skill_seq], dim=-1)
        proprio_fused = self.proprio_skill_fuse(proprio_fused)

        # Temporal input: spatial + fused proprio
        temporal_input = torch.cat([spatial_seq, proprio_fused], dim=-1)

        if is_sequence_training:
            key_padding_mask = None
            if isinstance(obs, dict) and 'key_padding_mask' in obs:
                kpm = obs['key_padding_mask']
                key_padding_mask = kpm.to(torch.bool)
            temporal_out = self.temporal_decoder(
                temporal_input,
                past_key_values=None,
                key_padding_mask=key_padding_mask,
            )
            logits = self.action_head(temporal_out)
            return logits
        else:
            temporal_out = self.temporal_decoder(
                temporal_input,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
            last_step = temporal_out[:, -1, :]
            self._cache_position.add_(1)
            return self.action_head(last_step)


class IsaacLabValue(nn.Module):
    """Transformer-based value function that mirrors IsaacLabPolicy encoders,
    but without temporal decoding. It fuses spatial and proprio summaries and
    predicts a scalar value.
    """

    def __init__(self, config, n_cubes=3):
        super().__init__()
        # Share encoder structure via composition
        self.enc = IsaacLabEncoders(config, n_cubes)

        # Fusion and value head
        self.fuse = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(self, obs: Dict[str, torch.Tensor]):
        spatial = self.enc._process_spatial(obs)
        proprio = self.enc._process_proprio(obs)
        fused = torch.cat([spatial, proprio], dim=-1)
        fused = self.fuse(fused)
        values = self.value_head(fused)  # [B, 1]
        return values


class IsaacLabDiscriminator(nn.Module):
    """Transformer-based discriminator that reuses IsaacLab encoders.

    It selects which encoder branches to use based on a configurable
    set of observation fields (default: spatial cube pose fields).
    """

    _SPATIAL_FIELDS = frozenset(("cube_positions", "cube_orientations"))
    _PROPRIO_FIELDS = frozenset((
        "actions",
        "eef_pos",
        "eef_quat",
        "gripper_pos",
        "joint_pos",
        "joint_vel",
    ))

    def __init__(self, config, n_cubes=3, *, skill_dim: int, fields=None):
        super().__init__()
        self.enc = IsaacLabEncoders(config, n_cubes)
        if fields is None:
            fields = ("cube_positions", "cube_orientations")
        self.fields = tuple(fields)
        field_set = set(self.fields)
        self.use_spatial = len(field_set & self._SPATIAL_FIELDS) > 0
        self.use_proprio = len(field_set & self._PROPRIO_FIELDS) > 0
        if not (self.use_spatial or self.use_proprio):
            raise ValueError("IsaacLabDiscriminator fields must include spatial or proprio inputs")

        feature_dim = config.hidden_size * (int(self.use_spatial) + int(self.use_proprio))
        self.fuse = nn.Linear(feature_dim, config.hidden_size) if feature_dim != config.hidden_size else None
        self.output_layer = nn.Linear(config.hidden_size, skill_dim)
        self.first_layer = self.fuse if self.fuse is not None else self.output_layer

    def forward(self, obs: Dict[str, torch.Tensor]):
        features = []
        if self.use_spatial:
            features.append(self.enc._process_spatial(obs))
        if self.use_proprio:
            features.append(self.enc._process_proprio(obs))
        if len(features) == 1:
            fused = features[0]
        else:
            fused = torch.cat(features, dim=-1)
        if self.fuse is not None:
            fused = self.fuse(fused)
        return self.output_layer(fused)
