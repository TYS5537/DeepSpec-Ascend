from typing import Optional

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4GroupedLinear,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4PreTrainedModel,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4SparseMoeBlock,
    DeepseekV4UnweightedRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from deepspec.modeling.dspark.common import (
    AcceptRatePredictor,
    DSparkForwardOutput,
    build_eval_mask,
    create_noise_embed,
    create_position_ids,
    log_sampler_stats,
    sample_anchor_positions,
)
from deepspec.modeling.dspark.markov_head import build_markov_head
from deepspec.utils.sampling import sample_tokens


def create_deepseek_v4_dspark_attention_mask(
    *,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    bsz, num_blocks = anchor_positions.shape
    q_len = num_blocks * block_size
    kv_len = seq_len + q_len

    q_idx = torch.arange(q_len, device=device)
    kv_idx = torch.arange(kv_len, device=device)
    q_block_id = q_idx // block_size
    kv_block_id = (kv_idx - seq_len).clamp_min(0) // block_size

    context_allowed = (
        (kv_idx.view(1, 1, -1) < seq_len)
        & (kv_idx.view(1, 1, -1) < anchor_positions[:, q_block_id].unsqueeze(-1))
    )
    draft_allowed = (
        (kv_idx.view(1, 1, -1) >= seq_len)
        & (q_block_id.view(1, -1, 1) == kv_block_id.view(1, 1, -1))
    )
    valid_block = block_keep_mask[:, q_block_id].unsqueeze(-1)
    allowed = (context_allowed | draft_allowed) & valid_block
    mask = torch.zeros(bsz, 1, q_len, kv_len, dtype=dtype, device=device)
    return mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)


class DeepseekV4DSparkAttention(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.rope_layer_type = (
            "main" if self.layer_type == "sliding_attention" else "compress"
        )
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_groups = int(config.num_attention_heads)
        self.head_dim = int(config.head_dim)
        self.sliding_window = int(config.sliding_window)
        self.attention_dropout = float(config.attention_dropout)
        self.is_causal = False
        self.scaling = self.head_dim**-0.5

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.o_b_proj = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=False,
        )
        self.sinks = nn.Parameter(torch.empty(self.num_heads))
        nn.init.zeros_(self.sinks)

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = hidden_states.shape[:-1]
        kv = self.kv_norm(self.kv_proj(hidden_states)).view(
            bsz,
            seq_len,
            1,
            self.head_dim,
        )
        kv = kv.transpose(1, 2)
        return apply_rotary_pos_emb(kv, cos, sin)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]
        hidden_shape = (bsz, q_len, -1, self.head_dim)
        cos, sin = position_embeddings[self.rope_layer_type]
        cos_ctx, sin_ctx = cos[:, :ctx_len, :], sin[:, :ctx_len, :]
        cos_q, sin_q = cos[:, -q_len:, :], sin[:, -q_len:, :]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = apply_rotary_pos_emb(self.q_b_norm(q), cos_q, sin_q)

        kv_ctx = self._project_kv(target_hidden_states, cos_ctx, sin_ctx)
        kv_noise = self._project_kv(hidden_states, cos_q, sin_q)
        kv = torch.cat([kv_ctx, kv_noise], dim=2)
        if past_key_values is not None:
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        self.is_causal = bool(kwargs.get("is_causal", False))
        attn_output, attn_weights = eager_attention_forward(
            self,
            q,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
        )
        attn_output = apply_rotary_pos_emb(
            attn_output.transpose(1, 2),
            cos_q,
            -sin_q,
        ).transpose(1, 2)
        grouped = attn_output.reshape(
            bsz,
            q_len,
            self.config.o_groups,
            -1,
        )
        grouped = self.o_a_proj(grouped).flatten(2)
        return self.o_b_proj(grouped), attn_weights


class DeepseekV4DSparkDecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV4DSparkAttention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = DeepseekV4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output, _ = self.self_attn(
            hidden_states=self.input_layernorm(collapsed),
            target_hidden_states=target_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(
            -2
        ) + torch.matmul(comb.to(dtype).transpose(-1, -2), hidden_states)

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(
            self.post_attention_layernorm(collapsed),
            input_ids=input_ids,
        )
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(
            -2
        ) + torch.matmul(comb.to(dtype).transpose(-1, -2), hidden_states)


class DeepseekV4DSparkModel(DeepseekV4PreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV4DSparkDecoderLayer"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        required_fields = (
            "target_layer_ids",
            "mask_token_id",
            "num_anchors",
            "enable_confidence_head",
            "markov_rank",
        )
        for field in required_fields:
            assert hasattr(config, field), f"config.{field} must be provided."
        if int(config.markov_rank) > 0:
            assert hasattr(config, "markov_head_type"), (
                "config.markov_head_type must be provided when markov_rank > 0."
            )
        if bool(config.enable_confidence_head):
            assert hasattr(config, "confidence_head_with_markov"), (
                "config.confidence_head_with_markov must be provided when "
                "enable_confidence_head is true."
            )
        self.target_layer_ids = config.target_layer_ids

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV4DSparkDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.hc_head = DeepseekV4HyperHead(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.mask_token_id = int(config.mask_token_id)
        self.num_anchors = int(config.num_anchors)

        self.markov_head = build_markov_head(config)

        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = False
        if self.enable_confidence_head:
            self.confidence_head_with_markov = bool(config.confidence_head_with_markov)
        if self.enable_confidence_head and self.confidence_head_with_markov:
            assert self.markov_head is not None

        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                input_dim += config.markov_rank
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)
        self.post_init()

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        freeze: bool = True,
    ):
        assert self.embed_tokens.weight.shape == embed_tokens.weight.shape
        assert self.lm_head.weight.shape == lm_head.weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            assert prev_token_ids is not None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                dtype=hidden_states.dtype
            )
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
            return self.confidence_head(features).float()
        return self.confidence_head(hidden_states).float()

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits
        return self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            hidden_states=hidden_states,
            temperature=temperature,
        )

    def sample_draft_token_step(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert base_logits.ndim == 2, (
            "sample_draft_token_step expects base_logits shaped [batch, vocab], "
            f"got {tuple(base_logits.shape)}."
        )
        if self.markov_head is None:
            step_logits = base_logits
        else:
            step_logits = self.markov_head.apply_step_logits(
                base_logits,
                token_ids=prev_token_ids,
                hidden_states=hidden_states,
            )
        sampled_token_ids = sample_tokens(
            step_logits.unsqueeze(1),
            temperature=temperature,
        ).squeeze(1)
        return sampled_token_ids, step_logits

    def _prepare_full_position_ids(
        self,
        *,
        position_ids: torch.LongTensor,
        target_hidden_states: torch.Tensor,
        noise_embedding: torch.Tensor,
    ) -> torch.LongTensor:
        ctx_len = target_hidden_states.shape[1]
        q_len = noise_embedding.shape[1]
        if position_ids.shape[1] == ctx_len + q_len:
            return position_ids
        assert position_ids.shape[1] == q_len, (
            "DeepseekV4DSparkModel expected position_ids to cover either "
            "context+draft or draft only."
        )
        ctx_offsets = torch.arange(ctx_len, device=position_ids.device).unsqueeze(0)
        ctx_start = (position_ids[:, :1] - ctx_len).clamp_min(0)
        context_position_ids = ctx_start + ctx_offsets
        return torch.cat([context_position_ids, position_ids], dim=1)

    def _forward_backbone(
        self,
        *,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del use_cache
        assert noise_embedding is not None, "noise_embedding must be provided."
        assert target_hidden_states is not None, "target_hidden_states must be provided."
        hidden_states = noise_embedding.unsqueeze(2).expand(
            -1,
            -1,
            self.config.hc_mult,
            -1,
        ).contiguous()
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        full_position_ids = self._prepare_full_position_ids(
            position_ids=position_ids,
            target_hidden_states=target_hidden_states,
            noise_embedding=noise_embedding,
        )
        position_embeddings = {
            "main": self.rotary_emb(
                noise_embedding,
                position_ids=full_position_ids,
                layer_type="main",
            ),
            "compress": self.rotary_emb(
                noise_embedding,
                position_ids=full_position_ids,
                layer_type="compress",
            ),
        }
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        return self.norm(self.hc_head(hidden_states))

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> DSparkForwardOutput:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=device,
        )
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        )
        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(
            bsz,
            -1,
        )
        draft_position_ids = create_position_ids(anchor_positions, self.block_size)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        dspark_attn_mask = create_deepseek_v4_dspark_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.block_size,
            dtype=noise_embedding.dtype,
            device=device,
        )
        output_hidden = self._forward_backbone(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=dspark_attn_mask,
        )

        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(
            bsz,
            num_blocks,
            self.block_size,
            -1,
        )

        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1,
            1,
            -1,
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(
                    -1,
                    anchor_positions.size(1),
                    -1,
                    -1,
                ),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    target_last_hidden_states.size(-1),
                ),
            )
            aligned_target_logits = self.compute_logits(aligned_target_hidden)
        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        draft_logits = self.compute_logits(output_hidden).reshape(
            bsz,
            num_blocks,
            self.block_size,
            -1,
        )
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits,
                token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
            )

        log_sampler_stats(
            seq_len=seq_len,
            loss_mask=loss_mask,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
        )

        confidence_pred = None
        if self.confidence_head is not None:
            if self.confidence_head_with_markov:
                assert self.markov_head is not None
                prev_embeddings = self.markov_head.get_prev_embeddings(
                    prev_token_ids
                ).to(dtype=output_hidden_4d.dtype)
                confidence_features = torch.cat(
                    [output_hidden_4d, prev_embeddings],
                    dim=-1,
                )
                confidence_pred = self.confidence_head(confidence_features).float()
            else:
                confidence_pred = self.confidence_head(output_hidden_4d).float()

        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )


__all__ = [
    "DeepseekV4DSparkModel",
    "create_deepseek_v4_dspark_attention_mask",
]
