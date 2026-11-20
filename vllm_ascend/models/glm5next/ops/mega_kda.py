# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash A5 fixed-window Decode with CANN Mega KDA."""

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any

import torch

MEGA_KDA_HIDDEN_SIZE = 4096
MEGA_KDA_NUM_HEADS = 64
MEGA_KDA_HEAD_DIM = 128
MEGA_KDA_SPEC_QUERY_LEN = 6
_NZ_FORMAT = 29


@dataclass
class MegaKDARecoverSSMRecords:
    correction: torch.Tensor | None = None
    replay_k: torch.Tensor | None = None
    replay_decay: torch.Tensor | None = None

    @property
    def key_gate(self) -> torch.Tensor:
        # The RecoverSSM context uses this existing attribute for its key pool.
        if self.replay_k is None:
            raise RuntimeError("GLM Mega KDA records are not bound to checkpoint pages")
        return self.replay_k

    def ensure_pages(
        self,
        num_pages: int,
        num_heads: int,
        spec_query_len: int,
        head_dim: int,
        device: torch.device,
    ) -> None:
        shape = (num_pages, num_heads, spec_query_len, head_dim)
        if self.correction is not None:
            if (
                self.correction.shape != shape
                or self.replay_k is None
                or self.replay_k.shape != shape
                or self.replay_decay is None
                or self.replay_decay.shape != shape
            ):
                raise ValueError(f"GLM Mega KDA records must have page-indexed shape {shape}")
            return
        self.correction = torch.empty(shape, dtype=torch.float32, device=device)
        self.replay_k = torch.empty(shape, dtype=torch.float16, device=device)
        self.replay_decay = torch.empty(shape, dtype=torch.float32, device=device)

    @classmethod
    def allocate(
        cls,
        num_pages: int,
        num_heads: int,
        spec_query_len: int,
        head_dim: int,
        device: torch.device,
    ) -> "MegaKDARecoverSSMRecords":
        records = cls()
        records.ensure_pages(num_pages, num_heads, spec_query_len, head_dim, device)
        return records


@dataclass(frozen=True)
class MegaKDAWeights:
    qkv_projection_weight: torch.Tensor
    decay_projection_a_weight: torch.Tensor
    decay_projection_b_weight: torch.Tensor
    beta_projection_weight: torch.Tensor
    output_gate_a_projection_weight: torch.Tensor
    output_gate_b_projection_weight: torch.Tensor
    output_norm_weight: torch.Tensor
    output_projection_weight: torch.Tensor
    conv1d_weight: torch.Tensor


def _to_nz(weight: torch.Tensor) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_format_cast(weight.detach().to(dtype=torch.bfloat16).contiguous(), _NZ_FORMAT)


def prepare_mega_kda_weights(layer: Any) -> MegaKDAWeights:
    projection_size = layer.local_projection_size
    head_dim = layer.head_dim
    input_weight = layer.in_proj_qkvbfg_a.weight
    qkv, beta, decay_a, gate_a = input_weight.split(
        [3 * projection_size, layer.local_num_heads, head_dim, head_dim], dim=0
    )
    weights = (
        qkv,
        decay_a,
        layer.f_b_proj.weight,
        beta,
        gate_a,
        layer.g_b_proj.weight,
        layer.o_proj.weight,
    )
    if any(weight.dtype != torch.bfloat16 for weight in weights):
        raise ValueError("GLM Mega KDA requires BF16 projection weights")
    conv_weight = layer._merged_conv_weight
    if conv_weight is None:
        raise RuntimeError("GLM Mega KDA convolution weights are not initialized")
    return MegaKDAWeights(
        qkv_projection_weight=_to_nz(qkv),
        decay_projection_a_weight=_to_nz(decay_a),
        decay_projection_b_weight=_to_nz(layer.f_b_proj.weight),
        beta_projection_weight=_to_nz(beta),
        output_gate_a_projection_weight=_to_nz(gate_a),
        output_gate_b_projection_weight=_to_nz(layer.g_b_proj.weight),
        output_norm_weight=layer.o_norm.weight.detach().to(dtype=torch.float32).contiguous(),
        output_projection_weight=_to_nz(layer.o_proj.weight),
        conv1d_weight=conv_weight,
    )


def find_mega_kda_op():
    for package in ("cann_ops_transformer_mega_kda_arch35", "cann_ops_transformer"):
        if find_spec(package) is not None:
            import_module(package)
            break
    else:
        return None
    try:
        op = torch.ops.cann_ops_transformer.mega_recurrent_kda.default
    except AttributeError:
        return None
    schema = str(op._schema)
    if "output_gate_a_projection_weight" not in schema or "output_gate_b_projection_weight" not in schema:
        return None
    return op


def get_mega_kda_op():
    op = find_mega_kda_op()
    if op is None:
        raise RuntimeError("GLM Mega KDA requires the low-rank gate ABI from ops-transformer PR 16634")
    return op


def run_mega_kda_decode(layer: Any, hidden_states: torch.Tensor, metadata: Any) -> torch.Tensor:
    weights = layer._mega_kda_weights
    records = layer.recoverssm_records
    if weights is None or not isinstance(records, MegaKDARecoverSSMRecords):
        raise RuntimeError("GLM Mega KDA weights or RecoverSSM records are not initialized")
    batch = metadata.num_spec_decodes
    total_tokens = batch * MEGA_KDA_SPEC_QUERY_LEN
    conv_state, checkpoint_state = layer.kv_cache
    records.ensure_pages(
        checkpoint_state.shape[0],
        layer.local_num_heads,
        MEGA_KDA_SPEC_QUERY_LEN,
        layer.head_dim,
        checkpoint_state.device,
    )
    if layer._conv_state_dim_first:
        conv_state = conv_state.transpose(-1, -2)
    operator = getattr(layer, "_mega_kda_operator", None) or get_mega_kda_op()
    output, _, _, _, _ = operator(
        hidden_states=hidden_states[:total_tokens],
        qkv_projection_weight=weights.qkv_projection_weight,
        decay_projection_a_weight=weights.decay_projection_a_weight,
        decay_projection_b_weight=weights.decay_projection_b_weight,
        beta_projection_weight=weights.beta_projection_weight,
        output_gate_projection_weight=None,
        output_norm_weight=weights.output_norm_weight,
        output_projection_weight=weights.output_projection_weight,
        conv1d_weight=weights.conv1d_weight,
        conv_state=conv_state,
        checkpoint_state=checkpoint_state,
        replay_u=records.correction,
        replay_k=records.replay_k,
        replay_decay=records.replay_decay,
        state_indices=metadata.spec_state_indices_tensor[:batch, 0],
        conv_num_accepted_tokens=metadata.num_accepted_tokens[:batch],
        A_log=layer.A_log.reshape(-1),
        dt_bias=layer.dt_bias.view(layer.local_num_heads, layer.head_dim),
        scale=layer.head_dim**-0.5,
        lower_bound=layer.kda_lower_bound,
        rms_norm_eps=float(layer.o_norm.eps),
        cu_seqlens=metadata.spec_query_start_loc[: batch + 1],
        output_gate_a_projection_weight=weights.output_gate_a_projection_weight,
        output_gate_b_projection_weight=weights.output_gate_b_projection_weight,
    )
    return output
