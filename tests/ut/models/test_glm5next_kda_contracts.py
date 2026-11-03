# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.models.glm5next.ops.kda as kda
import vllm_ascend.ops.kda as kda_ops


@pytest.mark.parametrize(
    "accepted,qkv_padding,direct_output",
    [(None, 0, False), (None, 64, False), ([1, 2, 1], 0, False), ([1, 2, 1], 64, False), (None, 64, True)],
)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_recurrent_raw_gates_rollback_slots_and_padding(monkeypatch, accepted, qkv_padding, direct_output, state_dtype):
    q, k, v = (torch.ones(1, 4, 1, 128 + qkv_padding * i, dtype=torch.bfloat16)[..., :128] for i in (1, 2, 3))
    gate = q * 2
    beta = torch.zeros(1, 4, 1, dtype=torch.bfloat16)
    state = torch.zeros(8, 1, 128, 128, dtype=state_dtype)
    starts = torch.tensor([0, 1, 3], dtype=torch.int32)
    slots = torch.tensor([[2, 3], [5, 6], [0, 0]], dtype=torch.int32)
    accepted_tensor = None if accepted is None else torch.tensor(accepted, dtype=torch.int32)

    def recurrent(q_arg, k_arg, v_arg, gate_arg, beta_arg, state_arg, *, cu_seqlens, ssm_state_indices, **kwargs):
        # Preserve each view's strides and storage without materializing Q/K/V.
        assert q_arg is q and k_arg is k and v_arg is v
        assert kwargs["use_gate_in_kernel"] and kwargs["use_beta_sigmoid_in_kernel"]
        assert kwargs["safe_gate"] and kwargs["lower_bound"] == -4.0
        assert state_arg is state
        torch.testing.assert_close(gate_arg, gate)
        torch.testing.assert_close(beta_arg, beta)
        assert cu_seqlens is starts
        torch.testing.assert_close(ssm_state_indices, slots[:2])
        assert kwargs["state_v_first"] and kwargs["inplace_final_state"]
        assert not kwargs["output_final_state"]
        if accepted_tensor is not None:
            torch.testing.assert_close(kwargs["num_accepted_tokens"], accepted_tensor[:2])
        result = q_arg.clone()
        result[:, 3:] = float("nan")
        return result, None

    monkeypatch.setattr(kda_ops, "recurrent_kda", recurrent)
    destination = torch.full((1, 8, 1, 128), float("nan"), dtype=q.dtype) if direct_output else None

    def writeback(source: torch.Tensor, target: torch.Tensor, ends: torch.Tensor) -> None:
        assert destination is not None
        assert target is destination and ends is starts
        assert torch.isnan(source[:, 3:]).all()
        target.zero_()
        target[:, :3].copy_(source[:, :3])

    monkeypatch.setattr(kda, "write_recurrent_output", writeback)
    out = kda.recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        state,
        starts,
        slots,
        torch.zeros(1),
        torch.zeros(128),
        -4,
        accepted_tensor,
        output_buffer=destination,
    )
    if direct_output:
        assert out is destination
    torch.testing.assert_close(out[:, :3], q[:, :3])
    if direct_output:
        # write_recurrent_output zeroes the captured buffer before copying.
        assert torch.count_nonzero(out[:, 3:]) == 0
    else:
        # The recurrent kernel writes valid rows only; padding rows are owned
        # by the caller (index_copy_ / write_recurrent_output), so the sentinel
        # NaNs written by the kernel must survive unchanged.
        assert torch.isnan(out[:, 3:]).all()


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compact", [False, True])
def test_chunk_uses_host_descriptors_and_preserves_vk_cache(monkeypatch, state_dtype, compact):
    q = torch.ones(1, 3, 1, 128, dtype=torch.bfloat16)
    state = torch.arange(4 * 128 * 128, dtype=torch.float32).reshape(4, 1, 128, 128).to(state_dtype)
    saved = state.clone()
    indices = torch.tensor([3, 1], dtype=torch.int32)
    has_initial = torch.tensor([True, False])
    keep = torch.tensor([0]) if compact else None
    metadata = SimpleNamespace(
        keep_meta=keep,
        cu_seqlens_host=(0, 3) if compact else (0, 1, 3),
        cu_seqlens_kern=None,
        chunk_indices_chunk64_host=(0, 0) if compact else (0, 0, 1, 0),
    )

    def chunk(q_arg, k_arg, v, g, beta, scale, chunk_size, **kwargs):
        assert q_arg is q and k_arg is q
        assert kwargs["epsilon"] == kda.KDA_L2NORM_EPSILON
        assert kwargs["use_qk_l2norm_in_kernel"]
        assert kwargs["state_v_first"] and kwargs["use_gate_in_kernel"] and kwargs["safe_gate"]
        assert kwargs["cu_seqlens"] is metadata.cu_seqlens_host
        assert kwargs["chunk_indices"] is metadata.chunk_indices_chunk64_host
        assert kwargs["initial_state"].dtype == torch.float32
        torch.testing.assert_close(kwargs["initial_state"][0], saved[3].float())
        if not compact:
            assert torch.count_nonzero(kwargs["initial_state"][1]) == 0
        torch.testing.assert_close(beta, torch.full_like(beta, 0.5))
        return v, torch.full_like(kwargs["initial_state"], 17)

    def unexpected_l2norm(x):
        raise AssertionError("GLM V2 must normalize Q/K inside ChunkKDA")

    monkeypatch.setattr(kda_ops, "chunk_kda_fwd", chunk)
    monkeypatch.setattr("vllm.third_party.flash_linear_attention.ops.l2norm.l2norm_fwd", unexpected_l2norm)
    out = kda.chunk_kda(
        q, q, q, q, torch.zeros(1, 3, 1), state, indices, has_initial, metadata, torch.zeros(1), torch.zeros(128), -4
    )
    torch.testing.assert_close(out, q)
    assert (state[3] == 17).all()
    torch.testing.assert_close(state[0], saved[0])
    torch.testing.assert_close(state[2], saved[2])
    if compact:
        torch.testing.assert_close(state[1], saved[1])
    else:
        assert (state[1] == 17).all()


def test_run_chunk_kda_defaults_to_external_l2norm(monkeypatch):
    q = torch.ones(1, 1, 1, 128, dtype=torch.bfloat16)
    state = torch.zeros(1, 1, 128, 128, dtype=torch.float32)
    norm_calls = []

    def norm(x):
        norm_calls.append(x)
        return x

    def chunk(q_arg, k_arg, v, g, beta, scale, chunk_size, **kwargs):
        assert "epsilon" not in kwargs
        assert "use_qk_l2norm_in_kernel" not in kwargs
        return v, state

    monkeypatch.setattr("vllm.third_party.flash_linear_attention.ops.l2norm.l2norm_fwd", norm)
    monkeypatch.setattr(kda_ops, "chunk_kda_fwd", chunk)
    output, final_state = kda_ops.run_chunk_kda(
        q,
        q,
        q,
        q,
        torch.zeros(1, 1, 1),
        state,
        (0, 1),
        (0, 0),
        torch.zeros(1),
        torch.zeros(128),
        lower_bound=-4,
    )

    assert len(norm_calls) == 2
    assert norm_calls[0] is q and norm_calls[1] is q
    assert output is q
    assert final_state is state
