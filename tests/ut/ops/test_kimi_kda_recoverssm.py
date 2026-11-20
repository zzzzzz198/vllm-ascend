# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for request- and page-indexed RecoverSSM records."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.glm5next.ops.mega_kda import MegaKDARecoverSSMRecords
from vllm_ascend.ops.triton.kda import recoverssm


class RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def make_layer(*, pages=11, rows=4, heads=2, dim=4, capacity=4, mega=False):
    conv = torch.zeros(pages, 3 * heads * dim, capacity + 2, dtype=torch.bfloat16)
    state = torch.zeros(pages, heads, dim, dim)
    record_rows = pages if mega else rows
    correction = torch.zeros(record_rows, heads, capacity, dim)
    key_gate = torch.zeros(
        record_rows,
        heads,
        capacity,
        dim if mega else 2 * dim,
        dtype=torch.float16 if mega else torch.bfloat16,
    )
    records = SimpleNamespace(correction=correction, key_gate=key_gate)
    if mega:
        records.replay_decay = torch.zeros_like(correction)
    return SimpleNamespace(
        kv_cache=(conv, state),
        recoverssm_records=records,
        A_log=torch.zeros(heads),
        dt_bias=torch.zeros(heads * dim),
        gate_lower_bound=-5.0,
        local_num_heads=heads,
        head_dim=dim,
    )


@pytest.fixture(autouse=True)
def device_helpers(monkeypatch):
    monkeypatch.setattr(
        recoverssm,
        "triton",
        SimpleNamespace(next_power_of_2=lambda x: 1 << (x - 1).bit_length(), cdiv=lambda x, y: (x + y - 1) // y),
    )
    monkeypatch.setattr(recoverssm, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(recoverssm, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(recoverssm, "is_950", lambda: False)
    monkeypatch.setattr(recoverssm, "get_ub_size_bytes", lambda: 192 * 1024)


def test_context_binds_request_records_and_detects_rebinding():
    layers = [make_layer() for _ in range(2)]
    context = recoverssm.KDARecoverSSMCommitContext.create(layers, spec_query_len=4, max_num_reqs=4)
    assert context.is_bound_to(layers)
    assert context.correction_caches[0] is layers[0].recoverssm_records.correction
    old = layers[0].kv_cache
    layers[0].kv_cache = (old[0].clone(), *old[1:])
    assert not context.is_bound_to(layers)
    layers[0].kv_cache = old
    layers[0].recoverssm_records.key_gate = layers[0].recoverssm_records.key_gate.clone()
    assert not context.is_bound_to(layers)


def test_context_rejects_mixed_layout_and_insufficient_scratch():
    layer = make_layer()
    layer.kv_cache = (*layer.kv_cache, torch.zeros(1))
    with pytest.raises(ValueError, match="convolution and checkpoint caches"):
        recoverssm.KDARecoverSSMCommitContext.create([make_layer(), layer], spec_query_len=4, max_num_reqs=4)
    with pytest.raises(ValueError, match="record capacity"):
        recoverssm.KDARecoverSSMCommitContext.create([make_layer(rows=2)], spec_query_len=4, max_num_reqs=4)


def test_verify_accepts_raw_beta_and_request_records(monkeypatch):
    layer = make_layer()
    records = layer.recoverssm_records
    q = torch.randn(1, 5, 2, 4, dtype=torch.bfloat16)
    beta = torch.randn(1, 5, 2, dtype=q.dtype)
    offsets = torch.tensor([0, 3, 5], dtype=torch.int32)
    indices = torch.tensor([7, 3], dtype=torch.int32)
    kernel = RecordingKernel()
    monkeypatch.setattr(recoverssm, "_kda_recoverssm_verify_kernel", kernel)
    out = recoverssm.kda_recoverssm_verify(
        q,
        q,
        q,
        q,
        beta,
        layer.A_log,
        layer.dt_bias,
        -5.0,
        layer.kv_cache[1],
        records.correction,
        records.key_gate,
        offsets,
        indices,
        4,
    )
    assert out.shape == q.shape
    assert len(kernel.calls) == 1
    assert kernel.calls[0][1][4] is beta


def test_commit_launches_for_request_records(monkeypatch):
    context = recoverssm.KDARecoverSSMCommitContext.create([make_layer()], spec_query_len=4, max_num_reqs=4)
    plan, compact, commit = RecordingKernel(), RecordingKernel(), RecordingKernel()
    monkeypatch.setattr(recoverssm, "_prepare_commit_plan_kernel", plan)
    monkeypatch.setattr(recoverssm, "_compact_conv_state_kernel", compact)
    monkeypatch.setattr(recoverssm, "_commit_kda_state_kernel", commit)
    context.commit(
        torch.tensor([2, 1], dtype=torch.int32),
        torch.tensor([7, 3], dtype=torch.int32),
        torch.tensor([0, 3, 5], dtype=torch.int32),
    )
    assert len(plan.calls) == len(compact.calls) == len(commit.calls) == 1


def test_mega_verify_writes_separate_normalized_key_and_decay(monkeypatch):
    layer = make_layer(mega=True)
    records = layer.recoverssm_records
    q = torch.randn(1, 5, 2, 4, dtype=torch.bfloat16)
    beta = torch.randn(1, 5, 2, dtype=q.dtype)
    kernel = RecordingKernel()
    monkeypatch.setattr(recoverssm, "_kda_recoverssm_verify_kernel", kernel)

    out = recoverssm.kda_recoverssm_verify(
        q,
        q,
        q,
        q,
        beta,
        layer.A_log,
        layer.dt_bias,
        -5.0,
        layer.kv_cache[1],
        records.correction,
        records.key_gate,
        torch.tensor([0, 3, 5], dtype=torch.int32),
        torch.tensor([7, 3], dtype=torch.int32),
        4,
        decay_cache=records.replay_decay,
    )

    assert out.shape == q.shape
    args, kwargs = kernel.calls[0][1:]
    assert args[10] is records.replay_decay
    assert kwargs["SEPARATE_KEY_DECAY"] is True


def test_mega_commit_uses_precomputed_decay_and_checks_rebinding(monkeypatch):
    layer = make_layer(mega=True)
    context = recoverssm.KDARecoverSSMCommitContext.create([layer], spec_query_len=4, max_num_reqs=4)
    assert context.is_bound_to([layer])
    assert context.decay_caches[0] is layer.recoverssm_records.replay_decay
    plan, compact = RecordingKernel(), RecordingKernel()
    mega_commit, old_commit = RecordingKernel(), RecordingKernel()
    monkeypatch.setattr(recoverssm, "_prepare_commit_plan_kernel", plan)
    monkeypatch.setattr(recoverssm, "_compact_conv_state_kernel", compact)
    monkeypatch.setattr(recoverssm, "_commit_mega_kda_state_kernel", mega_commit)
    monkeypatch.setattr(recoverssm, "_commit_kda_state_kernel", old_commit)

    context.commit(
        torch.tensor([2, 1], dtype=torch.int32),
        torch.tensor([7, 3], dtype=torch.int32),
        torch.tensor([0, 3, 5], dtype=torch.int32),
    )

    assert len(plan.calls) == len(compact.calls) == len(mega_commit.calls) == 1
    assert not old_commit.calls
    args, kwargs = mega_commit.calls[0][1:]
    assert args[9] is layer.recoverssm_records.replay_decay
    assert "USE_LOWER_BOUND" not in kwargs
    layer.recoverssm_records.replay_decay = layer.recoverssm_records.replay_decay.clone()
    assert not context.is_bound_to([layer])


def test_context_binds_empty_mega_records_to_checkpoint_pages():
    layer = make_layer(mega=True)
    records = MegaKDARecoverSSMRecords()
    layer.recoverssm_records = records

    context = recoverssm.KDARecoverSSMCommitContext.create([layer], spec_query_len=4, max_num_reqs=4)

    assert records.correction.shape == (11, 2, 4, 4)
    assert records.replay_k.dtype == torch.float16
    assert records.replay_decay.dtype == torch.float32
    assert context.correction_caches[0] is records.correction
    assert context.kg_caches[0] is records.replay_k
    assert context.decay_caches[0] is records.replay_decay
    assert context.is_bound_to([layer])
