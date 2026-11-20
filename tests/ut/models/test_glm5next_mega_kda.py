# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.models.glm5next.kda as model_kda
from vllm_ascend.models.glm5next import recoverssm_metadata
from vllm_ascend.models.glm5next.ops import mega_kda


def test_mega_records_keep_key_and_decay_in_separate_dtypes():
    records = mega_kda.MegaKDARecoverSSMRecords.allocate(3, 64, 6, 128, torch.device("cpu"))

    assert records.correction.shape == (3, 64, 6, 128)
    assert records.replay_k.shape == records.replay_decay.shape == records.correction.shape
    assert records.correction.dtype == records.replay_decay.dtype == torch.float32
    assert records.replay_k.dtype == torch.float16
    assert records.key_gate is records.replay_k


def test_mega_weights_split_fused_projection_without_changing_source(monkeypatch):
    hidden_size, heads, head_dim = 3, 2, 2
    projection_size = heads * head_dim
    input_weight = torch.arange(18 * hidden_size, dtype=torch.bfloat16).reshape(18, hidden_size)
    original = input_weight.clone()
    layer = SimpleNamespace(
        local_projection_size=projection_size,
        local_num_heads=heads,
        head_dim=head_dim,
        in_proj_qkvbfg_a=SimpleNamespace(weight=input_weight),
        f_b_proj=SimpleNamespace(weight=torch.ones(projection_size, head_dim, dtype=torch.bfloat16)),
        g_b_proj=SimpleNamespace(weight=torch.full((projection_size, head_dim), 2, dtype=torch.bfloat16)),
        o_proj=SimpleNamespace(weight=torch.full((hidden_size, projection_size), 3, dtype=torch.bfloat16)),
        o_norm=SimpleNamespace(weight=torch.ones(head_dim, dtype=torch.float16)),
        _merged_conv_weight=torch.ones(4, 3 * projection_size, dtype=torch.bfloat16),
    )
    monkeypatch.setattr(mega_kda, "_to_nz", lambda weight: weight.detach().clone())

    weights = mega_kda.prepare_mega_kda_weights(layer)

    torch.testing.assert_close(weights.qkv_projection_weight, input_weight[: 3 * projection_size])
    torch.testing.assert_close(weights.beta_projection_weight, input_weight[12:14])
    torch.testing.assert_close(weights.decay_projection_a_weight, input_weight[14:16])
    torch.testing.assert_close(weights.output_gate_a_projection_weight, input_weight[16:18])
    assert weights.output_norm_weight.dtype == torch.float32
    assert weights.conv1d_weight is layer._merged_conv_weight
    torch.testing.assert_close(input_weight, original)


def test_mega_weights_reject_quantized_projection(monkeypatch):
    layer = SimpleNamespace(
        local_projection_size=4,
        local_num_heads=2,
        head_dim=2,
        in_proj_qkvbfg_a=SimpleNamespace(weight=torch.ones(18, 3, dtype=torch.float8_e4m3fn)),
        f_b_proj=SimpleNamespace(weight=torch.ones(4, 2)),
        g_b_proj=SimpleNamespace(weight=torch.ones(4, 2)),
        o_proj=SimpleNamespace(weight=torch.ones(3, 4)),
    )
    monkeypatch.setattr(mega_kda, "_to_nz", lambda weight: weight)
    with pytest.raises(ValueError, match="BF16 projection weights"):
        mega_kda.prepare_mega_kda_weights(layer)


def test_mega_decode_maps_th_metadata_and_low_rank_gate(monkeypatch):
    records = mega_kda.MegaKDARecoverSSMRecords()
    weights = SimpleNamespace(
        **{
            name: torch.ones(1)
            for name in (
                "qkv_projection_weight",
                "decay_projection_a_weight",
                "decay_projection_b_weight",
                "beta_projection_weight",
                "output_gate_a_projection_weight",
                "output_gate_b_projection_weight",
                "output_norm_weight",
                "output_projection_weight",
                "conv1d_weight",
            )
        }
    )
    conv_state = torch.zeros(8, 8, 12, dtype=torch.bfloat16)
    checkpoint = torch.zeros(8, 1, 4, 4)
    hidden = torch.zeros(12, 3, dtype=torch.bfloat16)
    expected = torch.ones_like(hidden)
    layer = SimpleNamespace(
        _mega_kda_weights=weights,
        recoverssm_records=records,
        kv_cache=(conv_state, checkpoint),
        _conv_state_dim_first=False,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(4),
        local_num_heads=1,
        head_dim=4,
        kda_lower_bound=-4.0,
        o_norm=SimpleNamespace(eps=1e-6),
    )
    metadata = SimpleNamespace(
        num_spec_decodes=2,
        spec_state_indices_tensor=torch.tensor([[3], [4]], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([2, 1, 9], dtype=torch.int32),
        spec_query_start_loc=torch.tensor([0, 6, 12, 18], dtype=torch.int32),
    )
    calls = []

    def op(**kwargs):
        calls.append(kwargs)
        return expected, kwargs["conv_state"], kwargs["replay_u"], kwargs["replay_k"], kwargs["replay_decay"]

    monkeypatch.setattr(mega_kda, "get_mega_kda_op", lambda: op)
    output = mega_kda.run_mega_kda_decode(layer, hidden, metadata)

    assert output is expected
    assert records.correction.shape == (8, 1, 6, 4)
    args = calls[0]
    assert args["hidden_states"].data_ptr() == hidden.data_ptr()
    assert args["hidden_states"].shape == (12, 3)
    assert args["output_gate_projection_weight"] is None
    assert args["output_gate_a_projection_weight"] is weights.output_gate_a_projection_weight
    assert args["output_gate_b_projection_weight"] is weights.output_gate_b_projection_weight
    assert args["conv_state"] is conv_state and args["checkpoint_state"] is checkpoint
    assert args["replay_u"] is records.correction
    assert args["replay_k"] is records.replay_k and args["replay_decay"] is records.replay_decay
    torch.testing.assert_close(args["state_indices"], torch.tensor([3, 4], dtype=torch.int32))
    torch.testing.assert_close(args["conv_num_accepted_tokens"], torch.tensor([2, 1], dtype=torch.int32))
    torch.testing.assert_close(args["cu_seqlens"], torch.tensor([0, 6, 12], dtype=torch.int32))
    assert args["scale"] == 0.5 and args["lower_bound"] == -4.0 and args["rms_norm_eps"] == 1e-6


@pytest.mark.parametrize("use_output_buffer", [False, True])
def test_glm_forward_uses_mega_before_separate_projections(monkeypatch, use_output_buffer):
    layer = model_kda.Glm5NextLinearAttention.__new__(model_kda.Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer"
    layer._mega_kda_enabled = True
    hidden = torch.zeros(12, 4096, dtype=torch.bfloat16)
    expected = torch.ones_like(hidden)
    metadata = SimpleNamespace(mega_kda_decode=True)
    context = SimpleNamespace(attn_metadata={"layer": metadata})
    monkeypatch.setattr(model_kda, "get_forward_context", lambda: context)
    events = []
    monkeypatch.setattr(model_kda, "wait_for_kv_layer_from_connector", lambda prefix: events.append(("load", prefix)))
    monkeypatch.setattr(model_kda, "record_attention_compute_start", lambda: events.append("compute"))
    monkeypatch.setattr(model_kda, "maybe_save_kv_layer_to_connector", lambda *args: events.append("save"))
    monkeypatch.setattr(
        model_kda,
        "run_mega_kda_decode",
        lambda owner, inputs, meta: (events.append(("mega", owner, inputs, meta)) or expected),
    )

    output_buffer = torch.empty_like(hidden) if use_output_buffer else None
    result = layer._forward(hidden, output_buffer)

    assert events == [("load", "layer"), "compute", ("mega", layer, hidden, metadata), "save"]
    if use_output_buffer:
        assert result is None
        torch.testing.assert_close(output_buffer, expected)
    else:
        assert result is expected


@pytest.mark.parametrize(
    ("num_spec", "lengths", "expected"),
    [(5, [6], True), (5, [6, 6], True), (5, [6, 2], False), (3, [4], False)],
)
def test_mega_metadata_requires_pure_uniform_six_token_decode(monkeypatch, num_spec, lengths, expected):
    builder = recoverssm_metadata.AscendKDARecoverSSMMetadataBuilder.__new__(
        recoverssm_metadata.AscendKDARecoverSSMMetadataBuilder
    )
    builder.num_spec = num_spec
    builder.use_full_cuda_graph = False
    monkeypatch.setattr(builder, "_stage_spec", lambda metadata, batch_size: None)
    batch = len(lengths)
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    m = SimpleNamespace(num_reqs=batch, query_start_loc_cpu=starts)
    metadata = SimpleNamespace(
        num_spec_decodes=batch,
        num_spec_decode_tokens=sum(lengths),
        num_prefills=0,
        num_decodes=0,
        spec_state_indices_tensor=torch.arange(batch, dtype=torch.int32)[:, None],
        spec_query_start_loc=starts,
        num_accepted_tokens=torch.ones(batch, dtype=torch.int32),
    )

    builder._build_conv_metadata(m, metadata, None)

    assert metadata.mega_kda_decode is expected


def test_missing_cann_package_preserves_legacy_path(monkeypatch):
    monkeypatch.setattr(mega_kda, "find_spec", lambda package: None)
    assert mega_kda.find_mega_kda_op() is None
