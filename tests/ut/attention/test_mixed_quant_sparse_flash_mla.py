# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.attention import mixed_quant_sparse_flash_mla as mqsmla


def test_ori_fp8_metadata_uses_library_mode_one():
    metadata_op = MagicMock(return_value=torch.empty(2048, dtype=torch.int32))
    with patch.object(mqsmla, "_get_mixed_quant_sparse_flash_mla_ops", return_value=(MagicMock(), metadata_op)):
        result = mqsmla.mixed_quant_sparse_flash_mla_metadata_ori_fp8(
            num_heads_q=8,
            num_heads_kv=1,
            head_dim=512,
            ori_topk=16,
            cmp_topk=0,
            device="npu:0",
        )

    assert result is metadata_op.return_value
    kwargs = metadata_op.call_args.kwargs
    assert kwargs["quant_mode"] == 1
    assert kwargs["rope_head_dim"] == 64
    assert kwargs["has_ori_kv"] is True
    assert kwargs["has_cmp_kv"] is False
    assert kwargs["ori_topk"] == 16
    assert kwargs["layout_q"] == "TND"
    assert kwargs["layout_kv"] == "PA_BBND"
    assert "device" not in kwargs


def test_ori_fp8_attention_preserves_sparse_indices_and_drops_cmp():
    attention_op = MagicMock(return_value=(torch.empty(1), torch.empty(0)))
    query = torch.empty(2, 8, 512, dtype=torch.bfloat16)
    indices = torch.tensor([[[3, 1]], [[2, -1]]], dtype=torch.int32)
    cache = torch.empty(1, 128, 1, 544, dtype=torch.uint8)
    with patch.object(mqsmla, "_get_mixed_quant_sparse_flash_mla_ops", return_value=(attention_op, MagicMock())):
        result = mqsmla.mixed_quant_sparse_flash_mla_ori_fp8(
            query,
            ori_kv=cache,
            ori_sparse_indices=indices,
            cmp_kv=torch.empty(1),
            cmp_sparse_indices=torch.empty(1),
            cmp_topk_length=torch.empty(1),
            return_softmax_lse=True,
        )

    assert result is attention_op.return_value
    assert attention_op.call_args.args[0] is query
    kwargs = attention_op.call_args.kwargs
    assert kwargs["quant_mode"] == 4
    assert kwargs["rope_head_dim"] == 64
    assert kwargs["ori_sparse_indices"] is indices
    assert kwargs["ori_kv"] is cache
    assert kwargs["ori_mask_mode"] == 0
    assert kwargs["topk_value_mode"] == 1
    assert kwargs["return_softmax_lse"] is False
    assert kwargs["sinks"] is None
    assert not any(name in kwargs for name in ("cmp_kv", "cmp_sparse_indices", "cmp_topk_length"))
