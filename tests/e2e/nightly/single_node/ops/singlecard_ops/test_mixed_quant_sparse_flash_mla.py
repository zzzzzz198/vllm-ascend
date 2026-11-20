# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.sfa_v1 import MQSMLA_METADATA_SIZE, sparse_mqsmla
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType


def _metadata(block_table, length_buffer, valid_mask):
    return SimpleNamespace(
        block_table=block_table,
        block_size=128,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32, device="npu"),
        seq_lens=torch.tensor([256], dtype=torch.int32, device="npu"),
        max_query_len=3,
        max_seq_len=256,
        smla_metadata=torch.empty(MQSMLA_METADATA_SIZE, dtype=torch.int32, device="npu"),
        smla_topk_length=length_buffer,
        smla_valid_mask=valid_mask,
    )


def _run(cache, query, indices, metadata):
    output = sparse_mqsmla(query, cache, indices, metadata, 1.0)
    torch.npu.synchronize()
    return output


def test_a5_mqsmla_fp8_c8_noncontiguous_indices_and_metadata_refresh():
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("MQSMLA FP8 C8 integration requires Ascend 950/A5")

    query = torch.zeros(3, 4, 512, dtype=torch.bfloat16, device="npu")
    cache = torch.empty(2, 128, 1, 544, dtype=torch.uint8, device="npu")
    row_values = torch.arange(256, dtype=torch.float32, device="npu").remainder(128) + 1
    cache[..., :512].view(torch.float8_e4m3fn).copy_(row_values.view(2, 128, 1, 1))
    cache[..., 512:].view(torch.bfloat16).fill_(1.0)

    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device="npu")
    length_buffer = torch.zeros(4, 1, dtype=torch.int32, device="npu")
    valid_mask = torch.tensor([[True], [True], [True], [False]], device="npu")
    metadata = _metadata(block_table, length_buffer, valid_mask)
    metadata_address = metadata.smla_metadata.data_ptr()

    padded_indices = torch.full((3, 1, 4), -1, dtype=torch.int32, device="npu")
    padded_indices[0, 0, :2] = torch.tensor([0, 128], device="npu")
    padded_indices[1, 0, :3] = torch.tensor([1, 129, 130], device="npu")
    indices = padded_indices[..., :3]
    assert not indices.is_contiguous()

    output = _run(cache, query, indices, metadata)
    expected = torch.tensor([1.0, 7.0 / 3.0, 0.0], dtype=torch.bfloat16, device="npu")
    torch.testing.assert_close(output, expected.view(3, 1, 1).expand_as(output))
    assert metadata.smla_metadata.data_ptr() == metadata_address

    padded_indices.fill_(-1)
    padded_indices[0, 0, 0] = 130
    padded_indices[1, 0, :2] = torch.tensor([2, 3], device="npu")
    output = _run(cache, query, indices, metadata)
    expected = torch.tensor([3.0, 2.5, 0.0], dtype=torch.bfloat16, device="npu")
    torch.testing.assert_close(output, expected.view(3, 1, 1).expand_as(output))
    assert metadata.smla_metadata.data_ptr() == metadata_address
