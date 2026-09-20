# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SparseFlashMla 适配层（ori_smla 分支）。

本分支在原 fix_smla 实现的旁边，加入一条"私仓 0806"风格的调用路径，用于做单变量
对比实验。两条路径底层是同一个 CANN 算子 aclnnSparseFlashMla，差别只在：

  1) 绑定入口
       cann : torch.ops.cann_ops_transformer.sparse_flash_mla   （fix_smla 现用）
       ori  : torch.ops._C_ascend.npu_sparse_flash_mla           （私仓 device_op.py:1788）
     两者都通过 EXEC_NPU_CMD 调 aclnnSparseFlashMla。
  2) ori_win_left / ori_win_right
       fix_smla 传 -1 / -1（不限窗口）；私仓传 0 / 0。
  3) ori_topk_length 的来源
       fix_smla 由 positions 解析式推算；私仓由实际 indices 计数
       （(indices >= 0).sum(-1)，见私仓 device_op.py:1786）。

三个开关全部走环境变量，不需要重新编译即可逐条对照：

  SMLA_CALL_PATH=cann|ori            默认 cann（即 fix_smla 原行为）
  SMLA_ORI_WIN_LEFT / SMLA_ORI_WIN_RIGHT   默认 0 / 0（ori 路径专用）
  SMLA_ORI_CMP_MASK_MODE              默认 3（ori 路径专用，私仓取值）
  SMLA_ORI_TOPK_FROM_INDICES=0|1      默认 1（ori 路径下是否改用 indices 计数）
"""

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
import os
from typing import Any

import torch


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_CALL_PATH = os.environ.get("SMLA_CALL_PATH", "cann").strip().lower()
_ORI_WIN_LEFT = _env_int("SMLA_ORI_WIN_LEFT", 0)
_ORI_WIN_RIGHT = _env_int("SMLA_ORI_WIN_RIGHT", 0)
_ORI_CMP_MASK_MODE = _env_int("SMLA_ORI_CMP_MASK_MODE", 3)
_ORI_TOPK_FROM_INDICES = os.environ.get("SMLA_ORI_TOPK_FROM_INDICES", "1") != "0"


def use_ori_path() -> bool:
    return _CALL_PATH == "ori"


@lru_cache
def _get_sparse_flash_mla_ops() -> tuple[Callable, Callable]:
    """Load SparseFlashMla operators without importing vllm_ascend.ops."""
    try:
        import_module("cann_ops_transformer")
        namespace = torch.ops.cann_ops_transformer
        return namespace.sparse_flash_mla, namespace.sparse_flash_mla_metadata
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "DeepSeek-V4 BF16 KV on Ascend A5 requires SparseFlashMla from a matching CANN 9.2 toolkit."
        ) from exc


@lru_cache
def _get_ascend_custom_ops() -> tuple[Callable, Callable]:
    """私仓路径：vllm-ascend 自带的 _C_ascend 绑定（同样落到 aclnnSparseFlashMla）。"""
    import_module("torch_npu")
    namespace = torch.ops._C_ascend
    return namespace.npu_sparse_flash_mla, namespace.npu_sparse_flash_mla_metadata


def _add_compressed_kv_lengths(kwargs: dict[str, Any]) -> None:
    cmp_ratio = kwargs.get("cmp_ratio") or 0
    seqused_ori_kv = kwargs.get("seqused_ori_kv")
    if cmp_ratio <= 1 or seqused_ori_kv is None:
        return
    kwargs.setdefault("seqused_cmp_kv", seqused_ori_kv // cmp_ratio)
    kwargs.setdefault("cmp_residual_kv", seqused_ori_kv % cmp_ratio)
    if kwargs.get("max_seqlen_cmp_kv") is None and kwargs.get("max_seqlen_ori_kv") is not None:
        kwargs["max_seqlen_cmp_kv"] = kwargs["max_seqlen_ori_kv"] // cmp_ratio


def _drop_paged_kv_cu_seqlens(kwargs: dict[str, Any]) -> None:
    """Drop KV cu_seqlens; SparseFlashMla only accepts them when layout_kv is TND.

    This adapter always uses PA_BBND paged cache. Passing cu_seqlens_ori_kv or
    cu_seqlens_cmp_kv raises EZ0037 from aclnnSparseFlashMla.
    """
    kwargs.pop("cu_seqlens_ori_kv", None)
    kwargs.pop("cu_seqlens_cmp_kv", None)


def sparse_flash_mla_metadata(**kwargs):
    """Adapt existing DSA metadata kwargs to SparseFlashMla BF16 KV."""
    kwargs.pop("device", None)
    kwargs.pop("kv_quant_mode", None)
    # This adapter is only selected for the BF16 paged-KV path. SparseFlashMla
    # accepts PA_BBND for this cache; PA_ND belongs to the FP8 quantized op.
    kwargs["layout_kv"] = "PA_BBND"
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    if "max_seqlen_kv" in kwargs:
        kwargs["max_seqlen_ori_kv"] = kwargs.pop("max_seqlen_kv")
    _drop_paged_kv_cu_seqlens(kwargs)
    _add_compressed_kv_lengths(kwargs)
    if use_ori_path():
        _, metadata_op = _get_ascend_custom_ops()
        return metadata_op(**kwargs)
    _, metadata_op = _get_sparse_flash_mla_ops()
    return metadata_op(**kwargs)


def _ensure_sinks(kwargs: dict[str, Any]) -> None:
    """Reject calls that omit the required per-head sinks tensor.

    SparseFlashMla needs a per-head float32 sinks tensor and the caller must
    own it. A tensor allocated here would only be referenced by these kwargs,
    so ACL Graph capture bakes in an address that the allocator reuses as soon
    as the call returns; replaying that graph then reads freed memory and traps
    inside the kernel (507011). Device errors are reported asynchronously, so
    the failure surfaces at an unrelated synchronize() rather than at the
    operator that caused it, which makes it very hard to attribute.
    """
    if kwargs.get("sinks") is None:
        raise ValueError(
            "SparseFlashMla requires a caller-owned per-head sinks tensor; "
            "allocating one here would not survive ACL Graph capture."
        )


def _ori_apply_parameter_style(kwargs: dict[str, Any]) -> None:
    """私仓 0806 的调用参数风格（device_op.py:1788 附近）。"""
    kwargs["ori_win_left"] = _ORI_WIN_LEFT
    kwargs["ori_win_right"] = _ORI_WIN_RIGHT
    kwargs["cmp_mask_mode"] = _ORI_CMP_MASK_MODE
    if _ORI_TOPK_FROM_INDICES:
        indices = kwargs.get("ori_sparse_indices")
        if indices is not None:
            # 私仓：ori_topk_length = (topk_indices >= 0).sum(dim=-1, keepdim=True)
            kwargs["ori_topk_length"] = (
                (indices >= 0).sum(dim=-1, keepdim=True).to(torch.int32)
            )


def sparse_flash_mla(q: torch.Tensor, **kwargs):
    """Adapt existing DSA attention kwargs to SparseFlashMla BF16 KV."""
    kwargs.pop("kv_quant_mode", None)
    kwargs.pop("tile_size", None)
    kwargs.pop("rope_head_dim", None)
    kwargs["layout_kv"] = "PA_BBND"
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    _drop_paged_kv_cu_seqlens(kwargs)
    _add_compressed_kv_lengths(kwargs)
    _ensure_sinks(kwargs)
    if use_ori_path():
        _ori_apply_parameter_style(kwargs)
        attention_op, _ = _get_ascend_custom_ops()
        return attention_op(q, **kwargs)
    attention_op, _ = _get_sparse_flash_mla_ops()
    return attention_op(q, **kwargs)
