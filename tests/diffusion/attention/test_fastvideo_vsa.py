# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
import types
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
)
from vllm_omni.diffusion.attention.backends.fastvideo_vsa import (
    FastVideoVSABackend,
    FastVideoVSAImpl,
    validate_local_ulysses_config,
)
from vllm_omni.diffusion.attention.backends.registry import (
    DiffusionAttentionBackendEnum,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_fastvideo_vsa_backend_is_registered():
    assert DiffusionAttentionBackendEnum.FASTVIDEO_VSA.get_path().endswith("fastvideo_vsa.FastVideoVSABackend")


def test_auto_keeps_wan_tile256_on_fastvideo(monkeypatch):
    from vllm_omni.diffusion.attention.ops import flashinfer_block_sparse

    def fail(*args):
        raise AssertionError("Wan must not probe the FlashInfer tile64 kernel")

    monkeypatch.setattr(flashinfer_block_sparse, "require_flashinfer_sparse", fail)
    impl = FastVideoVSAImpl(num_heads=2, head_size=128, softmax_scale=128**-0.5)
    assert impl.provider == "fastvideo" and impl.precision == "bf16"


def test_fastvideo_vsa_reports_missing_optional_kernel(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ImportError, match="fastvideo-kernel"):
        FastVideoVSABackend.validate_available()


def test_fastvideo_vsa_tiles_3d_sequence_and_untiles(monkeypatch):
    calls: dict[str, Any] = {}
    fake_module = types.ModuleType("fastvideo_kernel")

    def fake_video_sparse_attn_bshd(
        q,
        k,
        v,
        variable_block_sizes,
        q_variable_block_sizes,
        compress_attn_weight,
        topk,
        block_size,
    ):
        calls["q_shape"] = tuple(q.shape)
        calls["vbs"] = variable_block_sizes.detach().cpu().tolist()
        calls["q_vbs"] = q_variable_block_sizes.detach().cpu().tolist()
        calls["compress_sum"] = float(compress_attn_weight.detach().cpu().sum())
        calls["compress_shape"] = tuple(compress_attn_weight.shape)
        calls["topk"] = topk
        calls["block_size"] = block_size
        return q + k + v

    setattr(fake_module, "video_sparse_attn_bshd", fake_video_sparse_attn_bshd)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", fake_module)

    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={
            "topk": 1,
            "block_size": (4, 8, 8),
            "min_seq_len": 1,
            "disable_when_sp_active": False,
        },
    )
    query = torch.randn(1, 300, 2, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    attn_metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (3, 10, 10)})
    monkeypatch.setattr(impl, "_fallback_reason", lambda *args, **kwargs: None)

    output = impl.forward_cuda(query, key, value, attn_metadata)

    assert output.shape == query.shape
    assert calls["q_shape"] == (1, 1024, 2, 8)
    assert calls["vbs"] == [192, 48, 48, 12]
    assert calls["q_vbs"] == [192, 48, 48, 12]
    assert calls["compress_shape"] == (1, 1024, 2, 8)
    assert calls["compress_sum"] == 0.0
    assert calls["topk"] == 1
    assert calls["block_size"] == (4, 8, 8)


def test_fastvideo_vsa_uses_learned_gate_when_provided(monkeypatch):
    calls: dict[str, Any] = {}
    fake_module = types.ModuleType("fastvideo_kernel")

    def fake_video_sparse_attn_bshd(
        q,
        k,
        v,
        variable_block_sizes,
        q_variable_block_sizes,
        compress_attn_weight,
        topk,
        block_size,
    ):
        calls["compress_sum"] = float(compress_attn_weight.detach().cpu().sum())
        calls["compress_shape"] = tuple(compress_attn_weight.shape)
        return q + k + v

    setattr(fake_module, "video_sparse_attn_bshd", fake_video_sparse_attn_bshd)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", fake_module)

    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={
            "topk": 1,
            "block_size": (4, 8, 8),
            "min_seq_len": 1,
            "disable_when_sp_active": False,
        },
    )
    query = torch.randn(1, 300, 2, 8)
    gate = torch.ones_like(query)
    metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (3, 10, 10), "gate_compress": gate})
    monkeypatch.setattr(impl, "_fallback_reason", lambda *args, **kwargs: None)

    output = impl.forward_cuda(query, query, query, metadata)

    assert output.shape == query.shape
    assert calls["compress_shape"] == (1, 1024, 2, 8)
    # Only the 300 real tokens carry ones; padded gate slots stay zero.
    assert calls["compress_sum"] == gate.numel()


def test_fastvideo_vsa_falls_back_without_dit_shape(monkeypatch):
    calls: dict[str, Any] = {}

    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={
            "topk": 1,
            "block_size": (4, 8, 8),
            "min_seq_len": 1,
            "disable_when_sp_active": False,
        },
    )

    def fake_fallback(query, key, value, attn_metadata, reason):
        calls["reason"] = reason
        return torch.zeros_like(query)

    monkeypatch.setattr(impl, "_fallback", fake_fallback)

    query = torch.randn(1, 512, 2, 8)
    output = impl.forward_cuda(query, query, query, AttentionMetadata())

    assert output.shape == query.shape
    assert calls["reason"] == "vsa_dit_seq_shape metadata is required"


def test_fastvideo_vsa_falls_back_for_mask(monkeypatch):
    calls: dict[str, Any] = {}

    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={"min_seq_len": 1},
    )

    def fake_fallback(query, key, value, attn_metadata, reason):
        calls["reason"] = reason
        return torch.zeros_like(query)

    monkeypatch.setattr(impl, "_fallback", fake_fallback)

    query = torch.randn(1, 512, 2, 8)
    mask = torch.ones(1, 512, dtype=torch.bool)
    output = impl.forward_cuda(query, query, query, AttentionMetadata(attn_mask=mask))

    assert output.shape == query.shape
    assert calls["reason"]


def test_fastvideo_vsa_allows_topk_equal_to_num_blocks():
    query = torch.randn(1, 512, 2, 8)
    metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (4, 8, 16)})

    all_blocks = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={"topk": 2, "block_size": (4, 8, 8), "min_seq_len": 1},
    )
    too_many = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        backend_kwargs={"topk": 3, "block_size": (4, 8, 8), "min_seq_len": 1},
    )

    # CPU/float32 is rejected later, but k=N itself must not trigger fallback.
    assert all_blocks._fallback_reason(query, query, query, metadata) == "dtype torch.float32 is not supported"
    assert too_many._fallback_reason(query, query, query, metadata) == "topk 3 > num_blocks 2"


@pytest.mark.parametrize("learned_gate", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_local_fastwan_tile64_layout_gate_and_edge_padding(monkeypatch, learned_gate, packed):
    fake_module = types.ModuleType("fastvideo_kernel")
    calls = []

    def fake_video_sparse_attn(q, k, v, **kwargs):
        calls.append(kwargs)
        assert q.shape == k.shape == v.shape == (2, 3, 576, 8)
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert kwargs["block_size"] == (4, 4, 4) and kwargs["topk"] == 2
        sizes = kwargs["variable_block_sizes"]
        assert sizes.tolist() == [48, 48, 24, 48, 48, 24, 24, 24, 12]
        torch.testing.assert_close(sizes, kwargs["q_variable_block_sizes"])
        gate = kwargs["compress_attn_weight"]
        assert gate.shape == q.shape
        padding = torch.arange(64)[None, :] >= sizes[:, None]
        for tensor in (q, k, v, gate):
            assert torch.count_nonzero(tensor.reshape(2, 3, 9, 64, 8)[:, :, padding]) == 0
        return q + 2 * k + 3 * v + gate

    fake_module.video_sparse_attn = fake_video_sparse_attn
    monkeypatch.setitem(sys.modules, "fastvideo_kernel", fake_module)
    impl = FastVideoVSAImpl(
        num_heads=3,
        head_size=8,
        softmax_scale=8**-0.5,
        backend_kwargs={"block_size": (4, 4, 4), "topk": 2, "min_seq_len": 1, "fallback_on_error": False},
    )
    valid = 300
    query, key, value, gate = [torch.randn(2, 320 if packed else valid, 3, 8) for _ in range(4)]
    extra = {"vsa_dit_seq_shape": (3, 10, 10)}
    if learned_gate:
        extra["gate_compress"] = gate
    padding = (
        PackedPaddingMetadata(
            q_length=valid,
            kv_length=valid,
            cu_seqlens_q=torch.tensor([0, valid], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, valid], dtype=torch.int32),
        )
        if packed
        else None
    )
    metadata = AttentionMetadata(extra=extra, packed_padding=padding)
    # Exercise real tiling and the custom op on CPU, bypassing CUDA-only admission.
    monkeypatch.setattr(impl, "_fallback_reason", lambda *args: None)
    output = impl.forward_cuda(query, key, value, metadata)
    expected = query[:, :valid] + 2 * key[:, :valid] + 3 * value[:, :valid]
    if learned_gate:
        expected = expected + gate[:, :valid]
    torch.testing.assert_close(output[:, :valid], expected)
    assert output.shape == query.shape and len(calls) == 1
    assert torch.count_nonzero(output[:, valid:]) == 0


@pytest.mark.parametrize("tile, reason", [((4, 4, 4), "dtype"), ((2, 8, 8), "block_elements")])
def test_local_fastwan_tile64_admission(tile, reason):
    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        backend_kwargs={"block_size": tile, "topk": 1, "min_seq_len": 1},
    )
    query = torch.randn(1, 300, 2, 8)
    metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (3, 10, 10)})
    assert impl._fallback_reason(query, query, query, metadata).startswith(reason)


def test_sdpa_fallback_never_attends_the_structural_padding(monkeypatch):
    # The backend advertises supports_packed_mask_free, so a model may skip
    # building the padding mask. Every fallback out of the VSA path must honour
    # that contract itself; SDPA reads attn_mask and nothing else.
    impl = FastVideoVSAImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        backend_kwargs={"topk": 1, "min_seq_len": 4096},
    )
    valid = 40
    torch.manual_seed(0)
    query = torch.randn(1, 64, 2, 8)
    padding = PackedPaddingMetadata(
        q_length=valid,
        kv_length=valid,
        cu_seqlens_q=torch.tensor([0, valid], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, valid], dtype=torch.int32),
    )

    monkeypatch.setattr(impl.sdpa_fallback, "forward", impl.sdpa_fallback.forward_cuda)

    baseline = impl.forward_cuda(query, query, query, AttentionMetadata(packed_padding=padding, extra={}))
    perturbed_input = query.clone()
    perturbed_input[:, valid:] += 100.0
    perturbed = impl.forward_cuda(
        perturbed_input, perturbed_input, perturbed_input, AttentionMetadata(packed_padding=padding, extra={})
    )

    assert baseline.shape == query.shape
    torch.testing.assert_close(baseline[:, :valid], perturbed[:, :valid])
    assert torch.count_nonzero(baseline[:, valid:]) == 0


def _local_ulysses_impl(monkeypatch, degree):
    from vllm_omni.diffusion import config
    from vllm_omni.diffusion.attention.backends import fastvideo_vsa

    monkeypatch.setattr(config, "get_current_diffusion_config_or_none", lambda: None)
    parallel = types.SimpleNamespace(ulysses_degree=degree, sequence_parallel_size=degree)
    ctx = types.SimpleNamespace(
        omni_diffusion_config=types.SimpleNamespace(parallel_config=parallel),
        sp_active=degree > 1,
        sp_padding_size=0,
    )
    monkeypatch.setattr(fastvideo_vsa, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(fastvideo_vsa, "is_forward_context_available", lambda: True)
    impl = FastVideoVSAImpl(
        num_heads=40,
        head_size=128,
        softmax_scale=128**-0.5,
        backend_kwargs={"block_size": (4, 4, 4), "allow_ulysses": True, "topk": 2},
    )
    return impl, ctx


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_local_ulysses_accepts_full_sequence_local_heads(monkeypatch, degree):
    impl, _ = _local_ulysses_impl(monkeypatch, degree)
    query = torch.zeros(1, 512, 40 // degree, 128)
    metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (4, 8, 16), "gate_compress": query})
    impl._validate_ulysses_inputs(query, metadata)
    assert not impl.fallback_on_error
    with pytest.raises(ValueError, match="refuses dense fallback"):
        impl._fallback(query, query, query, metadata, "test")


def test_local_ulysses_keeps_default_sp_fallback(monkeypatch):
    from vllm_omni.diffusion.attention.backends import fastvideo_vsa

    monkeypatch.setattr(fastvideo_vsa, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(fastvideo_vsa, "get_forward_context", lambda: types.SimpleNamespace(sp_active=True))
    impl = FastVideoVSAImpl(
        num_heads=40, head_size=128, softmax_scale=128**-0.5, backend_kwargs={"block_size": (4, 4, 4), "topk": 2}
    )
    # Admission only reads tensor properties; no CUDA allocation is needed.
    query = types.SimpleNamespace(ndim=4, shape=(1, 512, 20, 128), dtype=torch.bfloat16, device=torch.device("cuda"))
    metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (4, 8, 16)})
    assert impl._fallback_reason(query, query, query, metadata) == "sequence parallel context is active"


@pytest.mark.parametrize(
    "field, value",
    [
        ("ring_degree", 2),
        ("allgather_degree", 2),
        ("tensor_parallel_size", 2),
        ("cfg_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("data_parallel_size", 2),
        ("use_hsdp", True),
        ("ulysses_mode", "advanced_uaa"),
        ("sequence_parallel_size", 4),
        ("ulysses_degree", 3),
    ],
)
def test_local_ulysses_rejects_unsupported_topology(field, value):
    parallel = types.SimpleNamespace(ulysses_degree=2, sequence_parallel_size=2)
    setattr(parallel, field, value)
    with pytest.raises(ValueError, match="Local FastWan"):
        validate_local_ulysses_config(parallel)


def test_local_ulysses_rejects_ring_before_attention_dispatch(monkeypatch):
    from vllm_omni.diffusion import config

    parallel = types.SimpleNamespace(ulysses_degree=2, sequence_parallel_size=4, ring_degree=2)
    monkeypatch.setattr(
        config, "get_current_diffusion_config_or_none", lambda: types.SimpleNamespace(parallel_config=parallel)
    )
    with pytest.raises(ValueError, match="ring_degree"):
        FastVideoVSAImpl(
            num_heads=40,
            head_size=128,
            softmax_scale=128**-0.5,
            backend_kwargs={"block_size": (4, 4, 4), "allow_ulysses": True},
        )


@pytest.mark.parametrize("invalid", ["padding", "local_sequence", "global_heads", "local_gate", "inactive"])
def test_local_ulysses_rejects_incorrect_communication_contract(monkeypatch, invalid):
    impl, ctx = _local_ulysses_impl(monkeypatch, 2)
    query = torch.zeros(1, 512, 20, 128)
    gate = query
    if invalid == "padding":
        ctx.sp_padding_size = 1
    elif invalid == "local_sequence":
        query = query[:, :256]
    elif invalid == "global_heads":
        query = torch.zeros(1, 512, 40, 128)
    elif invalid == "local_gate":
        gate = gate[:, :256]
    elif invalid == "inactive":
        ctx.sp_active = False
    metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (4, 8, 16), "gate_compress": gate})
    with pytest.raises(ValueError, match="Local FastWan"):
        impl._validate_ulysses_inputs(query, metadata)


def test_local_ulysses_kernel_errors_are_not_retried_with_dense(monkeypatch):
    from vllm_omni.diffusion.attention.backends import fastvideo_vsa

    impl, _ = _local_ulysses_impl(monkeypatch, 2)
    query = torch.zeros(1, 512, 20, 128)
    metadata = AttentionMetadata(extra={"vsa_dit_seq_shape": (4, 8, 16), "gate_compress": query})
    monkeypatch.setattr(impl, "_fallback_reason", lambda *args: None)

    def fail(*args):
        raise RuntimeError("kernel sentinel")

    monkeypatch.setattr(fastvideo_vsa, "_fastvideo_vsa_bshd_op", fail)
    with pytest.raises(RuntimeError, match="kernel sentinel"):
        impl.forward_cuda(query, query, query, metadata)
