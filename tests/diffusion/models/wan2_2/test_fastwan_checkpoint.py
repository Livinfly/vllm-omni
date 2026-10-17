# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Our FastWan checkpoint adaptation must load biases and resolve mixed aliases."""

import pytest
import torch
from vllm.model_executor.layers.linear import QKVParallelLinear

from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel


class TinyWan(torch.nn.Module):
    load_weights = WanTransformer3DModel.load_weights

    def __init__(self):
        super().__init__()
        block = torch.nn.Module()
        block.attn1 = torch.nn.Module()
        block.attn1.to_qkv = QKVParallelLinear(4, 2, 2, bias=True, disable_tp=True)
        block.attn1.to_out = torch.nn.Linear(4, 4)
        block.attn2 = torch.nn.Module()
        block.attn2.to_out = torch.nn.Linear(4, 4)
        block.ffn = torch.nn.Module()
        block.ffn.net_0 = torch.nn.Module()
        block.ffn.net_0.proj = torch.nn.Linear(4, 8)
        block.ffn.net_2 = torch.nn.Linear(8, 4)
        self.blocks = torch.nn.ModuleList([block])


@pytest.mark.parametrize("reverse", [False, True])
def test_fastwan_mixed_checkpoint_loads_weights_and_biases(monkeypatch, reverse):
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    prefix = "vllm_omni.diffusion.models.wan2_2.wan2_2_transformer"
    monkeypatch.setattr(f"{prefix}.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(f"{prefix}.get_tensor_model_parallel_world_size", lambda: 1)
    model = TinyWan()
    weights, expected = [], {}
    for name, param in model.named_parameters():
        if ".to_qkv." in name:
            for i, part in enumerate(param.chunk(3, dim=0)):
                source = name.replace("attn1.to_qkv", f"to_{'qkv'[i]}")
                alias = source.replace("blocks.0.to_", "blocks.0.attn1.to_")
                value = torch.full_like(part, i + 1)
                weights.extend([(source, value), (alias, torch.zeros_like(value))])
            expected[name] = torch.cat([torch.full_like(part, i + 1) for i, part in enumerate(param.chunk(3))])
        else:
            source = name.replace("attn1.to_out", "to_out")
            source = source.replace("ffn.net_0.proj", "ffn.fc_in").replace("ffn.net_2", "ffn.fc_out")
            alias = name.replace("attn1.to_out", "attn1.to_out.0").replace("attn2.to_out", "attn2.to_out.0")
            alias = alias.replace("ffn.net_0", "ffn.net.0").replace("ffn.net_2", "ffn.net.2")
            expected[name] = torch.full_like(param, 4)
            weights.extend([(source, expected[name]), (alias, torch.zeros_like(param))])
    if reverse:
        weights.reverse()
    loaded = model.load_weights(weights)
    assert set(dict(model.named_parameters())) <= loaded
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, expected[name], rtol=0, atol=0)
