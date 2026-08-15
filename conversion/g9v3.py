from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .llama import LlamaModel


@ModelBase.register("G9v3ForCausalLM")
class G9v3Model(LlamaModel):
    model_arch = gguf.MODEL_ARCH.G9V3
    undo_permute = False
    _experts: list[dict[str, Tensor]] | None = None

    def set_vocab(self) -> None:
        self._set_vocab_gpt2()

        # The checkpoint declares both </s> and <|im_end|> as EOS. GGUF has one
        # EOS id, so register the turn-ending token as EOT; llama.cpp treats
        # both entries as end-of-generation tokens.
        eos_ids = self.hparams.get("eos_token_id")
        if isinstance(eos_ids, list):
            eos_id = eos_ids[0]
            extra_ids = [token_id for token_id in eos_ids if token_id != eos_id]
            if extra_ids:
                self.gguf_writer.add_eot_token_id(extra_ids[0])
                logger.info(f"gguf: registered eot_token_id={extra_ids[0]} from eos list {eos_ids}")

    def set_gguf_parameters(self) -> None:
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_weights_norm(bool(hparams.get("norm_topk_prob", True)))
        self.gguf_writer.add_expert_weights_scale(float(hparams["routed_scaling_factor"]))
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # q_proj is interleaved by head as [query, attention_gate]. Split it
        # into llama.cpp's Q and gate tensors. G9v3 uses NeoX-style RoPE, so Q
        # and K keep their native HF layout (no Llama weight permutation).
        if name.endswith("self_attn.q_proj.weight"):
            assert bid is not None
            n_head = self.hparams["num_attention_heads"]
            head_dim = self.hparams["head_dim"]
            expected = 2 * n_head * head_dim
            if data_torch.shape[0] != expected:
                raise ValueError(
                    f"G9v3 layer {bid}: q_proj output width {data_torch.shape[0]} "
                    f"does not match 2 * num_attention_heads * head_dim ({expected})"
                )

            q_gate = data_torch.reshape(n_head, 2 * head_dim, *data_torch.shape[1:])
            query = q_gate[:, :head_dim].reshape(n_head * head_dim, *data_torch.shape[1:])
            gate = q_gate[:, head_dim:].reshape(n_head * head_dim, *data_torch.shape[1:])

            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), query
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_GATE, bid), gate
            return

        # The HF checkpoint stores 320 independent experts. GGUF stores each
        # projection as one [expert, out, in] tensor for efficient MoE kernels.
        if re.search(r"mlp\.experts\.\d+\.", name):
            assert bid is not None
            n_experts = self.hparams["n_routed_experts"]
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch

            needed = [
                f"model.layers.{bid}.mlp.experts.{xid}.{proj}.weight"
                for xid in range(n_experts)
                for proj in ("gate_proj", "up_proj", "down_proj")
            ]
            if not all(tensor_name in self._experts[bid] for tensor_name in needed):
                return

            for proj in ("gate_proj", "up_proj", "down_proj"):
                tensors = [
                    self._experts[bid][f"model.layers.{bid}.mlp.experts.{xid}.{proj}.weight"]
                    for xid in range(n_experts)
                ]
                merged = torch.stack(tensors, dim=0)
                merged_name = f"model.layers.{bid}.mlp.experts.{proj}.weight"
                yield from ModelBase.modify_tensors(self, merged, merged_name, bid)
            self._experts[bid].clear()
            return

        yield from LlamaModel.modify_tensors(self, data_torch, name, bid)
