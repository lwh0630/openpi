# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch (批次大小)
  T: query length (查询序列长度)
  S: k/v length (键/值序列长度)
  N: num query heads (查询头数量)
  K: num k/v heads (键/值头数量)
  G: num query heads per k/v head (每个键/值头对应的查询头数量)
  H: head dim (头维度)
  D: d_model ("features") (模型维度，即特征维度)
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    """模型配置类。"""
    width: int  # 模型宽度，即特征维度 d_model
    depth: int  # 模型深度，即Transformer层数
    mlp_dim: int  # MLP（多层感知机）的隐藏层维度
    num_heads: int  # 注意力头的数量
    num_kv_heads: int  # 键值（Key/Value）头的数量，用于分组查询注意力（GQA）
    head_dim: int  # 每个注意力头的维度
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)  # LoRA配置字典


Variant = Literal["dummy", "gemma_300m", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """根据指定的Gemma变体返回配置。"""
    if variant == "dummy":
        # 虚拟配置，用于测试或小规模实验
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # Gemma 300M 参数配置
        # 311M 参数量
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        # Gemma 2B 参数配置
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        # Gemma 2B 带有LoRA的参数配置
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # Gemma 300M 带有LoRA的参数配置
        # 311M 参数量
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    """RMS归一化层，Gemma模型中使用的归一化方式。"""
    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # 原始数据类型，可能是半精度（如bfloat16）
        # 归一化参数，初始化为零
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
        # 计算方差，使用float32以确保精度
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        # 计算归一化输入，使用float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        # 通过学习到的参数进行缩放，使用float32（与Flax实现匹配）
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs.astype(dtype)  # 返回原始数据类型


@at.typecheck
class Embedder(nn.Module):
    """嵌入层模块，用于将token ID转换为词嵌入。"""

    vocab_size: int  # 词汇表大小
    embed_dim: int  # 嵌入维度

    def setup(self):
        """设置嵌入表。"""
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),  # 使用正态分布初始化
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        """将token ID编码为嵌入向量。"""
        x = self.input_embedding_table[(x,)]  # 查找嵌入
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)  # 缩放嵌入（Gemma的特殊处理）
        return x

    def decode(self, x):
        """将嵌入向量解码为logits（用于预测下一个token）。"""
        return jnp.dot(x, self.input_embedding_table.T)  # 与嵌入表的转置进行点积


@at.typecheck
class Attention(nn.Module):
    """注意力模块。"""

    configs: Sequence[Config]  # 专家配置列表

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # 所有专家必须共享相同的头维度、头数量和kv头数量，以便自注意力机制正常工作
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # 原始数据类型，可能是半精度

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:  # 如果当前专家没有输入，则跳过
                continue
            if config.num_kv_heads == config.num_heads:
                # 标准多头注意力（MHA）情况
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                # 分组查询注意力（GQA）情况
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        # 将所有专家的Q, K, V拼接起来
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        # 应用旋转位置嵌入（RoPE）到查询和键
        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5  # 缩放查询

        k = _apply_rope(k, positions=positions)

        # 确保数据类型在半精度（如果输入是半精度）
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            # 如果存在KV缓存，则拼接缓存的K和V
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        # 重新排列查询维度以匹配GQA的QKV乘法
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        # 计算注意力分数（logits），使用float32以避免精度问题
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        # 检查注意力掩码的形状
        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # 应用注意力掩码，将不需要注意力的位置设置为一个非常小的负数
        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # 参考gemma/modules.py中的值
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        # 计算注意力权重（probs），并转换回原始数据类型
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        # 计算加权和，得到编码后的注意力输出
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        # 重新排列维度
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                # 输出投影层
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)


@at.typecheck
class FeedForward(nn.Module):
    """前馈网络模块（MLP）。"""

    features: int  # 输入特征维度
    hidden_dim: int  # 隐藏层维度

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # 原始数据类型
        # 门控权重，用于GELU激活
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])  # 第一个线性变换
        gate_value = nn.gelu(ff_gate)  # 应用GELU激活

        ff1 = jnp.dot(x, w_gating[1])  # 第二个线性变换
        activations = gate_value * ff1  # 门控机制

        # 输出线性变换
        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


@at.typecheck
class Block(nn.Module):
    """Transformer块（一层）。"""

    configs: Sequence[Config]  # 专家配置列表

    dropout: float = 0.0  # Dropout比率
    dropout_bdims: tuple[int, ...] = ()  # 独立dropout的维度

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, decode, deterministic=True):  # noqa: FBT002
        # 激活值分片约束
        xs = sharding.activation_sharding_constraint(xs)
        # Dropout层
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        # 注意力模块
        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        for i, x in enumerate(xs):
            if x is not None:
                # 应用RMS归一化
                x = RMSNorm(name=_name("pre_attention_norm", i))(x)  # noqa: PLW2901
            pre_attn.append(x)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        # 执行注意力计算
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        # 应用Dropout
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        # 残差连接：输入 + 注意力输出
        xs = jax.tree.map(lambda x, y: x + y, xs, post_attn)
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                # 应用RMS归一化
                x = RMSNorm(name=_name("pre_ffw_norm", i))(x)  # noqa: PLW2901
                # 应用前馈网络（LoRA集成）
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)

        out = sharding.activation_sharding_constraint(out)

        # 应用Dropout
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        # 残差连接：输入 + 前馈网络输出
        xs = jax.tree.map(lambda x, y: x + y, xs, out)
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@at.typecheck
class Module(nn.Module):
    """Transformer模型，支持为不同token混合使用不同权重（专家混合）。"""

    configs: Sequence[Config]  # 专家配置列表，每个专家一个配置
    embed_dtype: str  # 嵌入层的数据类型

    dropout: float = 0.0  # Dropout比率
    dropout_bdims: tuple[int, ...] = ()  # 独立dropout的维度

    def setup(self):
        # 所有专家必须具有相同的深度
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # 仅使用第一个专家的嵌入维度
            name="embedder",
        )
        # 使用nn.remat进行梯度检查点，以节省内存
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),  # 0=self, 5=deterministic (确定性参数为静态参数)
            policy=jax.checkpoint_policies.nothing_saveable,  # 检查点策略
        )
        # 使用nn.scan构建Transformer层，实现层之间的共享变量轴
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},  # 参数在层之间共享
            split_rngs={"params": True, "dropout": True},  # 分割随机数生成器
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast),  # 输入轴映射：kv_cache, positions, mask, decode
            length=self.configs[0].depth,  # 层的数量
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        # 最终的RMS归一化层，每个专家一个
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        """将token ID嵌入为向量。"""
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        # token嵌入列表，每个专家一个，如果专家不运行则为None
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],  # 位置信息
        mask: at.Bool[at.Array, "b t s"],  # 注意力掩码
        *,
        kv_cache: KVCache | None = None,  # KV缓存
        deterministic: bool = True,  # 是否处于确定性模式（例如，推理时为True）
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        # 将嵌入数据转换为指定的数据类型
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        # 扩展注意力掩码维度
        mask = jnp.asarray(mask)[:, None, :, :]

        # 通过Transformer层
        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, deterministic)

        # 确保输出数据类型正确
        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        # 应用最终归一化层
        return [f(e) if e is not None else e for f, e in zip(self.final_norms, embedded, strict=True)], kv_cache

    def init(self):
        """方便初始化所有参数的方法，由于linen的特性而必需。"""
        # 通过调用embed和__call方法来触发参数初始化
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """将旋转位置嵌入（RoPE）应用于输入x。
    RoPE应用于x [B, L, H, D]，其中positions [B, L]。
    """
    # 计算频率指数
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    # 计算时间尺度
    timescale = max_wavelength**freq_exponents
    # 计算弧度
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    # 计算sin和cos分量
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    # 将x分割为两部分，并应用旋转
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # 原始的bigvision实现允许RoPE上转换为float32。然后在推理模式下（但不是训练模式下）立即再次下转换为缓存数据类型。
    # 我认为这都不是故意的。根据原始的DeepMind实现以及广泛使用的transformers实现，这里总是下转换回bfloat16是没问题的。
    return res.astype(x.dtype)


def _name(name, i):
    """为层生成名称。"""
    # 我们这样命名层是因为我们希望第一个专家的权重没有后缀（例如，“attn”），这样它们就可以从现有的PaliGemma检查点无缝加载。
    # 随后的专家将有一个后缀（例如，“attn_1”），并且它们的权重将从头开始初始化。
    # 实际上，我们只使用两个专家——PaliGemma和action专家。
    if i == 0:
        return name
    return f"{name}_{i}"
