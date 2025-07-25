import math
import re

import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp

import openpi.shared.array_typing as at


@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA.
    LoRA 配置。
    """

    # LoRA rank.
    # LoRA 的秩。
    rank: int
    # LoRA scaling factor.
    # LoRA 缩放因子。
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    # LoRA 参数的初始化函数。
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    # 启用秩稳定 LoRA：https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    # 应用 LoRA 的权重轴。通常应该是最后两个轴。
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    # LoRA 在 einsum 方程中使用的轴标签。原始方程中不能包含此标签。
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        # 计算缩放值。如果启用 rslora，则为 alpha / sqrt(rank)，否则为 alpha / rank。
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum.
    支持 LoRA 的 Einsum。可以作为 Gemma Einsum 的直接替代品。
    """

    # Shape of the weight.
    # 权重的形状。
    shape: tuple[int, ...]
    # Initialization function for the weight.
    # 权重的初始化函数。
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    # 如果不为 None，则对权重应用 LoRA。
    lora_config: LoRAConfig | None = None

    def setup(self):
        # 初始化主权重 w。
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # 设置 LoRA 参数。
            shape_a, shape_b = list(self.shape), list(self.shape)
            # 根据 LoRA 配置的轴和秩调整 w_a 和 w_b 的形状。
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            # 初始化 LoRA 的两个低秩矩阵 w_a 和 w_b。
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        # 原始数据类型，可能是半精度。
        dtype = x.dtype
        # 执行原始的 einsum 操作。
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            # 生成 LoRA 对应的 einsum 方程。
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            # 执行 LoRA 的第一个 einsum 操作。
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            # 执行 LoRA 的第二个 einsum 操作。
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            # 将 LoRA 结果乘以缩放因子并加到原始结果上。
            result = result + lora * config.scaling_value

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        # 检查 LoRA 标签是否已存在于方程中。
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        # 解析 einsum 方程，分为左侧、右侧和输出。
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        # 获取 LoRA 应用的轴的标签。
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        # 获取 LoRA 的新标签。
        label = self.lora_config.label

        # 构建 LoRA 第一个矩阵 (w_a) 的 einsum 方程。
        # 将 rhs 中对应 b_label 的轴替换为 LoRA 标签。
        a_rhs = rhs.replace(b_label, label)
        # 将 out 中对应 b_label 的轴替换为 LoRA 标签。
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        # 构建 LoRA 第二个矩阵 (w_b) 的 einsum 方程。
        # 将 rhs 中对应 a_label 的轴替换为 LoRA 标签。
        b_rhs = rhs.replace(a_label, label)
        # eqn_b 的输入是 eqn_a 的输出和 w_b。
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """Feed forward module.
    前馈网络模块。
    """

    features: int
    hidden_dim: int
    # If not None, apply LoRA to the weight.
    # 如果不为 None，则对权重应用 LoRA。
    lora_config: LoRAConfig | None = None

    def setup(self):
        # 初始化门控权重。
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        # 初始化线性层权重。
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # 设置 LoRA 参数。
            # TODO: 后续会提供简化的 init_fn API。
            # 初始化门控权重的 LoRA 参数 (w_a, w_b)。
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param(
                    "gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)
                ),
            )
            # 初始化线性层权重的 LoRA 参数 (w_a, w_b)。
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    @nn.compact
    def __call__(self, x):
        # 原始数据类型，可能是半精度。
        dtype = x.dtype
        # 计算第一个门控分支（对应 w_gating[0]）。
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            # 如果存在 LoRA，则使用门控 LoRA 参数的第一个部分。
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )
        # 应用 GELU 激活函数。
        gate_value = nn.gelu(ff_gate)

        # 计算第二个门控分支（对应 w_gating[1]）。
        ff1 = self._dot(
            x,
            self.w_gating[1],
            # 如果存在 LoRA，则使用门控 LoRA 参数的第二个部分。
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )
        # 将两个分支的结果相乘。
        activations = gate_value * ff1

        # 计算最终输出。
        outputs = self._dot(activations, self.w_linear, self.w_linear_lora)
        # 确保输出数据类型与输入相同。
        assert outputs.dtype == dtype
        return outputs

    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None) -> at.Array:
        # 执行基础的矩阵乘法。
        base = jnp.dot(x, w.astype(x.dtype))
        # 如果没有 LoRA 权重，直接返回基础结果。
        if lora_weights is None:
            return base
        # 如果有 LoRA 权重，计算 LoRA 增量并加到基础结果上。
        return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))
