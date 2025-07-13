import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """改编自 big_vision。

    token 可以关注到有效的输入 token，这些 token 的累计 `mask_ar` 小于或等于它们自己的累计 `mask_ar`。
    这样，`mask_ar` bool[?B, N] 可以用于设置几种类型的注意力，例如：

      [[1 1 1 1 1 1]]: 纯粹的因果注意力。

      [[0 0 0 1 1 1]]: prefix-lm 注意力。前 3 个 token 可以在它们之间互相关注，后 3 个 token 具有因果注意力。
          第一个条目也可以是 1，而不会改变行为。

      [[1 0 1 0 1 0 0 1 0 0]]: 4 个块之间的因果注意力。一个块的 token 可以关注所有先前的块以及同一块中的所有 token。

    Args:
      input_mask: bool[B, N] 如果是输入的一部分，则为 True，否则为 False（填充）。
      mask_ar: bool[?B, N] 如果先前的 token 不能依赖它，则为 True，否则为 False（表示与先前 token 共享相同的注意力掩码）。
    """
    # 确保 mask_ar 具有与 input_mask 相同的批次和序列维度。
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    # 沿序列维度计算累积和。这会为每个 token 创建一个值，表示其“注意力组”。
    cumsum = jnp.cumsum(mask_ar, axis=1)
    # attn_mask 决定因果注意力：位置 j 的 token 只能关注位置 i 的 token，如果 cumsum[i] <= cumsum[j]。
    # 这实现了上述各种注意力模式。
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    # valid_mask 确保注意力只发生在实际的输入 token 之间（而不是填充）。
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    # 最终的注意力掩码是因果注意力和有效输入 token 的逻辑与。
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """为标量位置计算正弦-余弦位置嵌入向量。

    Args:
        pos: 用于计算嵌入的标量位置，形状为 [batch_size]。
        embedding_dim: 嵌入向量的所需维度。必须是偶数。
        min_period: 正弦/余弦波的最小周期。
        max_period: 正弦/余弦波的最大周期。

    Returns:
        正弦-余弦位置嵌入，形状为 [batch_size, embedding_dim]。
    """
    # 为嵌入维度的一半创建从 0 到 1 的分数数组。
    # 这将用于为正弦/余弦波创建不同的周期。
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    # 计算每个维度的周期，在 min_period 和 max_period 之间呈指数分布。
    period = min_period * (max_period / min_period) ** fraction
    # 计算正弦和余弦函数的输入。这涉及到位置和 2*pi/period 的外积。
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,  # 使用最高精度以确保数值稳定性
    )
    # 连接正弦和余弦分量以形成最终嵌入。
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"  # 模型参数的数据类型
    paligemma_variant: _gemma.Variant = "gemma_2b"  # PaliGemma 模型的变体
    action_expert_variant: _gemma.Variant = "gemma_300m"  # 动作专家模型的变体

    # 设置模型特定默认值。
    action_dim: int = 32  # 动作空间的维度
    action_horizon: int = 50  # 预测的未来动作数量
    max_token_len: int = 48  # token 化提示的最大长度

    @property
    @override
    def model_type(self) -> _model.ModelType:
        """返回模型类型"""
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        """创建 Pi0 模型的一个实例。"""
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """定义模型的输入规范。

        Args:
            batch_size: 输入规范的批次大小。

        Returns:
            包含观察和动作规范的元组。
        """
        # 定义图像输入的规范（来自不同相机的 RGB 图像）
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        # 定义图像掩码的规范（指示图像是否存在）
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            # 定义完整的观察规范
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),  # 当前机器人状态
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),  # token 化提示
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),  # 提示掩码
            )
        # 定义动作规范
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """根据模型配置返回冻结过滤器。

        此过滤器用于指定在训练期间应冻结哪些参数，尤其是在使用 LoRA（低秩适配）时。
        """
        filters = []
        has_lora = False
        # 用于识别 PaliGemma LLM 参数的正则表达式
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        # 用于识别动作专家 LLM 参数的正则表达式
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            # 如果 PaliGemma 使用 LoRA，则将其参数添加到冻结过滤器中
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # 如果只有 PaliGemma 使用 LoRA，则将动作专家参数从冻结中排除
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            # 如果动作专家使用 LoRA，则将其参数添加到冻结过滤器中
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # 如果使用了任何 LoRA，则将所有 LoRA 参数从冻结中排除（因为它们通常是需要训练的）
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            # 如果没有添加特定过滤器，则返回 Nothing，表示此过滤器不冻结任何参数。
            return nnx.Nothing
        # 使用“All”运算符组合所有过滤器，这意味着所有条件都必须满足才能冻结。
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    """用于预测机器人动作的 Pi0 模型。"""

    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        """初始化 Pi0 模型。

        Args:
            config: Pi0 模型的配置对象。
            rngs: JAX 随机数生成器状态。
        """
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        # 获取 PaliGemma 和动作专家的配置。
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: 使用 NNX 重写 gemma。目前，使用 bridge。
        # 使用 nnx_bridge.ToNNX 将非 NNX 模块 (_gemma.Module) 桥接到 NNX。
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        # LLM 的延迟初始化。
        llm.lazy_init(rngs=rngs, method="init")
        # 初始化图像编码器 (SigLIP)。
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,  # 图像编码器的输出维度
                variant="So400m/14",  # SigLIP 模型变体
                pool_type="none",  # 不进行池化
                scan=True,  # 使用 scan 进行高效处理
                dtype_mm=config.dtype,  # 矩阵乘法的数据类型
            )
        )
        # 使用伪造的观察值对图像编码器进行延迟初始化。
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        # 将 LLM 和图像编码器存储在一个 Dict 中。
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # 状态和动作输入的线性投影。
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        # 用于处理动作和时间信息的 MLP。
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        # 输出动作的线性投影。
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """嵌入输入的“前缀”部分（图像和语言提示）。

        Args:
            obs: 包含图像和 token 化提示的观察值。

        Returns:
            包含以下内容的元组：
                - `tokens`: 嵌入的前缀 token。
                - `input_mask`: 指示有效输入 token 的掩码。
                - `ar_mask`: 用于注意力的自回归掩码。
        """
        input_mask = []
        ar_mask = []
        tokens = []
        # 嵌入图像
        for name in obs.images:
            # 使用 SigLIP 图像编码器对每个图像进行编码。
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            # 根据 image_masks 为图像 token 创建输入掩码。
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # 图像 token 可以互相关注，因此将 ar_mask 设置为 False。
            ar_mask += [False] * image_tokens.shape[1]

        # 添加语言（即 token 化输入）
        if obs.tokenized_prompt is not None:
            # 使用 LLM 嵌入 token 化提示。
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # 图像和语言输入之间是完全注意力，因此将 ar_mask 设置为 False。
            ar_mask += [False] * tokenized_inputs.shape[1]
        # 连接所有前缀 token 及其掩码。
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """嵌入输入的“后缀”部分（状态、带噪声的动作和时间步）。

        Args:
            obs: 包含当前状态的观察值。
            noisy_actions: 扩散中使用的带噪声的动作。
            timestep: 当前的扩散时间步。

        Returns:
            包含以下内容的元组：
                - `tokens`: 嵌入的后缀 token。
                - `input_mask`: 指示有效输入 token 的掩码。
                - `ar_mask`: 用于注意力的自回归掩码。
        """
        input_mask = []
        ar_mask = []
        tokens = []
        # 添加一个单独的状态 token
        # 将当前状态投影到嵌入维度。
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        # 状态 token 的掩码（始终存在）。
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # 图像/语言输入不关注状态或动作，因此将 ar_mask 设置为 True。
        ar_mask += [True]

        # 使用正弦-余弦位置编码嵌入时间步，敏感度范围为 [0, 1]
        # 使用正弦-余弦位置编码嵌入时间步。
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # 使用 MLP 混合时间步 + 动作信息
        # 将带噪声的动作投影到嵌入维度。
        action_tokens = self.action_in_proj(noisy_actions)
        # 重复时间嵌入以匹配动作时间范围。
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        # 连接动作和时间嵌入。
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        # 通过带有 Swish 激活函数的 MLP。
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        # 动作-时间 token 的掩码（始终存在）。
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # 图像/语言/状态输入不关注动作 token。
        # 第一个动作 token 是自回归的 (True)，随后的则不是 (False)。
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        # 连接所有后缀 token 及其掩码。
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """计算 Pi0 模型的损失。

        Args:
            rng: JAX 随机数生成器状态。
            observation: 当前的观察值。
            actions: 真实动作。
            train: 模型是否处于训练模式。

        Returns:
            计算出的损失。
        """
        # 将 RNG 分割用于预处理、噪声和时间。
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # 预处理观察值。
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        # 生成与动作相同形状的随机噪声。
        noise = jax.random.normal(noise_rng, actions.shape)
        # 从 Beta 分布中采样一个时间步 `t`。
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        # 计算带噪声的动作 (x_t) 和扩散的目标噪声 (u_t)。
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # 一次性对前缀 + 后缀进行一次大的前向传播
        # 嵌入前缀和后缀 token。
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        # 连接完整序列的掩码和自回归掩码。
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        # 创建完整的注意力掩码。
        attn_mask = make_attn_mask(input_mask, ar_mask)
        # 计算 token 的位置。
        positions = jnp.cumsum(input_mask, axis=1) - 1
        # 使用前缀和后缀对 LLM 进行前向传播。
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        # 将 LLM 输出的后缀 token 投影到动作维度，以获得预测噪声 (v_t)。
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # 计算预测噪声和目标噪声之间的均方误差损失。
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        """使用迭代扩散过程采样动作。

        Args:
            rng: JAX 随机数生成器状态。
            observation: 当前的观察值。
            num_steps: 要执行的扩散步数。

        Returns:
            采样的动作。
        """
        # 预处理观察值（推理期间不进行随机操作）。
        observation = _model.preprocess_observation(None, observation, train=False)
        # 初始化时间步递减值。
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        # 使用随机正态噪声初始化带噪声的动作。
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # 首先通过前缀的前向传播填充 KV 缓存
        # 嵌入前缀 token。
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        # 为前缀创建注意力掩码。
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        # 计算前缀 token 的位置。
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # 仅使用前缀 token 执行前向传播以填充 KV 缓存。
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            """单个扩散步。"""
            x_t, time = carry
            # 嵌入后缀 token（带噪声的动作和当前时间步）。
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` 的形状为 (b, suffix_len, suffix_len)，表示后缀 token 如何互相关注。
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` 的形状为 (b, suffix_len, prefix_len)，表示后缀 token 如何关注前缀 token。
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` 的形状为 (b, suffix_len, prefix_len + suffix_len)，表示后缀 token（生成查询）
            # 如何关注完整的前缀 + 后缀序列（生成键和值）。
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` 的形状为 (b, suffix_len)，表示后缀 token 的位置。
            # 这些位置是相对于完整序列（前缀 + 后缀）的。
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # 使用 KV 缓存执行前向传播以预测噪声。
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None  # 这里只期望后缀输出
            # 将后缀输出投影到动作维度，以获得预测噪声 (v_t)。
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # 更新 x_t 和时间以进行下一步。
            return x_t + dt * v_t, time + dt

        def cond(carry):
            """while 循环的条件（如果时间尚未接近 0 则继续）。"""
            x_t, time = carry
            # 对浮点误差具有鲁棒性
            return time >= -dt / 2

        # 使用 JAX while 循环运行扩散过程。
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
