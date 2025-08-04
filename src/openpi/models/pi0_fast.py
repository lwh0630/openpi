import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1 # PaliGemma模型的EOS（End-of-Sequence）标记

def make_attn_mask(input_mask, mask_ar):
    """
    根据输入掩码和自回归掩码创建注意力掩码。
    此函数改编自big_vision项目。

    令牌可以关注有效的输入令牌，这些令牌的累积mask_ar小于或等于它们自己的。
    这样，`mask_ar` (bool[?B, N]) 可以用于设置多种类型的注意力，例如：

      [[1 1 1 1 1 1]]: 纯因果注意力。

      [[0 0 0 1 1 1]]: prefix-lm 注意力。前3个令牌可以相互关注，
          后3个令牌具有因果注意力。第一个条目也可以是1，不改变行为。

      [[1 0 1 0 1 0 0 1 0 0]]: 4个块之间的因果注意力。一个块的令牌
          可以关注所有先前的块以及同一块中的所有令牌。

    Args:
      input_mask: bool[B, N]，如果属于输入部分则为True，如果为填充则为False。
      mask_ar: bool[?B, N]，如果先前的令牌不能依赖它则为True，
        如果它与先前的令牌共享相同的注意力掩码则为False。
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape) # 将mask_ar广播到input_mask的形状
    cumsum = jnp.cumsum(mask_ar, axis=1) # 计算mask_ar的累积和
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None] # 创建注意力掩码，基于累积和的比较
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None] # 创建有效输入掩码
    return jnp.logical_and(attn_mask, valid_mask) # 结合注意力掩码和有效输入掩码

@jax.vmap # 对批次中的每个示例进行映射
def left_to_right_align(x, input_mask, attn_mask):
    """
    将输入从左对齐转换为右对齐。
    由于使用了vmap，此操作在单个示例级别进行（而不是批次级别）。
    """
    assert x.ndim == 2 # 确保x是二维的
    assert input_mask.ndim == 1 # 确保input_mask是一维的
    assert attn_mask.ndim == 2 # 确保attn_mask是二维的
    assert x.shape[0] == input_mask.shape[0] # 确保x和input_mask的序列长度匹配
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape # 确保attn_mask是方阵
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1 # 计算实际序列长度
    x = jnp.roll(x, -seqlen, axis=0) # 将x向左滚动，实现右对齐
    input_mask = jnp.roll(input_mask, -seqlen, axis=0) # 将input_mask向左滚动
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1)) # 将attn_mask向左滚动
    return x, input_mask, attn_mask # 返回右对齐后的数据和掩码

def put_along_last_axis(arr, indices, values):
    """
    类似于np.put_along_axis(..., axis=-1)，因为JAX缺少此功能。
    在数组的最后一个轴上放置值。
    """
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim) # 确保维度匹配
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype) # 将索引转换为one-hot编码
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot) # 创建放置掩码
    put_values = jnp.einsum("...i,...in->...n", values, onehot) # 创建要放置的值
    return jnp.where(put_mask, put_values, arr) # 根据掩码放置值

@dataclasses.dataclass(frozen=True)
class Pi0FASTConfig(_model.BaseModelConfig):
    """
    Pi0FAST模型的配置类。
    继承自_model.BaseModelConfig。
    """
    dtype: str = "bfloat16" # 模型使用的数据类型
    paligemma_variant: _gemma.Variant = "gemma_2b" # PaliGemma模型的变体

    # 设置模型特定的默认值
    action_dim: int = 32 # 动作维度
    action_horizon: int = 32 # 动作预测范围
    max_token_len: int = 250 # 最大令牌长度

    @property
    @override
    def model_type(self) -> _model.ModelType:
        """返回模型的类型。"""
        return _model.ModelType.PI0_FAST

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FAST":
        """使用给定的随机数生成器创建Pi0FAST模型实例。"""
        return Pi0FAST(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """
        定义模型输入（观察和动作）的形状和数据类型规范。
        """
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32) # 图像输入规范
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_) # 图像掩码规范

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec, # 基础相机0的RGB图像
                    "base_1_rgb": image_spec, # 基础相机1的RGB图像
                    "wrist_0_rgb": image_spec, # 腕部相机0的RGB图像
                },
                image_masks={
                    "base_0_rgb": image_mask_spec, # 基础相机0的图像掩码
                    "base_1_rgb": image_mask_spec, # 基础相机1的图像掩码
                    "wrist_0_rgb": image_mask_spec, # 腕部相机0的图像掩码
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32), # 状态输入
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32), # 令牌化提示
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool), # 令牌化提示掩码
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32), # 令牌自回归掩码
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_), # 令牌损失掩码
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32) # 动作规范

        return observation_spec, action_spec # 返回观察和动作规范

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """
        根据模型配置返回冻结过滤器。
        用于指定在训练期间哪些参数应该被冻结。
        """
        if "lora" in self.paligemma_variant:
            # 如果变体包含"lora"，则冻结除lora层之外的LLM参数
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing # 否则不冻结任何参数

class Pi0FAST(_model.BaseModel):
    """
    Pi0FAST模型类。
    继承自_model.BaseModel。
    """
    def __init__(self, config: Pi0FASTConfig, rngs: nnx.Rngs):
        """
        初始化Pi0FAST模型。
        Args:
            config: Pi0FASTConfig实例，包含模型配置。
            rngs: nnx.Rngs实例，用于管理随机数生成器。
        """
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len) # 调用基类构造函数
        paligemma_config = _gemma.get_config(config.paligemma_variant) # 获取PaliGemma配置
        # TODO: 将gemma重写为NNX。目前，使用bridge。
        llm = nnx_bridge.ToNNX( # 将Gemma模型转换为NNX模块
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype, # 嵌入层数据类型
                cache_dtype=config.dtype, # 缓存数据类型
            )
        )
        llm.lazy_init(rngs=rngs, method="init") # 延迟初始化LLM模块
        img = nnx_bridge.ToNNX( # 将SigLip模型转换为NNX模块
            _siglip.Module(
                num_classes=paligemma_config.width, # 类别数量
                variant="So400m/14", # SigLip变体
                pool_type="none", # 池化类型
                scan=True, # 是否使用扫描
                dtype_mm=config.dtype, # 矩阵乘法数据类型
            )
        )
        # 使用伪观察数据初始化图像编码器，train=False表示不进行训练
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img) # 将LLM和图像编码器组合成一个NNX字典

    @at.typecheck # 启用类型检查
    def embed_inputs(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        """
        嵌入输入观察数据，包括图像和令牌化提示。
        返回嵌入、输入掩码和自回归掩码。
        """
        input_mask = [] # 输入掩码列表
        ar_mask = [] # 自回归掩码列表
        token_embeddings = [] # 令牌嵌入列表
        # 嵌入图像
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False) # 通过图像编码器嵌入图像

            token_embeddings.append(image_token_embeddings) # 添加图像令牌嵌入
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            ) # 为图像令牌创建输入掩码
            # 图像令牌相互关注 --> AR mask = 0
            ar_mask.append(0 * input_mask[-1]) # 图像令牌的自回归掩码为0

        # 添加令牌化输入
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required" # 确保令牌化提示存在
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required" # 确保令牌化提示掩码存在
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required" # 确保令牌自回归掩码存在
        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True) # 通过LLM嵌入令牌化提示
        token_embeddings.append(tokenized_inputs_embeddings) # 添加令牌化输入嵌入
        input_mask.append(obs.tokenized_prompt_mask) # 添加令牌化提示掩码
        ar_mask.append(obs.token_ar_mask) # 添加令牌自回归掩码

        # 返回嵌入、输入掩码和自回归掩码
        return (
            jnp.concatenate(token_embeddings, axis=1), # 拼接所有令牌嵌入
            jnp.concatenate(input_mask, axis=1), # 拼接所有输入掩码
            jnp.concatenate(ar_mask, axis=1), # 拼接所有自回归掩码
        )

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """
        计算模型的损失。
        """
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        ) # 预处理观察数据

        # 计算输入：一次性对前缀+后缀进行一次大的前向传播
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation) # 嵌入输入
        attn_mask = make_attn_mask(input_mask, ar_mask) # 创建注意力掩码

        # 计算one-hot目标：我们预测*下一个*令牌，因此将输入令牌移动一个位置。
        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:], # 目标是下一个令牌
            self.PaliGemma.llm.module.vocab_size, # 词汇表大小
        )

        # 每个输入预测*下一个*令牌，所以我们不输入最后一个令牌。
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1], # 嵌入前缀（不包含最后一个令牌）
            mask=attn_mask[:, :-1, :-1], # 注意力掩码
            return_prelogits=True, # 返回预logits
        )

        # 仅解码目标令牌的logits以节省内存
        # （解码矩阵乘法很大，因为它是一个 seq_len x vocab_size 的密集层）。
        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets.shape[1] :], # 预logits（仅包含目标部分的）
        )
        logp = jax.nn.log_softmax(logits, axis=-1) # 计算对数softmax

        # 计算令牌目标的交叉熵损失
        assert observation.token_loss_mask is not None, "Token loss mask is required" # 确保令牌损失掩码存在
        loss_mask = observation.token_loss_mask[:, 1:] # 损失掩码（不包含第一个令牌）
        token_pplx = jnp.sum(targets * logp, axis=-1) # 计算令牌困惑度
        return -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1) # 返回平均损失

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256, # 最大解码步数
        temperature: float = 0.0, # 采样温度
    ) -> _model.Actions:
        """
        根据观察数据采样动作。
        """
        # TODO: 这是一个获取图像键的hack。
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        ) # 预处理观察数据

        # 嵌入输入
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation) # 嵌入前缀输入
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask) # 创建前缀注意力掩码

        # 将所有输入令牌序列从左对齐转换为右对齐，高效KV缓存
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1] # 预填充大小
        prefill_len = jnp.sum(prefix_mask, axis=-1) # 预填充长度
        prefix_start = prefill_size - prefill_len # 前缀开始位置

        # 首先通过前缀的前向传播填充KV缓存
        # 填充注意力掩码以设置KV缓存的大小（预填充大小 + 最大解码步数）
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1 # 前缀位置
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, # 嵌入前缀
            mask=prefix_attn_mask, # 注意力掩码
            positions=prefix_positions, # 位置信息
            decode=True # 启用解码模式
        )

        # 准备解码 -- 最终的logit解码第一个令牌
        last_logit = prefix_logits[:, -1:] # 最后一个logit
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps)) # 初始化输出令牌

        def step(carry):
            """解码循环中的一步。"""
            last_logit, output_tokens, cache, _, step = carry

            # 从最后一个logit中采样令牌
            if temperature > 0.0:
                last_logit = last_logit / temperature # 应用温度
                token = jax.random.categorical(rng, last_logit, axis=-1) # 从分类分布中采样
            else:
                token = jnp.argmax(last_logit, axis=-1) # 选择概率最高的令牌
            # 将采样到的令牌放置到输出令牌数组中
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # 检查是否提前停止 --> 如果所有批次元素都包含EOS令牌则停止
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1) # 检查是否有EOS令牌
            all_eos = jnp.all(has_eos) # 检查所有批次是否都包含EOS令牌

            # 解码一步
            token_embedding = self.PaliGemma.llm(token, embed_only=True) # 嵌入当前令牌
            positions = prefill_len[:, None] + step + 1 # 计算当前令牌的位置
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            ) # 创建当前步的注意力掩码
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, # 嵌入令牌
                mask=mask, # 注意力掩码
                positions=positions, # 位置信息
                decode=True, # 启用解码模式
                kv_cache=cache # 传入KV缓存
            )

            return last_logit, output_tokens, kv_cache, all_eos, step + 1 # 返回更新后的状态

        def cond(carry):
            """解码循环的条件。"""
            _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps) # 当未达到所有EOS且未超过最大解码步数时继续

        # 使用jax.lax.while_loop以便我们可以jit整个解码循环。
        _, output_tokens, _, _, _ = jax.lax.while_loop(cond, step, (last_logit, output_tokens, kv_cache, False, 0)) # 执行解码循环
        return output_tokens # 返回采样到的动作（令牌）
