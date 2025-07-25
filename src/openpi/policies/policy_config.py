from collections.abc import Sequence
import dataclasses
import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass
class PolicyConfig:
    """策略配置类，用于定义策略的各种参数。"""
    model: _model.BaseModel
    norm_stats: dict[str, transforms.NormStats]

    input_layers: Sequence[transforms.DataTransformFn]  # 输入数据转换层序列
    output_layers: Sequence[transforms.DataTransformFn] # 输出数据转换层序列

    model_type: _model.ModelType = _model.ModelType.PI0 # 模型类型，默认为PI0
    default_prompt: str | None = None # 默认提示词，如果没有提供则为None
    sample_kwargs: dict[str, Any] | None = None # 采样时的额外关键字参数


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy:
    """从一个已训练的检查点创建策略。

    Args:
        train_config: 用于创建模型的训练配置。
        checkpoint_dir: 加载模型的目录。
        repack_transforms: 可选的转换组，将在所有其他转换之前应用。
        sample_kwargs: 传递给 `sample_actions` 方法的关键字参数。如果未提供，将使用默认参数。
        default_prompt: 策略使用的默认提示词。如果输入数据中不存在提示词，将注入此提示词。
        norm_stats: 策略使用的归一化统计数据。如果未提供，将从检查点目录加载归一化统计数据。
    """
    # 初始化 repack_transforms，如果未提供则创建一个空的 Group
    repack_transforms = repack_transforms or transforms.Group()
    # 尝试下载检查点目录，如果它是一个URL或者需要下载
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...") # 记录日志：正在加载模型
    # 加载模型，并从检查点目录中恢复模型参数，使用 bfloat16 数据类型
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    # 根据训练配置创建数据配置
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # 如果未提供归一化统计数据，则从检查点加载
        # 我们从检查点而不是配置的资产目录加载归一化统计数据，以确保
        # 策略使用与原始训练过程相同的归一化统计数据。
        if data_config.asset_id is None:
            # 如果资产ID为空，则抛出错误
            raise ValueError("Asset id is required to load norm stats.")
        # 从检查点目录的 assets 子目录中加载归一化统计数据
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # 返回一个策略实例
    return _policy.Policy(
        model, # 策略使用的模型
        transforms=[
            *repack_transforms.inputs, # 应用重新打包的输入转换
            transforms.InjectDefaultPrompt(default_prompt), # 注入默认提示词
            *data_config.data_transforms.inputs, # 应用数据配置中的输入转换
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm), # 应用归一化
            *data_config.model_transforms.inputs, # 应用模型配置中的输入转换
        ],
        output_transforms=[
            *data_config.model_transforms.outputs, # 应用模型配置中的输出转换
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm), # 应用反归一化
            *data_config.data_transforms.outputs, # 应用数据配置中的输出转换
            *repack_transforms.outputs, # 应用重新打包的输出转换
        ],
        sample_kwargs=sample_kwargs, # 采样时的关键字参数
        metadata=train_config.policy_metadata, # 策略元数据
    )
