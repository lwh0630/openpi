"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    """
    一个数据转换函数，用于从字典中移除所有字符串类型的键值对。
    字符串类型的数据不适用于计算归一化统计量，也不受JAX支持。
    """

    def __call__(self, x: dict) -> dict:
        """
        执行转换操作。

        Args:
            x: 输入字典，包含各种类型的数据。

        Returns:
            一个新字典，其中移除了所有值是字符串类型的键值对。
        """
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """
    创建并返回一个PyTorch数据加载器。

    Args:
        data_config: 数据配置对象，包含数据集的相关信息。
        action_horizon: 动作序列的长度。
        batch_size: 批处理大小。
        model_config: 模型配置对象。
        max_frames: 可选参数，指定要处理的最大帧数。

    Returns:
        一个元组，包含数据加载器实例和批次总数。

    Raises:
        ValueError: 如果data_config中没有repo_id。
    """
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    # 创建PyTorch数据集
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    # 使用转换函数封装数据集，移除字符串并应用其他数据转换
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,  # 应用重打包转换
            *data_config.data_transforms.inputs,  # 应用数据转换
            # 移除字符串，因为JAX不支持且计算归一化统计量不需要
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True  # 如果指定了最大帧数且小于数据集大小，则进行混洗
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False  # 否则不混洗
    # 创建PyTorch数据加载器
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=8,  # 设置工作进程数
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """
    创建并返回一个RLDS（Reinforcement Learning Datasets）数据加载器。

    Args:
        data_config: 数据配置对象。
        action_horizon: 动作序列的长度。
        batch_size: 批处理大小。
        max_frames: 可选参数，指定要处理的最大帧数。

    Returns:
        一个元组，包含数据加载器实例和批次总数。
    """
    # 创建RLDS数据集
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    # 使用转换函数封装可迭代数据集
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,  # 应用重打包转换
            *data_config.data_transforms.inputs,  # 应用数据转换
            # 移除字符串，因为JAX不支持且计算归一化统计量不需要
            RemoveStrings(),
        ],
        is_batched=True,  # 指示数据集是否已批处理
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    # 创建RLDS数据加载器
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    """
    主函数，用于计算给定配置的归一化统计量。

    Args:
        config_name: 配置的名称。
        max_frames: 可选参数，指定要处理的最大帧数。
    """
    # 获取指定名称的配置
    config = _config.get_config(config_name)
    # 根据配置创建数据配置
    data_config = config.data.create(config.assets_dirs, config.model)

    # 根据数据源类型选择创建相应的数据加载器
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, max_frames
        )

    # 定义需要计算统计量的键
    keys = ["state", "actions"]
    # 初始化每个键的运行统计对象
    stats = {key: normalize.RunningStats() for key in keys}

    # 遍历数据加载器中的每个批次，计算统计量
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            # 获取当前批次中对应键的值
            values = np.asarray(batch[key][0])
            # 更新运行统计量。将值重塑为二维数组，其中最后一维是特征维度。
            stats[key].update(values.reshape(-1, values.shape[-1]))

    # 获取最终的归一化统计量（均值和标准差）
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # 构建输出路径
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    # 保存归一化统计量到指定路径
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    # 使用tyro库解析命令行参数并调用main函数
    tyro.cli(main)
