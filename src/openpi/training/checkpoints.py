from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    """
    初始化检查点目录。

    Args:
        checkpoint_dir: 检查点目录的路径。
        keep_period: 保留检查点的周期。
        overwrite: 如果目录存在是否覆盖。
        resume: 如果目录存在是否恢复训练。

    Returns:
        一个包含 CheckpointManager 实例和是否恢复训练的布尔值的元组。
    """
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()  # 将路径解析为绝对路径
    resuming = False  # 标记是否正在恢复训练
    if checkpoint_dir.exists():  # 如果检查点目录已存在
        if overwrite:  # 如果指定覆盖
            checkpoint_dir.rmtree()  # 删除现有目录
            checkpoint_dir.mkdir(parents=True, exist_ok=True)  # 创建新目录
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")  # 记录日志
        elif resume:  # 如果指定恢复
            resuming = True  # 设置恢复标志为 True
        else:
            raise FileExistsError(  # 如果目录存在且未指定覆盖或恢复，则抛出错误
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)  # 确保检查点目录存在

    mngr = ocp.CheckpointManager(  # 创建 Orbax CheckpointManager 实例
        checkpoint_dir,
        item_handlers={  # 定义不同项的处理器
            "assets": CallbackHandler(),  # 资产使用 CallbackHandler
            "train_state": ocp.PyTreeCheckpointHandler(),  # 训练状态使用 PyTreeCheckpointHandler
            "params": ocp.PyTreeCheckpointHandler(),  # 参数使用 PyTreeCheckpointHandler
        },
        options=ocp.CheckpointManagerOptions(  # 配置检查点管理器选项
            max_to_keep=1,  # 最多保留一个检查点
            keep_period=keep_period,  # 保留检查点的周期
            create=False,  # 不在初始化时创建检查点
            async_options=ocp.AsyncOptions(timeout_secs=7200),  # 异步操作超时时间
        ),
    )

    # 特殊情况：检查点目录存在且用户请求恢复训练，但训练运行尚未保存第一个检查点。
    # 这种情况下，我们实际上不希望训练脚本尝试恢复检查点，因为它会失败。
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False  # 放弃恢复

    return mngr, resuming  # 返回检查点管理器和恢复标志


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    """
    保存训练状态。

    Args:
        checkpoint_manager: CheckpointManager 实例。
        state: 当前的训练状态。
        data_loader: 数据加载器实例。
        step: 当前训练步数。
    """
    def save_assets(directory: epath.Path):
        # 保存归一化统计信息。
        data_config = data_loader.data_config()  # 获取数据配置
        norm_stats = data_config.norm_stats  # 获取归一化统计信息
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)  # 保存归一化统计信息

    # 将可用于推理的参数拆分为单独的项。
    with at.disable_typechecking():  # 禁用类型检查
        train_state, params = _split_params(state)  # 拆分训练状态和参数
    items = {  # 定义要保存的项
        "assets": save_assets,  # 资产保存函数
        "train_state": train_state,  # 训练状态
        "params": {"params": params},  # 参数
    }
    checkpoint_manager.save(step, items)  # 保存检查点


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    """
    恢复训练状态。

    Args:
        checkpoint_manager: CheckpointManager 实例。
        state: 当前的训练状态（用于类型提示和结构）。
        data_loader: 数据加载器实例（此处未使用）。
        step: 要恢复的训练步数，如果为 None 则恢复最新步数。

    Returns:
        恢复后的训练状态。
    """
    del data_loader  # 此处不使用 data_loader

    with at.disable_typechecking():  # 禁用类型检查
        # 将可用于推理的参数拆分为单独的项。
        train_state, params = _split_params(state)  # 拆分训练状态和参数（用于定义恢复结构）
        restored = checkpoint_manager.restore(  # 恢复检查点
            step,
            items={  # 定义要恢复的项
                "train_state": train_state,  # 训练状态
                "params": {"params": params},  # 参数
            },
        )
    return _merge_params(restored["train_state"], restored["params"])  # 合并恢复后的参数和训练状态


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    """
    加载归一化统计信息。

    Args:
        assets_dir: 资产目录的路径。
        asset_id: 资产的 ID。

    Returns:
        包含归一化统计信息的字典，如果不存在则为 None。
    """
    norm_stats_dir = epath.Path(assets_dir) / asset_id  # 构建归一化统计信息目录路径
    norm_stats = _normalize.load(norm_stats_dir)  # 从指定目录加载归一化统计信息
    logging.info(f"Loaded norm stats from {norm_stats_dir}")  # 记录加载信息
    return norm_stats  # 返回归一化统计信息


class Callback(Protocol):
    """一个用于回调函数的协议。"""
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """
    一个用于异步调用任意函数的 CheckpointHandler。仅用于保存，不支持恢复。
    """

    def save(self, directory: epath.Path, args: CallbackSave):
        """
        保存回调函数。

        Args:
            directory: 保存目录。
            args: CallbackSave 参数。
        """
        if jax.process_index() == 0:  # 仅在主进程中执行
            args.callback(directory)  # 调用回调函数

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        """
        异步保存回调函数。

        Args:
            directory: 保存目录。
            args: CallbackSave 参数。

        Returns:
            一个包含 Future 对象的列表。
        """
        # 使用 asyncio.to_thread 在单独的线程中运行同步 save 方法，以避免阻塞主事件循环。
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        """
        恢复回调函数（不支持）。
        """
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    """用于 CallbackHandler 保存操作的参数。"""
    callback: Callback  # 要执行的回调函数


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs):
    """用于 CallbackHandler 恢复操作的参数（目前不支持）。"""
    ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    """
    将训练状态中的参数拆分出来。

    如果存在 EMA 参数，则将其作为主要参数，并将训练状态中的 EMA 参数设为 None。
    否则，将训练状态中的 params 作为主要参数，并将训练状态中的 params 设为空字典。

    Args:
        state: 训练状态。

    Returns:
        一个包含修改后的训练状态和拆分出的参数的元组。
    """
    if state.ema_params is not None:  # 如果存在 EMA 参数
        params = state.ema_params  # 将 EMA 参数作为主要参数
        train_state = dataclasses.replace(state, ema_params=None)  # 训练状态中的 EMA 参数设为 None
    else:
        params = state.params  # 否则将 params 作为主要参数
        train_state = dataclasses.replace(state, params={})  # 训练状态中的 params 设为空字典
    return train_state, params  # 返回拆分后的训练状态和参数


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    """
    将参数合并回训练状态。

    此函数假设 `_split_params` 中的逻辑被反转。
    如果 `train_state.params` 非空，则表示在拆分时使用了 EMA 参数。

    Args:
        train_state: 训练状态。
        params: 包含要合并的参数的字典。

    Returns:
        合并参数后的训练状态。
    """
    # 反转 `_split_params` 中的逻辑。假设 `params` 的存在意味着在拆分时使用了 EMA 参数。
    if train_state.params:  # 如果训练状态的 params 非空，说明之前是 ema_params
        return dataclasses.replace(train_state, ema_params=params["params"])  # 将恢复的参数合并为 ema_params
    return dataclasses.replace(train_state, params=params["params"])  # 否则合并为 params
