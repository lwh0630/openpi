import contextlib
import logging

import jax
import numpy as np

BATCH_AXIS = "batch"  # 批处理轴
FSDP_AXIS = "fsdp"  # FSDP（Fully Sharded Data Parallel）轴
# 在 FSDP 中,我们在批处理轴和 FSDP 轴上都对数据进行分片。
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)  # 数据轴,结合了批处理和FSDP轴


class _MeshState:
    """内部类,用于存储当前活跃的 JAX Mesh 对象。"""

    active_mesh: jax.sharding.Mesh | None = None


def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    """
    创建 JAX Mesh 对象,用于定义设备拓扑和分片策略。

    Args:
        num_fsdp_devices: 用于 FSDP 分片的设备数量。

    Returns:
        一个 JAX Mesh 对象。

    Raises:
        ValueError: 如果设备总数不能被 `num_fsdp_devices` 整除。
    """
    if jax.device_count() % num_fsdp_devices != 0:
        raise ValueError(f"设备数量 {jax.device_count()} 必须能被 FSDP 设备数量 {num_fsdp_devices} 整除。")
    # 计算 Mesh 的形状,其中第一个维度是批处理轴,第二个维度是 FSDP 轴。
    mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
    # 使用 jax.make_mesh 创建 Mesh 对象,并命名轴。
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))


@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    """
    一个上下文管理器,用于设置和管理全局活跃的 JAX Mesh。
    这是在 JAX 团队提供更好的 API 之前,推荐的维护全局 Mesh 引用的方式。
    它只在下面的 `activation_sharding_constraint` 中使用。

    Args:
        mesh: 要设置为活跃 Mesh 的 JAX Mesh 对象。

    Yields:
        无。

    Raises:
        ValueError: 如果尝试嵌套 `set_mesh` 上下文管理器。
    """
    if _MeshState.active_mesh is not None:
        raise ValueError("不能嵌套 set_mesh 上下文管理器。")
    _MeshState.active_mesh = mesh  # 设置活跃 Mesh
    try:
        yield  # 执行上下文中的代码
    finally:
        _MeshState.active_mesh = None  # 退出上下文时清除活跃 Mesh


def activation_sharding_constraint(pytree):
    """
    对 PyTree 中的激活值应用分片约束。
    如果当前没有活跃的 Mesh,则不应用任何约束。

    Args:
        pytree: 要应用分片约束的 PyTree。

    Returns:
        应用了分片约束的 PyTree,如果当前没有活跃的 Mesh,则返回原始 PyTree。
    """
    if _MeshState.active_mesh is None:
        return pytree
    # 使用 jax.lax.with_sharding_constraint 应用分片约束。
    # 分片规范为 DATA_AXIS,这意味着数据将在批处理和 FSDP 轴上分片。
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS))
    )


def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int = 4,  # 4 MiB (最小分片大小,单位：兆字节)
    log: bool = False,  # 是否记录分片决策
):
    """
    根据 Mesh 形状对 PyTree 中的数组应用 FSDP 分片。

    Args:
        pytree: 要应用分片的 PyTree。注意,只有具有 `.shape` 属性的数组类型才会被考虑分片。
        mesh: 用于对 PyTree 应用分片的 Mesh。
        min_size_mbytes: 考虑分片的数组的最小大小（单位：MiB）。任何小于此值的数组都将被复制。
        log: 如果为 True,将记录正在考虑分片的数组的分片决策。

    Returns:
        应用了分片的 PyTree。
    """
    min_size_bytes = min_size_mbytes * 2**20  # 将 MiB 转换为字节

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        """
        内部函数,根据数组的属性和 Mesh 形状决定如何分片单个数组。
        """
        # 如果 FSDP 实际上不会被使用（即 FSDP 轴的大小为 1）,则复制所有内容以避免不必要的日志记录。
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # 复制标量和向量数组（即没有 shape 属性或维度小于 2 的数组）。
        if not hasattr(array, "shape"):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # 复制小数组。
        if (arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # 沿最大且可被 FSDP 维度整除的轴对矩阵和更大的张量进行分片。
        axes = np.argsort(array.shape)[::-1]  # 按轴的大小降序排列
        spec = [None] * len(axes)  # 初始化分片规范
        for i in axes:
            # 如果当前轴的大小可以被 FSDP 轴的设备数量整除,则在此轴上进行分片。
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(
                        f"正在将 {jax.tree_util.keystr(kp)} (形状: {array.shape}, 大小: {arr_size / 2**20:.2f} MiB) 沿轴 {i} 分片"
                    )
                spec[i] = FSDP_AXIS  # 设置分片轴
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))

        # 如果没有找到有效的分片方式,则复制。
        if log:
            logging.warning(
                f"无法为形状为 {array.shape} 的 {jax.tree_util.keystr(kp)} 找到有效的分片方式,Mesh 形状为 {mesh.shape}"
            )
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 使用 tree_map_with_path 对 PyTree 中的每个数组应用 _shard_arr 函数。
    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)
