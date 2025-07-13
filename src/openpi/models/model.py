import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

ArrayT = TypeVar("ArrayT", at.Array, jax.ShapeDtypeStruct)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",               # 一个基础视图（机器人环境的全局视图）
    "left_wrist_0_rgb",         # 左手腕视图（来自左手腕摄像头）
    "right_wrist_0_rgb",        # 右手腕视图（来自右手腕摄像头）
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)

# 下面代码定义了模型输入数据格式
# 输入是一个嵌套字典，包含观测数据和动作数据
# 最终会转换为Observation和Actions对象（模型实际使用的数据结构）
# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad] # ah可能表示动作历史长度，ad表示动作维度
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """预处理观测数据，包括图像增强（如果train=True）、调整尺寸（如果需要）和填充默认图像掩码（如果需要）。

    Args:
        rng: 随机数生成器密钥，用于数据增强。如果没有增强需求则为None
        observation: 包含图像、掩码、状态和提示的输入观测数据
        train: 是否应用训练专用的数据增强。默认为False
        image_keys: 要处理的图像键序列。默认为IMAGE_KEYS
        image_resolution: 图像的目标分辨率（高度，宽度）。默认为IMAGE_RESOLUTION

    Returns:
        Observation: 处理后的观测数据，包含增强/调整尺寸后的图像和更新后的掩码

    Raises:
        ValueError: 如果观测数据中缺少必需的图像键
    """

    # 验证观测数据中是否包含所有需要的图像键
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        # 如果图像尺寸与目标分辨率不符，则调整尺寸
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

         # 训练模式下应用数据增强
        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    # 随机裁剪原图的95%区域，保持内容完整性同时引入变化
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    # 将裁剪后的图像缩放回原尺寸
                    augmax.Resize(width, height),
                    # 随机旋转-5度到5度之间，增加视角变化
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                # 颜色抖动增强：亮度±30%，对比度±40%，饱和度±50%
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            # 为批次中的每个图像生成独立的随机种子
            sub_rngs = jax.random.split(rng, image.shape[0])
            # 使用vmap批量应用增强链，保持JAX的高效并行计算
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    # 生成图像掩码字典，确保每个处理后的图像都有对应的掩码
    out_masks = {}
    for key in out_images:
        # 如果原始观测数据中没有该图像的掩码
        if key not in observation.image_masks:
            # do not mask by default
            # 创建默认全1掩码（表示所有像素都有效）
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            # 如果原始观测数据中有该图像的掩码，则直接使用
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """所有模型的共享配置基类。具体模型应继承此类并实现`create`方法来创建对应模型。
    
    属性:
        action_dim: 动作空间的维度大小
        action_horizon: 动作序列的长度
        max_token_len: 分词后提示文本的最大长度
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """抽象属性，定义模型类型。
        
        返回:
            模型类型枚举值(ModelType) pi0 or pi0_fast
        """

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """抽象方法，创建并初始化一个新模型。
        
        参数:
            rng: 用于参数初始化的随机数生成器密钥
            
        返回:
            新创建的BaseModel实例，包含初始化参数
        """

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """使用给定的参数创建模型实例。
        
        参数:
            params: 要加载的模型参数
            remove_extra_params: 是否移除模型中不存在的额外参数
            
        返回:
            使用给定参数初始化的BaseModel实例
            
        注意:
            在加载前会验证参数形状，并可选地过滤多余参数
        """
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """抽象方法，定义模型的输入规范。
        
        参数:
            batch_size: 输入规范的批次大小
            
        返回:
            包含(observation_spec, action_spec)的元组，每个都是jax.ShapeDtypeStruct
        """

    def fake_obs(self, batch_size: int = 1) -> Observation:
        """生成符合模型输入规范的假观测数据。
        
        参数:
            batch_size: 生成数据的批次大小
            
        返回:
            填充1的Observation对象，符合指定的形状和数据类型
        """
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        """生成符合模型输入规范的假动作数据。
        
        参数:
            batch_size: 生成数据的批次大小
            
        返回:
            填充1的Actions对象，符合指定的形状和数据类型
        """
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """所有模型实现的基类。具体模型应继承此类并通过super().__init__()初始化共享属性(action_dim, action_horizon和max_token_len)。

    属性:
        action_dim: 动作空间的维度大小
        action_horizon: 动作序列的长度(时间步数)
        max_token_len: 输入提示文本的最大分词长度
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...
    """抽象方法：计算模型损失(用于训练或评估)。

        参数:
            rng: JAX随机数生成器密钥
            observation: 输入观测数据，包含图像、状态和可选的语言提示
            actions: 真实动作数据(用于监督学习)
            train: 是否为训练模式(影响如dropout等行为)

        返回:
            计算得到的损失张量，形状[*b ah](*b表示任意批次维度，ah表示动作序列长度)
    """

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation) -> Actions: ...
    """抽象方法：从模型策略中采样动作。

        参数:
            rng: JAX随机数生成器密钥
            observation: 输入观测数据，包含图像、状态和可选的语言提示

        返回:
            从模型策略中采样的动作序列
    """

def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """从检查点恢复非结构化参数PyTree
    
    该函数兼容以下两种检查点格式：
    1. 训练过程中通过[save_state](openpi/training/checkpoints.py#L64-L85)保存的检查点
    2. openpi发布的预训练模型检查点

    Args:
        params_path: 检查点目录的本地路径
        restore_type: 参数恢复的目标类型，可设置为`np.ndarray`将参数作为numpy数组加载
        dtype: 恢复参数时使用的数据类型。如未提供，则使用检查点中的原始类型
        sharding: 参数的分片策略。如未提供，参数将在所有设备上复制

    Returns:
        恢复后的参数字典(PyTree结构)

    Raises:
        FileNotFoundError: 当指定的检查点路径不存在时抛出
    """
    params_path = pathlib.Path(params_path).resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Model params not found at: {params_path}")

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
