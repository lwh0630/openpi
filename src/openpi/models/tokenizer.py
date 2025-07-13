"""PaliGemma和FAST模型的Tokenizer实现。

包含两个主要类：
1. PaligemmaTokenizer: 基础PaliGemma模型的分词器
2. FASTTokenizer: 支持动作序列的增强分词器
"""

import logging

import numpy as np
import sentencepiece
from transformers import AutoProcessor

import openpi.shared.download as download


class PaligemmaTokenizer:
    """PaliGemma模型的分词器实现。
    
    使用SentencePiece进行文本分词，支持最大长度限制和自动填充/截断。
    
    Args:
        max_len: 最大分词长度，默认为48
    """
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        # 下载并加载PaliGemma分词器模型
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """将输入文本分词为token序列和注意力掩码。
        
        Args:
            prompt: 输入文本
            
        Returns:
            包含两个元素的元组:
                - tokens: token ID序列的numpy数组
                - mask: 注意力掩码的numpy数组(1表示真实token,0表示填充)
        """
        # 清理文本：去除空格、替换特殊字符
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        
        # 分词并添加特殊换行符token
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        
        # 根据最大长度进行填充或截断处理
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:
    """结合PaliGemma和FAST分词器的增强实现。
    
    支持文本、状态和动作序列的联合分词。
    
    Args:
        max_len: 最大分词长度，默认为256
        fast_tokenizer_path: FAST分词器路径，默认为"physical-intelligence/fast"
    """
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # 下载并加载基础PaliGemma分词器
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # 初始化FAST动作分词器
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # PaliGemma词表中跳过的特殊token数量

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """将输入文本、状态和动作序列联合分词。
        
        Args:
            prompt: 任务描述文本
            state: 环境状态向量
            actions: 可选的动作序列(预测时为None)
            
        Returns:
            包含四个元素的元组:
                - tokens: 联合token序列
                - token_mask: 注意力掩码(1表示真实token,0表示填充)
                - ar_mask: 自回归掩码(0表示前缀双向注意力,1表示后缀因果注意力)
                - loss_mask: 损失掩码(False表示前缀,True表示后缀)
        """
        # 清理并标准化输入文本
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # 将连续状态离散化为256个bins
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # 创建包含任务描述和状态的前缀
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        # 处理动作token(如果提供)
        if actions is not None:
            # 将FAST动作token转换为PaliGemma词表空间
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # 创建包含动作token的后缀
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # 创建不同注意力和损失模式的掩码
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)

        # 将序列填充或截断到最大长度
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        """从模型输出token中提取动作序列。
        
        Args:
            tokens: 模型输出token序列
            action_horizon: 预测的动作步数
            action_dim: 每个动作的维度
            
        Returns:
            解码后的动作序列numpy数组
        """
        # 将token解码回文本
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # 如果未找到动作token则返回零值
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # 提取并将动作token转换回FAST格式
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        """将FAST动作token映射到PaliGemma词表空间，通过从词表末尾偏移实现。"""
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
