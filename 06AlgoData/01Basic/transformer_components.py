import os
import sys

_current_dir = os.path.dirname(__file__)
_code_dir = os.path.join(_current_dir, 'code')
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

try:
    # 优先使用与当前笔记本掩码构造一致的实现
    from Practice03MachineTrans import (
        Embedding,
        PositionalEncoding,
        MultiHeadAttention,
        FeedForward,
        SublayerConnection,
        EncoderLayer,
        DecoderLayer,
        Encoder,
        Decoder,
        Transformer,
        Generator,
    )
except ModuleNotFoundError:
    from Practice02TransformerTrain import (
        Embedding,
        PositionalEncoding,
        MultiHeadAttention,
        FeedForward,
        SublayerConnection,
        EncoderLayer,
        DecoderLayer,
        Encoder,
        Decoder,
        Transformer,
        Generator,
    )

__all__ = [
    'Embedding',
    'PositionalEncoding',
    'MultiHeadAttention',
    'FeedForward',
    'SublayerConnection',
    'EncoderLayer',
    'DecoderLayer',
    'Encoder',
    'Decoder',
    'Transformer',
    'Generator',
]


