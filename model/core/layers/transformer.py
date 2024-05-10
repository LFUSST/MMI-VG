from typing import Optional
from typing import Type
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as f
from torch.nn.modules.transformer import _get_clones

from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import ConvModule, build_activation_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding, POSITIONAL_ENCODING

from mmdet.models.utils import build_linear_layer
from mmdet.models.utils.builder import TRANSFORMER
from .VLTVG import DiscriminativeFeatEnc

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "0, 1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enable = True


class LinearModule(nn.Module):
    """A linear block that bundles linear/activation/dropout layers.

    This block simplifies the usage of linear layers, which are commonly
    used with an activation layer (e.g., ReLU) and Dropout layer (e.g., Dropout).
    It is based upon three build methods: `build_linear_layer()`,
    `build_activation_layer()` and `build_dropout`.

    Args:
        linear (dict): Config dict for activation layer. Default: dict(type='Linear', bias=True)
        act (dict): Config dict for activation layer. Default: dict(type='ReLU', inplace=True).
        drop (dict): Config dict for dropout layer. Default: dict(type='Dropout', drop_prob=0.5)
    """

    def __init__(self,
                 linear=dict(type='Linear', bias=True),
                 act=dict(type='ReLU', inplace=True),
                 drop=dict(type='Dropout', drop_prob=0.5)):
        super(LinearModule, self).__init__()
        assert linear is None or isinstance(linear, dict)
        assert act is None or isinstance(act, dict)
        assert drop is None or isinstance(drop, dict)
        assert 'in_features' in linear and 'out_features' in linear

        self.with_activation = act is not None
        self.with_drop = drop is not None

        self.fc = build_linear_layer(linear)

        if self.with_activation:
            self.activate = build_activation_layer(act)

        if self.with_drop:
            self.drop = build_dropout(drop)

    def forward(self, input):
        input = self.fc(input)

        if self.with_activation:
            input = self.activate(input)

        if self.with_drop:
            input = self.drop(input)

        return input


def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos


@POSITIONAL_ENCODING.register_module()
class LearnedPositionalEncoding1D(nn.Module):
    """1D Position embedding with learnable embedding weights.

    Args:
        num_feature (int): The feature dimension for each position.
        num_embedding (int, optional): The dictionary size of embeddings.
            Default 5.
    """

    def __init__(self,
                 num_embedding=5,
                 num_feature=256,
                 padding_idx=-1,
                 ):
        super(LearnedPositionalEncoding1D, self).__init__()
        self.num_feature = num_feature

        self.num_embedding = num_embedding
        self.embedding = nn.Embedding(
            num_embedding, num_feature, padding_idx=padding_idx if padding_idx >= 0 else None)

    def forward(self, seq_in_embeds):
        """
        Args:
            seq_in_embeds (tensor): [bs, 5/num_ray*2+1, d_model].

        Returns:
            seq_in_pos_embeds (tensor): [bs, 5/num_ray*2+1, d_model].
        """
        seq_len = seq_in_embeds.size(1)
        position = torch.arange(seq_len, dtype=torch.long,
                                device=seq_in_embeds.device)
        position = position.unsqueeze(0).expand(seq_in_embeds.size()[:2])
        return self.embedding(position)


@POSITIONAL_ENCODING.register_module()
class SinePositionalEncoding2D(BaseModule):
    """Position encoding with sine and cosine functions.
    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feature,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SinePositionalEncoding2D, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                                                    'scale should be provided and in float or int type, ' \
                                                    f'found {type(scale)}'
        self.num_feature = num_feature
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feature, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feature)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# # 生成相对位置索引矩阵
# def make_token_bucket_position(bucket_size, max_position):
#     # 生成了一个 max_position x 1 大小的上下文位置和 1 x max_position 大小的张量记忆位置。
#     context_pos = torch.arange(max_position, dtype=torch.long)[:, None]  # 一列
#     memory_pos = torch.arange(max_position, dtype=torch.long)[None, :]  # 一行
#
#     relative_pos = context_pos - memory_pos
#     sign = torch.sign(relative_pos)  # 获取每个位置的正负符号
#     mid = bucket_size // 2
#     # 如果 relative_pos 的元素在 -mid 和 mid 之间，则 abs_pos 对应位置的元素值为 mid-1，否则为 torch.abs(relative_pos) 的绝对值。
#     # 根据相对位置张量 relative_pos，以及中间值 mid，计算了绝对位置编码。
#     abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos))  # 绝对位置编码
#
#     log_pos = torch.ceil(
#         torch.log(abs_pos / mid) / math.log((max_position - 1) / mid) * (mid - 1)) + mid  # torch.ceil 向上取整
#     log_pos = log_pos.int()
#     # 如果绝对位置编码小于等于中间值 mid，则使用相对位置编码作为位置索引；否则，使用经过转换的绝对位置编码乘以正负符号
#     bucket_pos = torch.where(abs_pos.le(mid), relative_pos, log_pos * sign).long()
#     # 返回的 bucket_pos 是最终的位置编码索引矩阵
#     return bucket_pos + bucket_size - 1
#
#
# def make_image_bucket_position(bucket_size, num_relative_distance):
#     coords_h = torch.arange(bucket_size)
#     coords_w = torch.arange(bucket_size)
#     coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#     coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#     # 计算了坐标之间的相对距离，2 是因为分别对 x/y 进行
#     relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#     relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#     relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
#     relative_coords[:, :, 1] += bucket_size - 1
#     # 对 x 分量进行了线性变换，这一步将坐标值映射到特定的范围内。
#     relative_coords[:, :, 0] *= 2 * bucket_size - 1
#     # * 2 代表长和宽两边
#     relative_position_index = torch.zeros(size=(bucket_size * bucket_size + 1,) * 2, dtype=relative_coords.dtype)
#     # 对每个坐标点的两个坐标轴上的值进行求和
#     relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#
#     # 将特定位置的值设置为 num_relative_distance 减去不同的偏移值。
#     # 这些特定位置的值可能用于标识起始位置、结束位置或者其他重要的相对位置关系。
#     relative_position_index[0, 0:] = num_relative_distance - 3
#     relative_position_index[0:, 0] = num_relative_distance - 2
#     relative_position_index[0, 0] = num_relative_distance - 1
#     return relative_position_index
#
#
# def Embedding(num_embeddings, embedding_dim, padding_idx=None, zero_init=False):
#     m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
#     nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
#     if padding_idx is not None:
#         nn.init.constant_(m.weight[padding_idx], 0)
#     if zero_init:
#         nn.init.constant_(m.weight, 0)
#     return m


class TransformerEncoderLayerWithPositionEmbedding(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayerWithPositionEmbedding,
              self).__init__(*args, **kwargs)

        self.w_resid = nn.Parameter(torch.ones(256), requires_grad=True)
        self.img_pos_ln = nn.LayerNorm(256)
        self.pos_scaling = float(256 / 8 * 2) ** -0.5
        self.pos_q_linear = nn.Linear(256, 256)
        self.pos_k_linear = nn.Linear(256, 256)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                rel_pos_bias: Optional[Tensor] = None,
                self_attn_bias: Optional[Tensor] = None) -> Tensor:
        q = k = with_pos_embed(src, pos)

        batch = q.size(0)
        pos = self.img_pos_ln(pos)
        pos_q = self.pos_q_linear(pos).view(
            q.size(0), q.size(1), 8, -1
        ).transpose(1, 2) * self.pos_scaling
        pos_k = self.pos_k_linear(pos).view(
            k.size(0), k.size(1), 8, -1
        ).transpose(1, 2)  # [b, 8, t, 32]
        self_attn_bias = torch.matmul(pos_q, pos_k.transpose(2, 3)).view(batch * 8, 400, -1)  # [b, 8, t, t]

        # self_attn_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))
        # self_attn_bias = (self_attn_bias + rel_pos_bias).view(batch * 8, 400, -1)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, attn_bias=self_attn_bias)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + torch.mul(self.w_resid, self.dropout2(src2))
        # src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoderWithPositionEmbedding(nn.TransformerEncoder):
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderWithPositionEmbedding,
              self).__init__(*args, **kwargs)

    #     self.image_bucket_size = 42
    #     # self.embed_image_positions = Embedding(self.image_bucket_size ** 2 + 1, 256)
    #
    #     image_num_rel_dis = (2 * self.image_bucket_size - 1) * (2 * self.image_bucket_size - 1) + 3
    #     image_rp_bucket = make_image_bucket_position(self.image_bucket_size, image_num_rel_dis)
    #     self.image_rel_pos_table_list = nn.ModuleList(
    #         [Embedding(image_num_rel_dis, 8, zero_init=True) for _ in range(6)]
    #     ).to(device)
    #
    #     self.register_buffer("image_rp_bucket", image_rp_bucket)
    #
    # def get_image_rel_pos_bias(self, image_position_ids, idx):
    #     bsz, seq_len = image_position_ids.shape  # seq_len代表图片的长和宽
    #     rp_bucket_size = self.image_rp_bucket.size(1)
    #
    #     # rp_bucket 的最终输出维度是 (bsz, seq_len, seq_len)，表示了图像中任意两个位置之间的相对位置索引。
    #     rp_bucket = self.image_rp_bucket.unsqueeze(0).expand(
    #         bsz, rp_bucket_size, rp_bucket_size
    #     ).gather(1, image_position_ids[:, :, None].expand(bsz, seq_len, rp_bucket_size)
    #              ).gather(2, image_position_ids[:, None, :].expand(bsz, seq_len, seq_len))
    #
    #     # 具体来说，这行代码的作用是：根据 rp_bucket 中的整数索引，从 self.image_rel_pos_table_list[idx].weight 嵌入矩阵中查找对应的嵌入向量
    #     values = f.embedding(rp_bucket, self.image_rel_pos_table_list[idx].weight)
    #     values = values.permute(0, 3, 1, 2)
    #     return values  # 图像相对位置偏置的张量

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                # rel_pos_bias: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        # batch = src.size(0)
        # image_position_idx = torch.arange(20).unsqueeze(0).expand(20, 20) + \
        #                      torch.arange(20).unsqueeze(1) * self.image_bucket_size + 1
        # image_position_idx = image_position_idx.view(-1)
        # image_position_ids = image_position_idx[None, :].expand(batch, 400).to(device)
        #
        # for idx, layer in enumerate(self.layers):
        #     rel_pos_bias = self.get_image_rel_pos_bias(image_position_ids, idx)

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           # rel_pos_bias=rel_pos_bias,
                           pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayerWithPositionEmbedding(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super(TransformerDecoderLayerWithPositionEmbedding,
              self).__init__(*args, **kwargs)

        self.w_resid = nn.Parameter(torch.ones(256), requires_grad=True)

        self.query_pos_ln = nn.LayerNorm(256)
        self.query_pos_scaling = float(256 / 8 * 2) ** -0.5
        self.query_pos_q_linear = nn.Linear(256, 256)
        self.query_pos_k_linear = nn.Linear(256, 256)

        self.img_pos_ln = nn.LayerNorm(256)
        self.pos_scaling = float(256 / 8 * 2) ** -0.5
        self.pos_q_linear = nn.Linear(256, 256)
        self.pos_k_linear = nn.Linear(256, 256)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                self_attn_bias: Optional[Tensor] = None,
                need_weights: bool = False
                # rel_pos_bias: Optional[Tensor] = None,
                # need_weights: bool = True
                ):
        q = k = with_pos_embed(tgt, query_pos)

        batch = q.size(0)

        query_pos = self.query_pos_ln(query_pos)
        query_pos_q = self.query_pos_q_linear(query_pos).view(
            q.size(0), q.size(1), 8, -1
        ).transpose(1, 2) * self.query_pos_scaling

        query_pos_k = self.query_pos_k_linear(query_pos).view(
            k.size(0), q.size(1), 8, -1
        ).transpose(1, 2)  # [b, 8, 3, 32]
        self_attn_bias = torch.matmul(query_pos_q, query_pos_k.transpose(2, 3)).view(
            batch * 8, q.size(1), -1)  # [b*8, 3, 3]

        # # token_len = q.size(1)
        # self_attn_bias = torch.matmul(query_pos_q, query_pos_k.transpose(2, 3))
        # self_attn_bias = (self_attn_bias + rel_pos_bias).view(batch * 8, token_len, -1)

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              attn_bias=self_attn_bias
                              )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        pos = self.img_pos_ln(pos)
        pos_k = self.pos_k_linear(pos).view(
            k.size(0), 400, 8, -1
        ).transpose(1, 2)  # [b, 8, 400, 32]
        cross_attn_bias = torch.matmul(query_pos_q, pos_k.transpose(2, 3)).view(
            batch * 8, q.size(1), -1)  # [b, 8, 3, 400]

        # token_to_image
        if need_weights:
            tgt2, attn_weights = self.multihead_attn(query=with_pos_embed(tgt, query_pos),
                                                     key=with_pos_embed(memory, pos),
                                                     value=memory, attn_mask=memory_mask,
                                                     key_padding_mask=memory_key_padding_mask,
                                                     attn_bias=cross_attn_bias,
                                                     need_weights=True)
        else:
            tgt2 = self.multihead_attn(query=with_pos_embed(tgt, query_pos),
                                       key=with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask,
                                       attn_bias=cross_attn_bias
                                       )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 全连接层
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + torch.mul(self.w_resid, self.dropout3(tgt2))
        # tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        if need_weights:
            return tgt, attn_weights
        else:
            return tgt


class TransformerDecoderWithPositionEmbedding(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm):
        super(TransformerDecoderWithPositionEmbedding, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        #
        # self.token_bucket_size = 42
        # self.max_position = 18
        #
        # token_num_rel_dis = 2 * self.token_bucket_size - 1
        # token_rp_bucket = make_token_bucket_position(self.token_bucket_size, self.max_position)
        # self.token_rel_pos_table_list = nn.ModuleList(
        #     [Embedding(token_num_rel_dis, 8, zero_init=True) for _ in range(6)]
        # ).to(device)
        #
        # self.register_buffer("token_rp_bucket", token_rp_bucket)

    # 根据输入张量 x，获取相对位置偏置的嵌入值
    # def get_rel_pos_bias(self, x, idx):
    #     seq_len = x.size(1)
    #     rp_bucket = self.token_rp_bucket[:seq_len, :seq_len]
    #     # 获取相对位置（rp_bucket提供相对位置索引）嵌入权重的值，并将结果存储在 values 中
    #     # rp_bucket 是提供相对位置索引的矩阵， self.token_rel_pos_table_list[idx].weight 则是相对位置编码的权重矩阵
    #     # values 是一个形状为 (seq_len, seq_len, num_attention_heads) 的张量
    #     values = f.embedding(rp_bucket, self.token_rel_pos_table_list[idx].weight)
    #     values = values.unsqueeze(0).expand(x.size(0), -1, -1,
    #                                         -1)  # (batch_size, seq_len, seq_len, num_attention_heads)
    #     values = values.permute([0, 3, 1, 2])  # (batch_size, num_attention_heads, seq_len, seq_len)
    #     return values.contiguous()

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                # rel_pos_bias: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        # for idx, layer in enumerate(self.layers):
        #     self_rel_pos_bias = self.get_rel_pos_bias(tgt, idx)

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos,
                           # rel_pos_bias=self_rel_pos_bias,
                           query_pos=query_pos,
                           need_weights=False
                           # need_weights=True
                           )
        output = self.norm(output)

        return output


# class LearnedPositionalEncoding_y_word(nn.Module):
#     """1D Position embedding with learnable embedding weights.
#
#     Args:
#         num_feature (int): The feature dimension for each position.
#         num_embedding (int, optional): The dictionary size of embeddings.
#             Default 5.
#     """
#
#     def __init__(self,
#                  num_embedding=15,
#                  num_feature=256,
#                  padding_idx=-1,
#                  ):
#         super(LearnedPositionalEncoding_y_word, self).__init__()
#         self.num_feature = num_feature
#
#         self.num_embedding = num_embedding
#         self.embedding = nn.Embedding(
#             num_embedding, num_feature, padding_idx=padding_idx if padding_idx >= 0 else None)
#
#     def forward(self, seq_in_embeds):
#         """
#         Args:
#             seq_in_embeds (tensor): [bs, 5/num_ray*2+1, d_model].
#
#         Returns:
#             seq_in_pos_embeds (tensor): [bs, 5/num_ray*2+1, d_model].
#         """
#         seq_len = seq_in_embeds.size(1)
#         position = torch.arange(seq_len, dtype=torch.long,
#                                 device=seq_in_embeds.device)
#         position = position.unsqueeze(0).expand(seq_in_embeds.size()[:2])
#         return self.embedding(position)


@TRANSFORMER.register_module()
class AutoRegressiveTransformer(BaseModule):
    """Implements the Auto-regressive transformer.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder, decoder):
        super(AutoRegressiveTransformer, self).__init__(init_cfg=None)
        self.d_model = decoder['layer']['d_model']
        self.encoder = TransformerEncoderWithPositionEmbedding(
            TransformerEncoderLayerWithPositionEmbedding(
                **encoder.pop('layer')),
            **encoder)
        self.decoder = TransformerDecoderWithPositionEmbedding(
            TransformerDecoderLayerWithPositionEmbedding(
                **decoder.pop('layer')),
            **decoder,
            norm=nn.LayerNorm(self.d_model))

        self.ED_fusion = DiscriminativeFeatEnc()
        # self.y_word_positional_encoding = LearnedPositionalEncoding_y_word()

        self.linear_y_word = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(self.d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def _init_layers(self,
                     in_ch,
                     vocab_size,
                     x_positional_encoding,
                     seq_positional_encoding):

        self.x_positional_encoding = build_positional_encoding(
            x_positional_encoding)

        self.seq_positional_encoding = build_positional_encoding(
            seq_positional_encoding)

        self.query_embedding = nn.Embedding(vocab_size, self.d_model)

        self.input_proj = ConvModule(
            in_channels=in_ch,
            out_channels=self.d_model,
            kernel_size=1,
            bias=True,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32)
        )

    def tri_mask(self, length):
        mask = (torch.triu(torch.ones(length, length))
                == 1).float().transpose(0, 1)
        mask.masked_fill_(mask == 0, float('-inf'))
        mask.masked_fill_(mask == 1, float(0.))
        return mask

    def x_mask_pos_enc(self, x, img_metas):
        batch_size = x.size(0)

        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        # CAUTION: do not support random flipping
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            x_mask[img_id, :img_h, :img_w] = 0

        x_mask = f.interpolate(x_mask.unsqueeze(
            1), size=x.size()[-2:]).to(torch.bool).squeeze(1)

        x_pos_embeds = self.x_positional_encoding(x_mask)

        x_mask = x_mask.view(batch_size, -1)
        x_pos_embeds = x_pos_embeds.view(
            batch_size, self.d_model, -1).transpose(1, 2)

        return x_mask, x_pos_embeds

    def forward_encoder(self, x, x_mask, x_pos_embeds, y_word, y_mask):
        """Args:
            x (Tensor): [batch_size, c, h, w].

            x_mask (tensor): [batch_size, h, w], dtype is torch.bool, True means
                ignored positions.

            x_pos_embeds (tensor): [batch_size, d_model, h, w].

        Returns:
            memory (tensor): encoder outputs, [batch_size, h*w, d_model].
        """
        batch_size = x.size(0)
        x = self.input_proj(x)

        x = x.view(batch_size, self.d_model, -1).transpose(1, 2)

        y_word = self.linear_y_word(y_word)
        memory = self.encoder(x,
                              src_key_padding_mask=x_mask,
                              pos=x_pos_embeds)

        # y_word_pos_embeds = self.y_word_positional_encoding(y_word)

        memory = self.ED_fusion(memory, x_mask, x_pos_embeds, y_word, y_mask)

        return memory

    def forward_decoder(self,
                        seq_in_embeds,
                        memory,
                        x_pos_embeds,
                        x_mask):
        seq_in_pos_embeds = self.seq_positional_encoding(seq_in_embeds)

        seq_in_mask = self.tri_mask(
            seq_in_embeds.size(1)).to(seq_in_embeds.device)

        tgt = self.decoder(seq_in_embeds, memory,
                           pos=x_pos_embeds,
                           query_pos=seq_in_pos_embeds,
                           memory_key_padding_mask=x_mask,
                           tgt_mask=seq_in_mask)
        return tgt
        # tgt, attn_weights = self.decoder(seq_in_embeds, memory,
        #                                  pos=x_pos_embeds,
        #                                  query_pos=seq_in_pos_embeds,
        #                                  memory_key_padding_mask=x_mask,
        #                                  tgt_mask=seq_in_mask)
        #
        # return tgt, attn_weights

# class MLPBlock(nn.Module):
#     def __init__(
#             self,
#             embedding_dim: int,
#             mlp_dim: int,
#             act: Type[nn.Module] = nn.GELU,
#     ) -> None:
#         super().__init__()
#         self.lin1 = nn.Linear(embedding_dim, mlp_dim)
#         self.lin2 = nn.Linear(mlp_dim, embedding_dim)
#         self.act = act()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.lin2(self.act(self.lin1(x)))
