import math

import torch
import random
import torch.nn as nn
import torch.nn.functional as f

from mmdet.models.losses import CrossEntropyLoss
from mmdet.models.utils import build_transformer

from seqtr.models import HEADS
from seqtr.core.layers import LinearModule
from seqtr.core.losses import LabelSmoothCrossEntropyLoss, L1Loss, HuberLoss
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "0, 1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enable = True


@HEADS.register_module()
class SeqHead(nn.Module):
    def __init__(self,
                 in_ch=1024,
                 num_bin=64,
                 multi_task=False,
                 shuffle_fraction=-1,
                 mapping="relative",
                 top_p=-1,
                 num_ray=12,
                 det_coord=[0],
                 det_coord_weight=1.5,
                 loss=dict(
                     type="L1Loss",
                     neg_factor=0.1
                 ),
                 # 对 decoder 输出的 token 进行类别预测，预测（x, y），记得后续添加 sigmoid，将点坐标限制到（0，1）
                 predictor=dict(
                     num_fcs=3, in_chs=[256, 256, 256], out_chs=[256, 256, 2],
                     fc=[
                         dict(
                             linear=dict(type='Linear', bias=True),
                             act=dict(type='ReLU', inplace=True),
                             drop=None),
                         dict(
                             linear=dict(type='Linear', bias=True),
                             act=dict(type='ReLU', inplace=True),
                             drop=None),
                         dict(
                             linear=dict(type='Linear', bias=True),
                             act=None,
                             drop=None)
                     ]
                 ),
                 transformer=dict(
                     type='AutoRegressiveTransformer',
                     encoder=dict(
                         num_layers=6,
                         layer=dict(
                             d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu',
                             batch_first=True)),
                     decoder=dict(
                         num_layers=6,
                         layer=dict(
                             d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu',
                             batch_first=True),
                     )),
                 x_positional_encoding=dict(
                     type='SinePositionalEncoding2D',
                     num_feature=128,
                     normalize=True),
                 seq_positional_encoding=dict(
                     type='LearnedPositionalEncoding1D',
                     num_embedding=3,
                     num_feature=256)
                 ):
        super(SeqHead, self).__init__()
        self.num_bin = num_bin
        self.multi_task = multi_task
        self.shuffle_fraction = shuffle_fraction
        assert mapping in ["relative", "absolute"]
        self.mapping = mapping
        self.top_p = top_p
        self.num_ray = num_ray  # 多边形取点
        self.det_coord = det_coord
        self.det_coord_weight = det_coord_weight

        self.transformer = build_transformer(transformer)
        self.d_model = self.transformer.d_model

        self._init_layers(in_ch,
                          predictor,
                          multi_task,
                          x_positional_encoding,
                          seq_positional_encoding)

        loss_type = loss.pop('type')
        if loss_type == "CrossEntropyLoss":
            self.loss_ce = CrossEntropyLoss()
        elif loss_type == "LabelSmoothCrossEntropyLoss":
            self.loss_ce = LabelSmoothCrossEntropyLoss(
                neg_factor=loss.pop('neg_factor', 0.1))
        elif loss_type == "L1Loss":
            self.loss_ce = L1Loss()
        elif loss_type == "HuberLoss":
            self.loss_ce = HuberLoss()

    def _init_layers(self,
                     in_ch,
                     predictor_cfg,
                     multi_task,
                     x_positional_encoding,
                     seq_positional_encoding):
        num_fcs = predictor_cfg.pop('num_fcs')
        in_chs, out_chs = predictor_cfg.pop(
            'in_chs'), predictor_cfg.pop('out_chs')
        fc_cfg = predictor_cfg.pop('fc')
        assert num_fcs == len(fc_cfg) == len(in_chs) == len(out_chs)
        predictor = []
        for i in range(num_fcs):
            _cfg = fc_cfg[i]
            _cfg['linear']['in_features'] = in_chs[i]
            _cfg['linear']['out_features'] = out_chs[i]
            predictor.append(LinearModule(**_cfg))

        self.vocab_size = self.num_bin * self.num_bin + 1
        self.end = 2
        self.predictor = nn.Sequential(*predictor)

        if multi_task:
            # bbox_token, x1, y1, x2, y2, mask_token, x1, y1, ..., xN, yN
            self.task_embedding = nn.Embedding(2, self.d_model)

        self.transformer._init_layers(in_ch,
                                      self.vocab_size,  # 4097 = 64*64+1
                                      x_positional_encoding,
                                      seq_positional_encoding)

    # 浮点数形式的箱化
    def quantize(self, seq, img_metas):
        if self.mapping == "relative":
            num_pts = seq.size(1) // 2
            # 从 img_metas 中提取了 pad_shape 的信息，并创建了一个张量 norm_factor，用于归一化序列值。
            norm_factor = [img_meta['pad_shape'][:2][::-1]  # 得到填充后的 h 、 w 并进行反转得到 w, h
                           for img_meta in img_metas]
            norm_factor = seq.new_tensor(norm_factor)
            norm_factor = torch.cat(
                [norm_factor for _ in range(num_pts)], dim=1)  # [w, h, w, h ...]
            return (seq / norm_factor).float()
        elif self.mapping == "absolute":
            return (seq / 640.).float()

    # 预测头预测出来的是 [0, 1] 之间的值，恢复原图大小
    def dequantize(self, seq, scale_factor):
        if self.mapping == "relative":
            return seq * scale_factor
        elif self.mapping == "absolute":
            return seq * 640.

    # 选择 seq 序列中的一部分进行打乱，以增加数据的多样性或随机性。
    def shuffle_sequence(self, seq):
        batch_size, num_pts = seq.size(0), seq.size(1) // 2
        seq = seq.reshape(batch_size * num_pts, 2)
        # 根据给定的比例 self.shuffle_fraction 从 0 到 batch_size - 1 的范围内随机选择一定数量的索引。
        shuffle_idx = random.sample(
            range(batch_size), int(batch_size * self.shuffle_fraction))
        shuffle_idx = [idx * num_pts for idx in shuffle_idx]
        # 此行代码的作用是生成一个随机排列的索引（长度为 num_pts），该索引可以用于对序列进行洗牌操作，从而打乱序列中相应部分的顺序。
        perm = torch.randperm(num_pts, device=seq.device)
        for idx in shuffle_idx:
            s = idx
            e = s + num_pts
            seq[s:e, :] = seq[s:e, :][perm]
        seq = seq.reshape(batch_size, num_pts * 2)
        return seq

    def sequentialize(self,
                      img_metas,
                      gt_bbox=None,
                      gt_mask_vertices=None,
                      ):
        """Args:
            gt_bbox (list[tensor]): [4, ].

            gt_mask_vertices (tensor): [batch_size, 2 (in x, y order), num_ray].
        """
        with_bbox = gt_bbox is not None
        with_mask = gt_mask_vertices is not None
        assert with_bbox or with_mask
        batch_size = len(img_metas)

        # 原始坐标
        if with_bbox:
            seq_in_bbox = torch.vstack(gt_bbox)

        if with_mask:
            seq_in_mask = gt_mask_vertices.transpose(1, 2).reshape(batch_size,
                                                                   -1)  # [batch_size, 2 * num_ray]， 变成[x,y,x,y....]

        if with_bbox and with_mask:
            assert self.multi_task
            seq_in = torch.cat([seq_in_bbox, seq_in_mask], dim=-1)
        elif with_bbox:
            seq_in = seq_in_bbox
        elif with_mask:
            seq_in = seq_in_mask

        seq_in = self.quantize(seq_in, img_metas)  # (0, 1)之间的浮点数

        if with_mask:
            # 将序列中小于 0 的值（通常是标记为特殊值的）替换为 self.end。这个值可能表示序列中的结束标记或特殊情况（填充的值为 -1）。
            seq_in[seq_in < 0] = self.end
        # 将序列中不等于 self.end 的值, 限制在区间 [0, 1] 内。
        seq_in[seq_in != self.end].clamp_(min=0, max=1)

        if with_bbox and with_mask:
            # bbox_token, x1, y1, x2, y2, mask_token, x1, y1, ..., xN, yN
            # 只对轮廓点进行打乱
            if self.shuffle_fraction > 0.:
                seq_in[:, 4:] = self.shuffle_sequence(seq_in[:, 4:])
            seq_in_bbox, seq_in_mask = torch.split(
                seq_in, [4, seq_in.size(1) - 4], dim=1)

            seq_in_bbox = seq_in_bbox.reshape(batch_size, 2, 2)
            seq_in_mask = seq_in_mask.reshape(batch_size, -1, 2)
            # # 构建双线性插值
            # bin 化
            seq_in_bbox_bin = torch.tensor(seq_in_bbox) * (self.num_bin - 1)
            seq_in_mask_bin = torch.tensor(seq_in_mask) * (self.num_bin - 1)
            # 取整四周点
            quant_box11 = [[[math.floor(p[0]), math.floor(p[1])] for p in point_box] for point_box in
                           seq_in_bbox_bin]  # 左上
            quant_box21 = [[[math.ceil(p[0]), math.floor(p[1])] for p in point_box] for point_box in
                           seq_in_bbox_bin]  # 右上
            quant_box12 = [[[math.floor(p[0]), math.ceil(p[1])] for p in point_box] for point_box in
                           seq_in_bbox_bin]  # 左下
            quant_box22 = [[[math.ceil(p[0]), math.ceil(p[1])] for p in point_box] for point_box in
                           seq_in_bbox_bin]  # 右下
            # 规定每个点的 embedding 序号， 范围在（0， 64*64-1）
            quant_box11_merge = [[p[0] + p[1] * 64 for p in point] for point in quant_box11]
            quant_box21_merge = [[p[0] + p[1] * 64 for p in point] for point in quant_box21]
            quant_box12_merge = [[p[0] + p[1] * 64 for p in point] for point in quant_box12]
            quant_box22_merge = [[p[0] + p[1] * 64 for p in point] for point in quant_box22]
            # 四周点与原始点的相对距离
            quant_box_delta_x1 = [[p[0] - math.floor(p[0]) for p in point] for point in seq_in_bbox_bin]
            quant_box_delta_x1 = torch.tensor(quant_box_delta_x1)
            quant_box_delta_x2 = 1 - quant_box_delta_x1
            quant_box_delta_y1 = [[p[1] - math.floor(p[1]) for p in point] for point in seq_in_bbox_bin]
            quant_box_delta_y1 = torch.tensor(quant_box_delta_y1)
            quant_box_delta_y2 = 1 - quant_box_delta_y1

            region_coord11 = [[[math.floor(p[0]), math.floor(p[1])] for p in point] for point in seq_in_mask_bin]  # 左上
            region_coord21 = [[[math.ceil(p[0]), math.floor(p[1])] for p in point] for point in seq_in_mask_bin]  # 右上
            region_coord12 = [[[math.floor(p[0]), math.ceil(p[1])] for p in point] for point in seq_in_mask_bin]  # 左下
            region_coord22 = [[[math.ceil(p[0]), math.ceil(p[1])] for p in point] for point in seq_in_mask_bin]  # 右下
            region_coord11_merge = [[p[0] + p[1] * 64 for p in point] for point in region_coord11]
            region_coord21_merge = [[p[0] + p[1] * 64 for p in point] for point in region_coord21]
            region_coord12_merge = [[p[0] + p[1] * 64 for p in point] for point in region_coord12]
            region_coord22_merge = [[p[0] + p[1] * 64 for p in point] for point in region_coord22]

            region_coord_delta_x1 = [[p[0] - math.floor(p[0]) for p in point] for point in seq_in_mask_bin]
            region_coord_delta_x1 = torch.tensor(region_coord_delta_x1)
            region_coord_delta_x2 = 1 - region_coord_delta_x1
            region_coord_delta_y1 = [[p[1] - math.floor(p[1]) for p in point] for point in seq_in_mask_bin]
            region_coord_delta_y1 = torch.tensor(region_coord_delta_y1)
            region_coord_delta_y2 = 1 - region_coord_delta_y1

            # 构建目标序列， 检测结束之后也有 EOS token
            # targets = torch.cat([seq_in_bbox, seq_in_bbox.new_full((batch_size, 1, 2), self.end),
            # seq_in_mask, seq_in_mask.new_full((batch_size, 1, 2), self.end)], dim=1)
            targets = torch.cat([seq_in_bbox, seq_in_mask, seq_in_mask.new_full((batch_size, 1, 2), self.end)], dim=1)

            # 输入嵌入
            seq_in_embeds_bbox11 = self.transformer.query_embedding(
                torch.tensor(quant_box11_merge).to(device))
            seq_in_embeds_bbox12 = self.transformer.query_embedding(
                torch.tensor(quant_box12_merge).to(device))
            seq_in_embeds_bbox21 = self.transformer.query_embedding(
                torch.tensor(quant_box21_merge).to(device))
            seq_in_embeds_bbox22 = self.transformer.query_embedding(
                torch.tensor(quant_box22_merge).to(device))
            quant_box_delta_x1 = quant_box_delta_x1.unsqueeze(-1).repeat(1, 1, seq_in_embeds_bbox11.shape[-1]).to(
                device)
            quant_box_delta_x2 = quant_box_delta_x2.unsqueeze(-1).repeat(1, 1, seq_in_embeds_bbox12.shape[-1]).to(
                device)
            quant_box_delta_y1 = quant_box_delta_y1.unsqueeze(-1).repeat(1, 1, seq_in_embeds_bbox21.shape[-1]).to(
                device)
            quant_box_delta_y2 = quant_box_delta_y2.unsqueeze(-1).repeat(1, 1, seq_in_embeds_bbox22.shape[-1]).to(
                device)
            token_embedding_bbox = seq_in_embeds_bbox11 * quant_box_delta_x2 * quant_box_delta_y2 + \
                                   seq_in_embeds_bbox12 * quant_box_delta_x2 * quant_box_delta_y1 + \
                                   seq_in_embeds_bbox21 * quant_box_delta_x1 * quant_box_delta_y2 + \
                                   seq_in_embeds_bbox22 * quant_box_delta_x1 * quant_box_delta_y1

            seq_in_embeds_mask11 = self.transformer.query_embedding(
                torch.tensor(region_coord11_merge).to(device))
            seq_in_embeds_mask12 = self.transformer.query_embedding(
                torch.tensor(region_coord12_merge).to(device))
            seq_in_embeds_mask21 = self.transformer.query_embedding(
                torch.tensor(region_coord21_merge).to(device))
            seq_in_embeds_mask22 = self.transformer.query_embedding(
                torch.tensor(region_coord22_merge).to(device))

            region_coord_delta_x1 = region_coord_delta_x1.unsqueeze(-1).repeat(1, 1, seq_in_embeds_mask11.shape[-1]).to(
                device)
            region_coord_delta_x2 = region_coord_delta_x2.unsqueeze(-1).repeat(1, 1, seq_in_embeds_mask12.shape[-1]).to(
                device)
            region_coord_delta_y1 = region_coord_delta_y1.unsqueeze(-1).repeat(1, 1, seq_in_embeds_mask21.shape[-1]).to(
                device)
            region_coord_delta_y2 = region_coord_delta_y2.unsqueeze(-1).repeat(1, 1, seq_in_embeds_mask22.shape[-1]).to(
                device)

            token_embedding_mask = seq_in_embeds_mask11 * region_coord_delta_x2 * region_coord_delta_y2 + \
                                   seq_in_embeds_mask12 * region_coord_delta_x2 * region_coord_delta_y1 + \
                                   seq_in_embeds_mask21 * region_coord_delta_x1 * region_coord_delta_y2 + \
                                   seq_in_embeds_mask22 * region_coord_delta_x1 * region_coord_delta_y1

            # 给任务 token 赋予权重进行嵌入
            task_bbox = self.task_embedding.weight[0].unsqueeze(
                0).unsqueeze(0).expand(batch_size, -1, -1)
            task_mask = self.task_embedding.weight[1].unsqueeze(
                0).unsqueeze(0).expand(batch_size, -1, -1)

            # seq_in_embeds = torch.cat(
            # [task_bbox, token_embedding_bbox, task_mask, token_embedding_mask], dim=1)

            seq_in_embeds = torch.cat(
                [task_bbox, token_embedding_bbox, token_embedding_mask], dim=1)
            return seq_in_embeds, targets

        else:
            if with_mask and self.shuffle_fraction > 0.:
                seq_in = self.shuffle_sequence(seq_in)

            seq_in = torch.tensor(seq_in.reshape(batch_size, -1, 2))
            seq_in_bin = seq_in * (self.num_bin - 1)

            quant11 = [[[math.floor(p[0]), math.floor(p[1])] for p in point] for point in seq_in_bin]  # 左上
            quant21 = [[[math.ceil(p[0]), math.floor(p[1])] for p in point] for point in seq_in_bin]  # 右上
            quant12 = [[[math.floor(p[0]), math.ceil(p[1])] for p in point] for point in seq_in_bin]  # 左下
            quant22 = [[[math.ceil(p[0]), math.ceil(p[1])] for p in point] for point in seq_in_bin]  # 右下

            quant11_merge = torch.tensor([[p[0] + p[1] * 64 for p in point] for point in quant11])
            quant21_merge = torch.tensor([[p[0] + p[1] * 64 for p in point] for point in quant21])
            quant12_merge = torch.tensor([[p[0] + p[1] * 64 for p in point] for point in quant12])
            quant22_merge = torch.tensor([[p[0] + p[1] * 64 for p in point] for point in quant22])

            delta_x1 = [[p[0] - math.floor(p[0]) for p in point] for point in seq_in_bin]
            delta_x1 = torch.tensor(delta_x1)
            delta_x2 = 1 - delta_x1
            delta_y1 = [[p[1] - math.floor(p[1]) for p in point] for point in seq_in_bin]
            delta_y1 = torch.tensor(delta_y1)
            delta_y2 = 1 - delta_y1

            # 对四个角点的坐标进行嵌入
            seq_in_embeds11 = self.transformer.query_embedding(
                quant11_merge.to(device))
            seq_in_embeds12 = self.transformer.query_embedding(
                quant12_merge.to(device))
            seq_in_embeds21 = self.transformer.query_embedding(
                quant21_merge.to(device))
            seq_in_embeds22 = self.transformer.query_embedding(
                quant22_merge.to(device))

            delta_x1 = delta_x1.unsqueeze(-1).repeat(1, 1, seq_in_embeds11.shape[-1]).to(device)
            delta_x2 = delta_x2.unsqueeze(-1).repeat(1, 1, seq_in_embeds12.shape[-1]).to(device)
            delta_y1 = delta_y1.unsqueeze(-1).repeat(1, 1, seq_in_embeds21.shape[-1]).to(device)
            delta_y2 = delta_y2.unsqueeze(-1).repeat(1, 1, seq_in_embeds22.shape[-1]).to(device)

            token_embedding = seq_in_embeds11 * delta_x2 * delta_y2 + \
                              seq_in_embeds12 * delta_x2 * delta_y1 + \
                              seq_in_embeds21 * delta_x1 * delta_y2 + \
                              seq_in_embeds22 * delta_x1 * delta_y1

            targets = torch.cat([seq_in, seq_in.new_full((batch_size, 1, 2), self.end)], dim=1)  # （0，1）之间的数

            seq_in_embeds = torch.cat(
                [token_embedding.new_zeros((batch_size, 1, self.d_model)), token_embedding], dim=1)
            return seq_in_embeds, targets

    def forward_train(self,
                      x_mm,
                      img_metas,
                      y_word, y_mask,
                      gt_bbox=None,
                      gt_mask_vertices=None,
                      ):
        """Args:
            x_mm (tensor): [batch_size, c, h, w].

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `seqtr/datasets/pipelines/formatting.py:CollectData`.

            gt_bbox (list[tensor]): [4, ], [tl_x, tl_y, br_x, br_y] format,
                and the coordinates are in 'img_shape' scale.

            gt_mask_vertices (list[tensor]): [batch_size, 2, num_ray], padded values are -1,
                the coordinates are in 'pad_shape' scale.
        """
        with_bbox = gt_bbox is not None
        with_mask = gt_mask_vertices is not None

        x_mask, x_pos_embeds = self.transformer.x_mask_pos_enc(x_mm, img_metas)

        memory = self.transformer.forward_encoder(x_mm, x_mask, x_pos_embeds, y_word, y_mask)

        seq_in_embeds, targets = self.sequentialize(
            img_metas,
            gt_bbox=gt_bbox,
            gt_mask_vertices=gt_mask_vertices)

        # 预测的 Embedding 值
        logits = self.transformer.forward_decoder(
            seq_in_embeds, memory, x_pos_embeds, x_mask)

        logits = self.predictor(logits)
        logits = f.sigmoid(logits)  # 将预测点约束回（0， 1）

        # logits, targets 的形状为 [64, 3, 2]
        loss_ce = self.loss(
            logits, targets, with_bbox=with_bbox, with_mask=with_mask)

        # training statistics
        with torch.no_grad():
            if with_mask and with_bbox:
                logits_bbox = logits[:, :2, :]
                seq_out_bbox = logits_bbox
                # logits_mask = logits[:, 3:-1, :]
                logits_mask = logits[:, 2:-1, :]
                seq_out_mask = logits_mask
                return dict(loss_multi_task=loss_ce), \
                       dict(seq_out_bbox=seq_out_bbox.detach(),
                            seq_out_mask=seq_out_mask.detach())
            else:
                logits = logits[:, :-1, :]
                seq_out = logits
                if with_bbox:
                    return dict(loss_det=loss_ce), \
                           dict(seq_out_bbox=seq_out.detach())
                elif with_mask:
                    return dict(loss_mask=loss_ce), \
                           dict(seq_out_mask=seq_out.detach())

    # 设置不同标记的不同损失，并计算 L1 损失函数
    def loss(self, logits, targets, with_bbox=False, with_mask=False):
        """Args:
            logits (tensor): [batch_size, 1+2 or 1+num_ray, 2].

            target (tensor): [batch_size, 1+2 or 1+num_ray, 2].
        """
        batch_size, num_token = logits.size()[:2]

        if with_bbox and with_mask:
            weight = logits.new_ones((batch_size, num_token))
            overlay = [self.det_coord_weight if i %
                                                3 in self.det_coord else 1. for i in range(3)]
            overlay = torch.tensor(
                overlay, device=weight.device, dtype=weight.dtype)
            for elem in weight:
                elem[:3] = overlay
            weight = weight.reshape(-1)
        elif with_bbox:
            weight = logits.new_tensor([self.det_coord_weight if i % 3 in self.det_coord else 1.
                                        for i in range(batch_size * num_token)])  # [Task] token 的权重设置为 1.5
        elif with_mask:
            weight = logits.new_tensor(
                [1. for _ in range(batch_size * num_token)])  # len(weight) = 832
            # print(len(weight))
            weight[targets.view(-1)[::2] == self.end] /= 10.  # [EOS] 的权重设置为 0.1

        loss_ce = self.loss_ce(logits, targets, weight=weight)
        return loss_ce

    def forward_test(self, x_mm, img_metas, y_word, y_mask, with_bbox=False, with_mask=False):
        x_mask, x_pos_embeds = self.transformer.x_mask_pos_enc(x_mm, img_metas)
        memory = self.transformer.forward_encoder(x_mm, x_mask, x_pos_embeds, y_word, y_mask)
        return self.generate_sequence(memory, x_mask, x_pos_embeds,
                                      # img_metas,
                                      with_bbox=with_bbox,
                                      with_mask=with_mask)

    def generate(self, seq_in_embeds, memory, x_pos_embeds, x_mask, decode_steps, with_mask,
                 # img_metas
                 ):
        seq_out = []
        for step in range(decode_steps):
            # out, attn_weights = self.transformer.forward_decoder(
            #     seq_in_embeds, memory, x_pos_embeds, x_mask)
            out = self.transformer.forward_decoder(
                seq_in_embeds, memory, x_pos_embeds, x_mask)

            # print(attn_weights.shape)
            #
            # img_meta = img_metas[31]
            #
            # attn_weight = attn_weights[31, step, :]
            # print(attn_weight.shape)
            # filename = img_meta['filename']
            # print(img_meta['expression'])
            #
            # img = mmcv.imread(filename).astype(numpy.uint8)
            #
            # imshow_attention(img, attn_weight)

            logits = out[:, -1, :]
            logits = self.predictor(logits)  # [64, 1, 2]
            if self.multi_task:
                if step < 2:
                    logits = logits[:, :]
            else:
                if not with_mask:
                    logits = logits[:, :]
            logits = f.sigmoid(logits)
            logits = logits * (self.num_bin - 1)
            output11 = [[math.floor(p[0]), math.floor(p[1])] for p in logits]  # 左上
            output21 = [[math.ceil(p[0]), math.floor(p[1])] for p in logits]  # 右上
            output12 = [[math.floor(p[0]), math.ceil(p[1])] for p in logits]  # 左下
            output22 = [[math.ceil(p[0]), math.ceil(p[1])] for p in logits]  # 右下

            output11_merge = [[p[0] + p[1] * 64] for p in output11]  # [64, 1]
            output21_merge = [[p[0] + p[1] * 64] for p in output21]
            output12_merge = [[p[0] + p[1] * 64] for p in output12]
            output22_merge = [[p[0] + p[1] * 64] for p in output22]

            delta_x1 = [[p[0] - math.floor(p[0])] for p in logits]  # [64, 1]
            delta_x1 = torch.tensor(delta_x1).to(device)
            delta_x2 = (1 - delta_x1).to(device)
            delta_y1 = [[p[1] - math.floor(p[1])] for p in logits]
            delta_y1 = torch.tensor(delta_y1).to(device)
            delta_y2 = (1 - delta_y1).to(device)

            output_embeds11 = self.transformer.query_embedding(  # [64, 1, 256]
                torch.tensor(output11_merge).to(device))
            output_embeds12 = self.transformer.query_embedding(
                torch.tensor(output12_merge).to(device))
            output_embeds21 = self.transformer.query_embedding(
                torch.tensor(output21_merge).to(device))
            output_embeds22 = self.transformer.query_embedding(
                torch.tensor(output22_merge).to(device))

            delta_x1 = delta_x1.unsqueeze(-1).repeat(1, 1, output_embeds11.shape[-1]).to(device)
            delta_x2 = delta_x2.unsqueeze(-1).repeat(1, 1, output_embeds12.shape[-1]).to(device)
            delta_y1 = delta_y1.unsqueeze(-1).repeat(1, 1, output_embeds21.shape[-1]).to(device)
            delta_y2 = delta_y2.unsqueeze(-1).repeat(1, 1, output_embeds22.shape[-1]).to(device)

            token_embedding = output_embeds11 * delta_x2 * delta_y2 + \
                              output_embeds12 * delta_x2 * delta_y1 + \
                              output_embeds21 * delta_x1 * delta_y2 + \
                              output_embeds22 * delta_x1 * delta_y1

            seq_in_embeds = torch.cat(
                [seq_in_embeds, token_embedding], dim=1)

            seq_out.append(logits / (self.num_bin - 1))

        seq_out = torch.cat(seq_out, dim=1)

        return seq_out  # [0, 1]

    def generate_sequence(self, memory, x_mask, x_pos_embeds,
                          # img_metas,
                          with_bbox=False, with_mask=False):
        """Args:
            memory (tensor): encoder's output, [batch_size, h*w, d_model].

            x_mask (tensor): [batch_size, h*w], dtype is torch.bool, True means
                ignored position.

            x_pos_embeds (tensor): [batch_size, h*w, d_model].
        """
        batch_size = memory.size(0)
        if with_bbox and with_mask:
            task_bbox = self.task_embedding.weight[0].unsqueeze(
                0).unsqueeze(0).expand(batch_size, -1, -1)
            # 所有预测出来的序列
            seq_out_bbox = self.generate(
                task_bbox, memory, x_pos_embeds, x_mask, 2, False,
                # img_metas
                )  # [0, 1]
            seq_out_bbox = torch.tensor(seq_out_bbox.reshape(batch_size, 2, 2))
            task_mask = self.task_embedding.weight[1].unsqueeze(
                0).unsqueeze(0).expand(batch_size, -1, -1)

            seq_out_bbox_bin = seq_out_bbox * (self.num_bin - 1)

            quant_box11 = [[[math.floor(p[0]), math.floor(p[1])] for p in point_box] for point_box in
                           seq_out_bbox_bin]  # 左上
            quant_box21 = [[[math.ceil(p[0]), math.floor(p[1])] for p in point_box] for point_box in
                           seq_out_bbox_bin]  # 右上
            quant_box12 = [[[math.floor(p[0]), math.ceil(p[1])] for p in point_box] for point_box in
                           seq_out_bbox_bin]  # 左下
            quant_box22 = [[[math.ceil(p[0]), math.ceil(p[1])] for p in point_box] for point_box in
                           seq_out_bbox_bin]  # 右下
            quant_box11_merge = [[p[0] + p[1] * 64 for p in point] for point in quant_box11]
            quant_box21_merge = [[p[0] + p[1] * 64 for p in point] for point in quant_box21]
            quant_box12_merge = [[p[0] + p[1] * 64 for p in point] for point in quant_box12]
            quant_box22_merge = [[p[0] + p[1] * 64 for p in point] for point in quant_box22]
            quant_box_delta_x1 = [[p[0] - math.floor(p[0]) for p in point] for point in seq_out_bbox_bin]
            quant_box_delta_x1 = torch.tensor(quant_box_delta_x1)
            quant_box_delta_x2 = 1 - quant_box_delta_x1
            quant_box_delta_y1 = [[p[1] - math.floor(p[1]) for p in point] for point in seq_out_bbox_bin]
            quant_box_delta_y1 = torch.tensor(quant_box_delta_y1)
            quant_box_delta_y2 = 1 - quant_box_delta_y1
            seq_in_embeds_bbox11 = self.transformer.query_embedding(
                torch.tensor(quant_box11_merge).to(device))
            seq_in_embeds_bbox12 = self.transformer.query_embedding(
                torch.tensor(quant_box12_merge).to(device))
            seq_in_embeds_bbox21 = self.transformer.query_embedding(
                torch.tensor(quant_box21_merge).to(device))
            seq_in_embeds_bbox22 = self.transformer.query_embedding(
                torch.tensor(quant_box22_merge).to(device))

            quant_box_delta_x1 = quant_box_delta_x1.unsqueeze(-1).repeat(1, 1, seq_in_embeds_bbox11.shape[-1]).to(
                device)
            quant_box_delta_x2 = quant_box_delta_x2.unsqueeze(-1).repeat(1, 1, seq_in_embeds_bbox12.shape[-1]).to(
                device)
            quant_box_delta_y1 = quant_box_delta_y1.unsqueeze(-1).repeat(1, 1, seq_in_embeds_bbox21.shape[-1]).to(
                device)
            quant_box_delta_y2 = quant_box_delta_y2.unsqueeze(-1).repeat(1, 1, seq_in_embeds_bbox22.shape[-1]).to(
                device)

            token_embedding_bbox = seq_in_embeds_bbox11 * quant_box_delta_x2 * quant_box_delta_y2 + \
                                   seq_in_embeds_bbox12 * quant_box_delta_x2 * quant_box_delta_y1 + \
                                   seq_in_embeds_bbox21 * quant_box_delta_x1 * quant_box_delta_y2 + \
                                   seq_in_embeds_bbox22 * quant_box_delta_x1 * quant_box_delta_y1
            # seq_in_embeds_mask = torch.cat(
            # [task_bbox, token_embedding_bbox, task_mask], dim=1)

            seq_in_embeds_mask = torch.cat(
                [task_bbox, token_embedding_bbox], dim=1)
            seq_out_mask = self.generate(
                seq_in_embeds_mask, memory, x_pos_embeds, x_mask, self.num_ray, True,
                # img_metas
                )
            return dict(seq_out_bbox=seq_out_bbox,
                        seq_out_mask=seq_out_mask)
        else:
            seq_in_embeds = memory.new_zeros((batch_size, 1, self.d_model))
            if with_mask:
                decode_steps = self.num_ray
            elif with_bbox:
                decode_steps = 2
            seq_out = self.generate(
                seq_in_embeds, memory, x_pos_embeds, x_mask, decode_steps, with_mask,
                # img_metas
            )

            if with_bbox:
                return dict(seq_out_bbox=seq_out)
            elif with_mask:
                return dict(seq_out_mask=seq_out)


# import numpy
# import cv2
# from matplotlib import pyplot as plt
#
#
# def imshow_attention(img, attn_weights):
#     img = numpy.ascontiguousarray(img)[:, :, ::-1]
#     h, w = img.shape[:2]
#     attn_weights = attn_weights.reshape(20, 20)
#     attn_weights = attn_weights.cpu().numpy()
#
#     plt.clf()
#     plt.axis('off')
#     plt.imshow(img, alpha=0.7)
#     attn_mask = cv2.resize(attn_weights, (w, h))
#     attn_mask = (attn_mask * 255).astype(numpy.uint8)
#     plt.imshow(attn_mask, alpha=0.3,
#                interpolation="bilinear", cmap="jet")
#     plt.show()


# import torch
# import random
# import torch.nn as nn
# import torch.nn.functional as f
#
# from mmdet.models.losses import CrossEntropyLoss
# from mmdet.models.utils import build_transformer
#
# from seqtr.models import HEADS
# from seqtr.core.layers import LinearModule
# from seqtr.core.losses import LabelSmoothCrossEntropyLoss
#
#
# @HEADS.register_module()
# class SeqHead(nn.Module):
#     def __init__(self,
#                  in_ch=1024,
#                  num_bin=1000,
#                  multi_task=False,
#                  shuffle_fraction=-1,
#                  mapping="relative",
#                  top_p=-1,
#                  num_ray=18,
#                  det_coord=[0],
#                  det_coord_weight=1.5,
#                  loss=dict(
#                      type="LabelSmoothCrossEntropyLoss",
#                      neg_factor=0.1
#                  ),
#                  predictor=dict(
#                      num_fcs=3, in_chs=[256, 256, 256], out_chs=[256, 256, 1001],
#                      fc=[
#                          dict(
#                              linear=dict(type='Linear', bias=True),
#                              act=dict(type='ReLU', inplace=True),
#                              drop=None),
#                          dict(
#                              linear=dict(type='Linear', bias=True),
#                              act=dict(type='ReLU', inplace=True),
#                              drop=None),
#                          dict(
#                              linear=dict(type='Linear', bias=True),
#                              act=None,
#                              drop=None)
#                      ]
#                  ),
#                  transformer=dict(
#                      type='AutoRegressiveTransformer',
#                      encoder=dict(
#                          num_layers=6,
#                          layer=dict(
#                              d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu', batch_first=True)),
#                      decoder=dict(
#                          num_layers=3,
#                          layer=dict(
#                              d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu', batch_first=True),
#                      )),
#                  x_positional_encoding=dict(
#                      type='SinePositionalEncoding2D',
#                      num_feature=128,
#                      normalize=True),
#                  seq_positional_encoding=dict(
#                      type='LearnedPositionalEncoding1D',
#                      num_embedding=5,
#                      num_feature=256)
#                  ):
#         super(SeqHead, self).__init__()
#         self.num_bin = num_bin
#         self.multi_task = multi_task
#         self.shuffle_fraction = shuffle_fraction
#         assert mapping in ["relative", "absolute"]
#         self.mapping = mapping
#         self.top_p = top_p
#         self.num_ray = num_ray
#         self.det_coord = det_coord
#         self.det_coord_weight = det_coord_weight
#
#         self.transformer = build_transformer(transformer)
#         self.d_model = self.transformer.d_model
#
#         self._init_layers(in_ch,
#                           predictor,
#                           multi_task,
#                           x_positional_encoding,
#                           seq_positional_encoding)
#
#         loss_type = loss.pop('type')
#         if loss_type == "CrossEntropyLoss":
#             self.loss_ce = CrossEntropyLoss()
#         elif loss_type == "LabelSmoothCrossEntropyLoss":
#             self.loss_ce = LabelSmoothCrossEntropyLoss(
#                 neg_factor=loss.pop('neg_factor', 0.1))
#
#     def _init_layers(self,
#                      in_ch,
#                      predictor_cfg,
#                      multi_task,
#                      x_positional_encoding,
#                      seq_positional_encoding):
#         num_fcs = predictor_cfg.pop('num_fcs')
#         in_chs, out_chs = predictor_cfg.pop(
#             'in_chs'), predictor_cfg.pop('out_chs')
#         fc_cfg = predictor_cfg.pop('fc')
#         assert num_fcs == len(fc_cfg) == len(in_chs) == len(out_chs)
#         predictor = []
#         for i in range(num_fcs):
#             _cfg = fc_cfg[i]
#             _cfg['linear']['in_features'] = in_chs[i]
#             _cfg['linear']['out_features'] = out_chs[i]
#             predictor.append(LinearModule(**_cfg))
#             if i == num_fcs - 1:
#                 self.vocab_size = out_chs[i]
#         assert self.vocab_size == self.num_bin + 1
#         self.end = self.vocab_size - 1
#         self.predictor = nn.Sequential(*predictor)
#
#         if multi_task:
#             # bbox_token, x1, y1, x2, y2, mask_token, x1, y1, ..., xN, yN
#             self.task_embedding = nn.Embedding(2, self.d_model)
#
#         self.transformer._init_layers(in_ch,
#                                       self.vocab_size,
#                                       x_positional_encoding,
#                                       seq_positional_encoding)
#
#     def quantize(self, seq, img_metas):
#         if self.mapping == "relative":
#             num_pts = seq.size(1) // 2
#             norm_factor = [img_meta['pad_shape'][:2][::-1]
#                            for img_meta in img_metas]
#             norm_factor = seq.new_tensor(norm_factor)
#             norm_factor = torch.cat(
#                 [norm_factor for _ in range(num_pts)], dim=1)
#             return (seq / norm_factor * self.num_bin).long()
#         elif self.mapping == "absolute":
#             return (seq / 640. * self.num_bin).long()
#
#     def dequantize(self, seq, scale_factor):
#         if self.mapping == "relative":
#             return seq * scale_factor / self.num_bin
#         elif self.mapping == "absolute":
#             return seq * 640. / self.num_bin
#
#     def shuffle_sequence(self, seq):
#         batch_size, num_pts = seq.size(0), seq.size(1) // 2
#         seq = seq.reshape(batch_size * num_pts, 2)
#         shuffle_idx = random.sample(
#             range(batch_size), int(batch_size * self.shuffle_fraction))
#         shuffle_idx = [idx * num_pts for idx in shuffle_idx]
#         perm = torch.randperm(num_pts, device=seq.device)
#         for idx in shuffle_idx:
#             s = idx
#             e = s + num_pts
#             seq[s:e, :] = seq[s:e, :][perm]
#         seq = seq.reshape(batch_size, num_pts * 2)
#         return seq
#
#     def sequentialize(self,
#                       img_metas,
#                       gt_bbox=None,
#                       gt_mask_vertices=None,
#                       ):
#         """Args:
#             gt_bbox (list[tensor]): [4, ].
#
#             gt_mask_vertices (tensor): [batch_size, 2 (in x, y order), num_ray].
#         """
#         with_bbox = gt_bbox is not None
#         with_mask = gt_mask_vertices is not None
#         assert with_bbox or with_mask
#         batch_size = len(img_metas)
#
#         if with_bbox:
#             seq_in_bbox = torch.vstack(gt_bbox)
#
#         if with_mask:
#             seq_in_mask = gt_mask_vertices.transpose(1, 2).reshape(batch_size, -1)
#
#         if with_bbox and with_mask:
#             assert self.multi_task
#             seq_in = torch.cat([seq_in_bbox, seq_in_mask], dim=-1)
#         elif with_bbox:
#             seq_in = seq_in_bbox
#         elif with_mask:
#             seq_in = seq_in_mask
#
#         seq_in = self.quantize(seq_in, img_metas)
#         if with_mask:
#             seq_in[seq_in < 0] = self.end
#         seq_in[seq_in != self.end].clamp_(min=0, max=self.num_bin-1)
#
#         if with_bbox and with_mask:
#             # bbox_token, x1, y1, x2, y2, mask_token, x1, y1, ..., xN, yN
#             if self.shuffle_fraction > 0.:
#                 seq_in[:, 4:] = self.shuffle_sequence(seq_in[:, 4:])
#             seq_in_bbox, seq_in_mask = torch.split(
#                 seq_in, [4, seq_in.size(1)-4], dim=1)
#             targets = torch.cat([seq_in_bbox, seq_in_bbox.new_full(
#                 (batch_size, 1), self.end), seq_in_mask, seq_in_mask.new_full((batch_size, 1), self.end)], dim=-1)
#             seq_in_embeds_bbox = self.transformer.query_embedding(
#                 seq_in_bbox)
#             seq_in_embeds_mask = self.transformer.query_embedding(
#                 seq_in_mask)
#             task_bbox = self.task_embedding.weight[0].unsqueeze(
#                 0).unsqueeze(0).expand(batch_size, -1, -1)
#             task_mask = self.task_embedding.weight[1].unsqueeze(
#                 0).unsqueeze(0).expand(batch_size, -1, -1)
#             seq_in_embeds = torch.cat(
#                 [task_bbox, seq_in_embeds_bbox, task_mask, seq_in_embeds_mask], dim=1)
#             return seq_in_embeds, targets
#         else:
#             if with_mask and self.shuffle_fraction > 0.:
#                 seq_in = self.shuffle_sequence(seq_in)
#             seq_in_embeds = self.transformer.query_embedding(seq_in)
#             targets = torch.cat(
#                 [seq_in, seq_in.new_full((batch_size, 1), self.end)], dim=-1)
#             seq_in_embeds = torch.cat(
#                 [seq_in_embeds.new_zeros((batch_size, 1, self.d_model)), seq_in_embeds], dim=1)
#             return seq_in_embeds, targets
#
#     def forward_train(self,
#                       x_mm,
#                       img_metas,
#                       y_word, y_mask,
#                       gt_bbox=None,
#                       gt_mask_vertices=None,
#                       ):
#         """Args:
#             x_mm (tensor): [batch_size, c, h, w].
#
#             img_metas (list[dict]): list of image info dict where each dict
#                 has: 'img_shape', 'scale_factor', and may also contain
#                 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#                 For details on the values of these keys see
#                 `seqtr/datasets/pipelines/formatting.py:CollectData`.
#
#             gt_bbox (list[tensor]): [4, ], [tl_x, tl_y, br_x, br_y] format,
#                 and the coordinates are in 'img_shape' scale.
#
#             gt_mask_vertices (list[tensor]): [batch_size, 2, num_ray], padded values are -1,
#                 the coordinates are in 'pad_shape' scale.
#         """
#         with_bbox = gt_bbox is not None
#         with_mask = gt_mask_vertices is not None
#
#         x_mask, x_pos_embeds = self.transformer.x_mask_pos_enc(x_mm, img_metas)
#
#         memory = self.transformer.forward_encoder(x_mm, x_mask, x_pos_embeds, y_word, y_mask)
#
#         seq_in_embeds, targets = self.sequentialize(
#             img_metas,
#             gt_bbox=gt_bbox,
#             gt_mask_vertices=gt_mask_vertices)
#         logits = self.transformer.forward_decoder(
#             seq_in_embeds, memory, x_pos_embeds, x_mask)
#         logits = self.predictor(logits)
#         loss_ce = self.loss(
#             logits, targets, with_bbox=with_bbox, with_mask=with_mask)
#
#         # training statistics
#         with torch.no_grad():
#             if with_mask and with_bbox:
#                 logits_bbox = logits[:, :4, :-1]
#                 scores_bbox = f.softmax(logits_bbox, dim=-1)
#                 _, seq_out_bbox = scores_bbox.max(
#                     dim=-1, keepdim=False)
#                 logits_mask = logits[:, 5:, :]
#                 scores_mask = f.softmax(logits_mask, dim=-1)
#                 _, seq_out_mask = scores_mask.max(
#                     dim=-1, keepdim=False)
#                 return dict(loss_multi_task=loss_ce), \
#                     dict(seq_out_bbox=seq_out_bbox.detach(),
#                          seq_out_mask=seq_out_mask.detach())
#             else:
#                 if with_bbox:
#                     logits = logits[:, :-1, :-1]
#                 scores = f.softmax(logits, dim=-1)
#                 _, seq_out = scores.max(dim=-1, keepdim=False)
#
#                 if with_bbox:
#                     return dict(loss_det=loss_ce), \
#                         dict(seq_out_bbox=seq_out.detach())
#                 elif with_mask:
#                     return dict(loss_mask=loss_ce), \
#                         dict(seq_out_mask=seq_out.detach())
#
#     def loss(self, logits, targets, with_bbox=False, with_mask=False):
#         """Args:
#             logits (tensor): [batch_size, 1+4 or 1+2*num_ray, vocab_size].
#
#             target (tensor): [batch_size, 1+4 or 1+2*num_ray].
#         """
#         batch_size, num_token = logits.size()[:2]
#
#         if with_bbox and with_mask:
#             weight = logits.new_ones((batch_size, num_token))
#             overlay = [self.det_coord_weight if i %
#                        5 in self.det_coord else 1. for i in range(5)]
#             overlay = torch.tensor(
#                 overlay, device=weight.device, dtype=weight.dtype)
#             for elem in weight:
#                 elem[:5] = overlay
#             weight = weight.reshape(-1)
#         elif with_bbox:
#             weight = logits.new_tensor([self.det_coord_weight if i % 5 in self.det_coord else 1.
#                                         for i in range(batch_size * num_token)])
#         elif with_mask:
#             weight = logits.new_tensor(
#                 [1. for _ in range(batch_size * num_token)])
#             weight[targets.view(-1) == self.end] /= 10.
#
#         loss_ce = self.loss_ce(logits, targets, weight=weight)
#         return loss_ce
#
#     def forward_test(self, x_mm, img_metas, y_word, y_mask, with_bbox=False, with_mask=False):
#         x_mask, x_pos_embeds = self.transformer.x_mask_pos_enc(x_mm, img_metas)
#         memory = self.transformer.forward_encoder(x_mm, x_mask, x_pos_embeds, y_word, y_mask)
#         return self.generate_sequence(memory, x_mask, x_pos_embeds,
#                                       with_bbox=with_bbox,
#                                       with_mask=with_mask)
#
#     def generate(self, seq_in_embeds, memory, x_pos_embeds, x_mask, decode_steps, with_mask):
#         seq_out = []
#         for step in range(decode_steps):
#             out = self.transformer.forward_decoder(
#                 seq_in_embeds, memory, x_pos_embeds, x_mask)
#             logits = out[:, -1, :]
#             logits = self.predictor(logits)
#             if self.multi_task:
#                 if step < 4:
#                     logits = logits[:, :-1]
#             else:
#                 if not with_mask:
#                     logits = logits[:, :-1]
#             probs = f.softmax(logits, dim=-1)
#             if self.top_p > 0.:
#                 sorted_score, sorted_idx = torch.sort(
#                     probs, descending=True)
#                 cum_score = sorted_score.cumsum(dim=-1)
#                 sorted_idx_to_remove = cum_score > self.top_p
#                 sorted_idx_to_remove[...,
#                                      1:] = sorted_idx_to_remove[..., :-1].clone()
#                 sorted_idx_to_remove[..., 0] = 0
#                 idx_to_remove = sorted_idx_to_remove.scatter(
#                     1, sorted_idx, sorted_idx_to_remove)
#                 probs = probs.masked_fill(idx_to_remove, 0.)
#                 next_token = torch.multinomial(probs, num_samples=1)
#             else:
#                 _, next_token = probs.max(dim=-1, keepdim=True)
#
#             seq_in_embeds = torch.cat(
#                 [seq_in_embeds, self.transformer.query_embedding(next_token)], dim=1)
#
#             seq_out.append(next_token)
#
#         seq_out = torch.cat(seq_out, dim=-1)
#
#         return seq_out
#
#     def generate_sequence(self, memory, x_mask, x_pos_embeds, with_bbox=False, with_mask=False):
#         """Args:
#             memory (tensor): encoder's output, [batch_size, h*w, d_model].
#
#             x_mask (tensor): [batch_size, h*w], dtype is torch.bool, True means
#                 ignored position.
#
#             x_pos_embeds (tensor): [batch_size, h*w, d_model].
#         """
#         batch_size = memory.size(0)
#         if with_bbox and with_mask:
#             task_bbox = self.task_embedding.weight[0].unsqueeze(
#                 0).unsqueeze(0).expand(batch_size, -1, -1)
#             seq_out_bbox = self.generate(
#                 task_bbox, memory, x_pos_embeds, x_mask, 4, False)
#             task_mask = self.task_embedding.weight[1].unsqueeze(
#                 0).unsqueeze(0).expand(batch_size, -1, -1)
#             seq_in_embeds_box = self.transformer.query_embedding(
#                 seq_out_bbox)
#             seq_in_embeds_mask = torch.cat(
#                 [task_bbox, seq_in_embeds_box, task_mask], dim=1)
#             seq_out_mask = self.generate(
#                 seq_in_embeds_mask, memory, x_pos_embeds, x_mask, 2 * self.num_ray + 1, True)
#             return dict(seq_out_bbox=seq_out_bbox,
#                         seq_out_mask=seq_out_mask)
#         else:
#             seq_in_embeds = memory.new_zeros((batch_size, 1, self.d_model))
#             if with_mask:
#                 decode_steps = self.num_ray * 2 + 1
#             elif with_bbox:
#                 decode_steps = 4
#             seq_out = self.generate(
#                 seq_in_embeds, memory, x_pos_embeds, x_mask, decode_steps, with_mask)
#             if with_bbox:
#                 return dict(seq_out_bbox=seq_out)
#             elif with_mask:
#                 return dict(seq_out_mask=seq_out)
