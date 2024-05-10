import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.parameter import Parameter


class DiscriminativeFeatEnc(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.img2text_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1, batch_first=True)
        self.img2textcond_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1, batch_first=True)
        self.img2img_attn = MHAttentionRPE(d_model=256, h=8, dropout=0.1,
                                           pos_x_range=[-20, 20],
                                           pos_y_range=[-20, 20],
                                           pos_index_offset=20)
        self.text_proj = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1)
        self.img_proj = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1)

        # 计算视觉-语言验证得分
        self.tf_pow = 2
        self.tf_scale = Parameter(torch.Tensor([1.0]))  # 1.0
        self.tf_sigma = Parameter(torch.Tensor([0.5]))  # 0.5

        # 视觉-语言融合的两次归一化
        self.norm_text_cond_img = nn.LayerNorm(256)
        self.norm_img = nn.LayerNorm(256)

        # self.y_pos_ln = nn.LayerNorm(256)
        # self.y_pos_scaling = float(256 / 8 * 2) ** -0.5
        # self.y_pos_q_linear = nn.Linear(256, 256)
        # self.y_pos_k_linear = nn.Linear(256, 256)
        #
        # self.img_pos_ln = nn.LayerNorm(256)
        # self.img_pos_scaling = float(256 / 8 * 2) ** -0.5
        # self.img_pos_q_linear = nn.Linear(256, 256)
        # self.img_pos_k_linear = nn.Linear(256, 256)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, img_feat, img_key_padding_mask, img_pos,
                word_feat, word_key_padding_mask, word_pos=None):
        # visual-linguistic verification
        img_query = self.with_pos_embed(img_feat, img_pos)

        # # 绝对注意力偏置
        # batch = img_feat.size(0)
        # img_pos = self.img_pos_ln(img_pos)
        # img_pos_q = self.img_pos_q_linear(img_pos).view(
        #     img_feat.size(0), 400, 8, -1
        # ).transpose(1, 2) * self.img_pos_scaling  # [b, 8, 400, 32]
        #
        # word_pos = self.y_pos_ln(word_pos)
        # word_pos_k = self.y_pos_k_linear(word_pos).view(
        #     word_feat.size(0), word_feat.size(1), 8, -1
        # ).transpose(1, 2)
        #
        # self_attn_bias = torch.matmul(img_pos_q, word_pos_k.transpose(2, 3)).view(
        #     batch * 8, img_feat.size(1), -1)

        text_info = self.img2text_attn(
            query=img_query, key=self.with_pos_embed(word_feat, word_pos),
            value=word_feat, key_padding_mask=word_key_padding_mask,
        )[0]

        text_embed = self.text_proj(text_info)
        img_embed = self.img_proj(img_feat)
        verify_score = (f.normalize(img_embed, p=2, dim=-1) *
                        f.normalize(text_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True)
        verify_score = self.tf_scale * \
                       torch.exp(- (1 - verify_score).pow(self.tf_pow) \
                                 / (2 * self.tf_sigma ** 2))  # torch.Size([64, 400, 1])

        # plt.imshow(verify_score[0].reshape(20, 20).detach().cpu())
        # plt.show()

        # language-guided context encoder
        text_cond_info = self.img2textcond_attn(
            query=img_feat, key=self.with_pos_embed(word_feat, word_pos),
            value=word_feat, key_padding_mask=word_key_padding_mask,
        )[0]

        q = k = img_feat + text_cond_info

        # # 修改 Decoder 的 q, k, v
        # q = img_feat + text_cond_info
        # q = self.with_pos_embed(q, img_pos)
        # k = self.with_pos_embed(img_feat, img_pos)

        text_cond_img_ctx = self.img2img_attn(
            query=q, key=k, value=img_feat, key_padding_mask=img_key_padding_mask)[0].transpose(0, 1)

        # discriminative feature
        fuse_img_feat = (self.norm_img(img_feat) +
                         self.norm_text_cond_img(text_cond_img_ctx)) * verify_score

        return fuse_img_feat


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        if num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = []

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = f.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def position_embedding_sine(num_pos_feats=128, temperature=10000, normalize=True, scale=2 * math.pi,
                            x_range=[-20, 20], y_range=[-20, 20], device=None):
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")

    x_embed = torch.arange(x_range[0], x_range[1] + 1, device=device)  #
    y_embed = torch.arange(y_range[0], y_range[1] + 1, device=device)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[-1] + eps) * scale
        x_embed = x_embed / (x_embed[-1] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=-1).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=-1).flatten(1)
    return pos_x, pos_y


class MHAttentionRPE(nn.Module):
    """ With relative position embedding """

    def __init__(self, d_model, h, dropout=0.1, return_raw_attention=False,
                 pos_x_range=[-20, 20], pos_y_range=[-20, 20], pos_index_offset=20,
                 learnable_pos_embed=False):
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.scaling = float(self.d_k) ** -0.5
        self.return_raw_attention = return_raw_attention

        self.in_proj_weight = Parameter(torch.Tensor(3 * d_model, d_model))
        self.in_proj_bias = Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn = None
        # self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout
        self._reset_parameters()

        self.learnable_pos_embed = learnable_pos_embed
        if learnable_pos_embed:
            self.pos_x = nn.Embedding(pos_x_range[1] - pos_x_range[0] + 1, d_model // 2)  # d_model // 2 表示每个位置编码的维度
            self.pos_y = nn.Embedding(pos_y_range[1] - pos_y_range[0] + 1, d_model // 2)
        else:
            pos_x, pos_y = position_embedding_sine(d_model // 2, normalize=True,
                                                   x_range=pos_x_range, y_range=pos_y_range)
            self.register_buffer('pos_x', pos_x)  # [x_range, C]
            self.register_buffer('pos_y', pos_y)  # [y_range, C]

        self.pos_index_offset = pos_index_offset  # 用于指定位置编码向量中的索引偏移量

        # self.img_pos_ln = nn.LayerNorm(256)
        # self.pos_scaling = float(256 / 8 * 2) ** -0.5
        # self.pos_q_linear = nn.Linear(256, 256)
        # self.pos_k_linear = nn.Linear(256, 256)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        # 初始化输入、输出投影的偏置向量
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None):
        # def forward(self, query, key, value, key_padding_mask=None, pos):

        # batch = query.size(0)
        # pos = self.img_pos_ln(pos)
        # pos_q = self.pos_q_linear(pos).view(
        #     query.size(0), query.size(1), 8, -1
        # ).transpose(1, 2) * self.pos_scaling
        # pos_k = self.pos_k_linear(pos).view(
        #     key.size(0), key.size(1), 8, -1
        # ).transpose(1, 2)  # [b, 8, t, 32]
        # cross_attn_bias = torch.matmul(pos_q, pos_k.transpose(2, 3)).view(batch * 8, 400, -1)  # [b, 8, t, t]

        # 需要将输入维度变换成如下：
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        tgt_len, bs, dim = query.size()

        src_len, _, dim = key.size()

        # 提取 Q,K,V 各自的权重矩阵与偏置矩阵。
        weight_q, bias_q = self.in_proj_weight[0:dim], self.in_proj_bias[0:dim]
        weight_k, bias_k = self.in_proj_weight[dim:dim * 2], self.in_proj_bias[dim:dim * 2]
        weight_v, bias_v = self.in_proj_weight[dim * 2:], self.in_proj_bias[dim * 2:]

        # 将 Q,K,V 投射到新的表示空间
        q = query.matmul(weight_q.t()) + bias_q
        k = key.matmul(weight_k.t()) + bias_k
        v = value.matmul(weight_v.t()) + bias_v

        # 调整形状，以便用于多头注意力
        q = q.view(tgt_len, bs * self.h, -1).transpose(0, 1)  # [bs*h, tgt_len, dim//h]
        k = k.view(src_len, bs * self.h, -1).permute(1, 2, 0)  # [bs*h, dim//h, src_len], To calculate qTk (bmm)
        v = v.view(src_len, bs * self.h, -1).transpose(0, 1)

        # 对通道维度进行缩放
        q = q * self.scaling
        attn_weights = torch.bmm(q, k)  # [bs*h, tgt_len, src_len]

        # attn_weights += cross_attn_bias

        # compute the relative positions
        bs, HW = key_padding_mask.size()  # 填充掩码的形状
        # print(HW)
        assert (HW == 400) and (HW == tgt_len)

        img_mask = ~key_padding_mask.view(bs, 20, 20)  # ~ 表示取反

        # 进行累计求和，该位置（包含）及其之前有效像素的数量
        yy = img_mask.cumsum(1, dtype=torch.float32).view(bs, -1)  # [bs, HW],  1~20
        xx = img_mask.cumsum(2, dtype=torch.float32).view(bs, -1)  # [bs, HW],  1~20

        # yy[:, :, None] 将向量yy扩展为形状为(bs, HW, 1)， [:, None, :] 将其扩展为形状为(bs, 1, HW)
        # 得到的 diff_yy 是一个形状为 (bs, HW, HW) 的张量，表示每个像素位置之间的垂直方向上的差异值。
        # diff_xx 是一个形状为 (bs, HW, HW) 的张量，表示每个像素位置之间的水平方向上的差异值。
        diff_yy = yy[:, :, None] - yy[:, None, :]  # [bs, HW, HW]
        diff_xx = xx[:, :, None] - xx[:, None, :]  # [bs, HW, HW]

        # 分别表示垂直方向和水平方向上的位置嵌入权重。
        if self.learnable_pos_embed:
            k_posy = self.pos_y.weight.matmul(weight_k.t()[:dim // 2])  # [x_range, dim]
            k_posx = self.pos_x.weight.matmul(weight_k.t()[dim // 2:])  # [y_range, dim]
        else:
            k_posy = self.pos_y.matmul(weight_k.t()[:dim // 2])  # [x_range, dim]
            k_posx = self.pos_x.matmul(weight_k.t()[dim // 2:])  # [y_range, dim]

        # repeat(1, bs, 1, 1) 在第二个维度上重复 bs 次，将维度变为 [y_range, bs, self.h, dim//self.h]
        k_posy = k_posy.view(-1, 1, self.h, dim // self.h).repeat(1, bs, 1, 1). \
            reshape(-1, bs * self.h, dim // self.h).permute(1, 2, 0)  # [bs*h, dim//h, y_range]
        k_posx = k_posx.view(-1, 1, self.h, dim // self.h).repeat(1, bs, 1, 1). \
            reshape(-1, bs * self.h, dim // self.h).permute(1, 2, 0)  # [bs*h, dim//h, x_range]

        # 计算查询张量 q 与位置编码张量 k_posy 和 k_posx 之间的相对位置注意力权重。
        posy_attn_weights = torch.bmm(q, k_posy).view(bs, self.h, HW, -1)  # [bs, h, HW, y_range]
        posx_attn_weights = torch.bmm(q, k_posx).view(bs, self.h, HW, -1)  # [bs, h, HW, x_range]

        # 为了计算相对位置编码的索引，使得每个查询序列在注意力计算中都会根据其相对位置获取相应的位置编码。
        # 这里使用 None 来增加维度，然后利用 repeat 函数复制 self.h 份，以便与注意力头数对应。
        diff_yy_idx = diff_yy[:, None].repeat(1, self.h, 1, 1) + self.pos_index_offset
        diff_xx_idx = diff_xx[:, None].repeat(1, self.h, 1, 1) + self.pos_index_offset  # [bs, h, HW, HW]

        # 根据相对位置编码的索引 diff_yy_idx 和 diff_xx_idx 从 posy_attn_weights 和 posx_attn_weights 中获取相对位置的注意力权重。
        posy_attn_weights = torch.gather(posy_attn_weights, -1, diff_yy_idx.long())  # [bs, h, HW, HW]
        posx_attn_weights = torch.gather(posx_attn_weights, -1, diff_xx_idx.long())  # [bs, h, HW, HW]

        # 将在 y 方向和 x 方向上的相对位置注意力权重相加，将结果重塑为形状 [bs*self.h, HW, HW]
        pos_attn_weights = (posy_attn_weights + posx_attn_weights).view(bs * self.h, HW, -1)
        #  将原来的注意力权重 attn_weights 与相对位置注意力权重 pos_attn_weights 相加，得到最终的注意力权重 attn_weights。
        #  注意，相对位置注意力是在原始的注意力权重上进行增强的。
        attn_weights = attn_weights + pos_attn_weights

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(-1, self.h, tgt_len, src_len)
            # 将 attn_weights 中满足 key_padding_mask 的位置（即填充位置）的值替换为负无穷（float('-inf')）。
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [bs, 1, 1, src_len]
                float('-inf')
            )
            attn_weights = attn_weights.view(-1, tgt_len, src_len)
        raw_attn_weights = attn_weights
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = f.dropout(attn_weights, p=self.dropout_p, training=self.training)
        attn_output = torch.bmm(attn_weights, v)
        self.attn = attn_weights

        # 通过调用 .contiguous() 函数，可以将张量转换为连续的内存布局。如果张量已经是连续的，.contiguous() 不会做任何操作，否则会复制数据到一个新的连续张量中。
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bs, -1)
        attn_output = f.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        if self.return_raw_attention:
            return attn_output, raw_attn_weights
        return attn_output, attn_weights

