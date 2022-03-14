import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embed_dim, out_feat):
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(embed_dim, out_feat)
        self.linear2 = nn.Linear(out_feat, 1, bias=False)

    def forward(self, x, mask=None):
        h = self.linear1(x)
        h = torch.tanh(h)
        h = self.linear2(h)
        h = h.squeeze()
        attn = F.softmax(h, dim=-1)
        attn = attn.unsqueeze(dim=-1)
        x = torch.mul(attn, x)
        x = torch.sum(x, -1)

        return x


class TransFormer(nn.Module):
    def __init__(self, embed_dim, num_heads, out_feat):
        super(TransFormer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=0.3, batch_first=True)
        self.attention = Attention(embed_dim, out_feat)

    def forward(self, x, y, mask=None):
        x, _ = self.multihead_attention(query=x, key=x, value=x)
        x = self.attention(x)

        return torch.cat((x, y), dim=-1)


if __name__ == "__main__":
    x = torch.ones(2, 3, 2)
    y = torch.ones(2, 3)


    transformer = TransFormer(embed_dim=2, num_heads=1, out_feat=5)
    result = transformer(x, y)

    print(result.shape)
    print(result)
