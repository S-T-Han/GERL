from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embed_dim, out_feat):
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(embed_dim, out_feat, bias=True)
        self.linear2 = nn.Linear(out_feat, 1, bias=False)

    def forward(self, x, mask):
        h = self.linear1(x)
        h = torch.tanh(h)
        h = self.linear2(h)
        h = h.squeeze()
        attn = F.softmax(h, dim=-1)
        attn = attn.unsqueeze(dim=-1)
        x = x.sum(dim=-1)

        return x


class TransFormer(nn.Module):
    def __init__(self, num_embed, embed_dim, num_heads, drop_out, batch_first):
        super(TransFormer, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embed, embedding_dim=embed_dim, padding_idx=0)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=drop_out, batch_first=batch_first)


if __name__ == "__main__":
    x = torch.ones(2, 3, 2)
    y = torch.tensor([[[1], [2], [3]], [[4], [5], [6]]])

    attention = Attention(2, 10)
    z = attention(x)
    print(z)
    print('stop here')

    t = torch.tensor(
        [
        [[-0.3228, -0.0914, -0.5141, -0.1389, -0.8249,  0.3055,  0.1391,
        -0.5007,  0.7760, -0.2543],
        [-0.3228, -0.0914, -0.5141, -0.1389, -0.8249,  0.3055,  0.1391,
        -0.5007,  0.7760, -0.2543],
        [-0.3228, -0.0914, -0.5141, -0.1389, -0.8249,  0.3055,  0.1391,
        -0.5007,  0.7760, -0.2543]],

        [[-0.3228, -0.0914, -0.5141, -0.1389, -0.8249,  0.3055,  0.1391,
        -0.5007,  0.7760, -0.2543],
        [-0.3228, -0.0914, -0.5141, -0.1389, -0.8249,  0.3055,  0.1391,
        -0.5007,  0.7760, -0.2543],
        [-0.3228, -0.0914, -0.5141, -0.1389, -0.8249,  0.3055,  0.1391,
        -0.5007,  0.7760, -0.2543]]])
    print(torch.tanh(t))
