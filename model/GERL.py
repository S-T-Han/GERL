import sys
sys.path.append('/home/sthan/Codefield/python/GERL/')

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from utils.nn import TransFormer, Attention
from model.data import MIND, GolVe
from model.dataloading import DataLoader, NewsSampler, UserSampler, NegativeSampler


class GERL(nn.Module):
    def __init__(
        self, 
        user_id_vocab_size, user_id_embed_dim, user_id_out_feat, 
        title_vocab_size, title_embed_dim, title_num_heads, title_out_feat,
        topic_vocab_size, topic_embed_dim,
        transformer_out_feat,
        news_id_vocab_size, news_id_embed_dim, news_id_out_feat, 
        word_vec=None):
        assert 2 * user_id_embed_dim == (title_embed_dim + topic_embed_dim) + news_id_embed_dim
        super(GERL, self).__init__()
        self.user_id_embed = nn.Embedding(user_id_vocab_size, user_id_embed_dim, padding_idx=0)
        self.user_id_attention = Attention(user_id_embed_dim, user_id_out_feat)
        self.title_embed = nn.Embedding.from_pretrained(word_vec, freeze=True, padding_idx=0) \
            if word_vec is not None else nn.Embedding(title_vocab_size, title_embed_dim, padding_idx=0)
        self.topic_embed = nn.Embedding(topic_vocab_size, topic_embed_dim, padding_idx=0)
        self.transformer = TransFormer(title_embed_dim, title_num_heads, title_out_feat)
        self.transformer_attention = Attention(title_embed_dim + topic_embed_dim, transformer_out_feat)
        self.news_id_embed = nn.Embedding(news_id_vocab_size, news_id_embed_dim, padding_idx=0)
        self.news_id_attention = Attention(news_id_embed_dim, news_id_out_feat)

    def userEncoder(
        self, 
        user_user_id_batch, 
        user_news_title_batch, user_news_topic_batch, 
        neighbor_users_user_id_batch):
        user_user_id_batch = self.user_id_embed(user_user_id_batch)

        user_news_title_batch, user_news_topic_batch = \
            self.title_embed(user_news_title_batch), self.topic_embed(user_news_topic_batch)
        user_news_title_batch, user_news_topic_batch = \
            torch.chunk(user_news_title_batch, user_news_title_batch.shape[1], dim=1), \
            torch.chunk(user_news_topic_batch, user_news_topic_batch.shape[1], dim=1)
        transformer_batch = []
        for i in range(0, len(user_news_title_batch)):
            transformer_batch.append(
                (self.transformer(
                    user_news_title_batch[i].squeeze(dim=1), user_news_topic_batch[i].squeeze(dim=1)))
                .unsqueeze(dim=1))
        transformer_batch = torch.cat(transformer_batch, dim=1)
        transformer_batch = self.transformer_attention(transformer_batch)

        neighbor_users_user_id_batch = self.user_id_embed(neighbor_users_user_id_batch)
        neighbor_users_user_id_batch = self.user_id_attention(neighbor_users_user_id_batch)

        user_code_batch = torch.cat(
            (user_user_id_batch, transformer_batch, neighbor_users_user_id_batch), dim=-1)

        return user_code_batch

    def newsEncoder(
        self, 
        news_title_batch, news_topic_batch, 
        neighbor_news_title_batch, neighbor_news_topic_batch, neighbor_news_news_id_batch):
        news_title_batch, news_topic_batch = \
            self.title_embed(news_title_batch), self.topic_embed(news_topic_batch)
        news_transformer_batch = self.transformer(news_title_batch, news_topic_batch)

        neighbor_news_title_batch, neighbor_news_topic_batch = \
            self.title_embed(neighbor_news_title_batch), self.topic_embed(neighbor_news_topic_batch)
        neighbor_news_title_batch, neighbor_news_topic_batch = \
            torch.chunk(neighbor_news_title_batch, neighbor_news_title_batch.shape[1], dim=1), \
            torch.chunk(neighbor_news_topic_batch, neighbor_news_topic_batch.shape[1], dim=1)
        neighbor_transformer_batch = []
        for i in range(0, len(neighbor_news_title_batch)):
            neighbor_transformer_batch.append(
                (self.transformer(
                    neighbor_news_title_batch[i].squeeze(dim=1), neighbor_news_topic_batch[i].squeeze(dim=1)))
                .unsqueeze(dim=1))
        neighbor_transformer_batch = torch.cat(neighbor_transformer_batch, dim=1)
        neighbor_transformer_batch = self.transformer_attention(neighbor_transformer_batch)

        neighbor_news_news_id_batch = self.news_id_embed(neighbor_news_news_id_batch)
        neighbor_news_news_id_batch = self.news_id_attention(neighbor_news_news_id_batch)

        news_code_batch = torch.cat(
            (news_transformer_batch, neighbor_transformer_batch, neighbor_news_news_id_batch), dim=-1)

        return news_code_batch

    def forward(
        self, 
        user_user_id_batch, 
        user_news_title_batch, user_news_topic_batch, 
        neighbor_users_user_id_batch, 
        news_title_batch, news_topic_batch, 
        neighbor_news_title_batch, neighbor_news_topic_batch, neighbor_news_news_id_batch):
        user_code_batch = self.userEncoder(
            user_user_id_batch, 
            user_news_title_batch, user_news_topic_batch, 
            neighbor_users_user_id_batch)
        news_code_batch = self.newsEncoder(
            news_title_batch, news_topic_batch, 
            neighbor_news_title_batch, neighbor_news_topic_batch, neighbor_news_news_id_batch)
        score_batch = torch.bmm(user_code_batch.unsqueeze(dim=1), news_code_batch.unsqueeze(dim=2))
        score_batch = score_batch.squeeze(dim=1)
        score_batch = torch.sigmoid(score_batch)
        score_batch = torch.clamp(score_batch, 0.0, 1.0)

        return score_batch


class GERLLoss(nn.Module):
    def __init__(self, neg_prop):
        super(GERLLoss, self).__init__()
        self.neg_prop = neg_prop

    def forward(self, score_batch):
        score_batch = score_batch.view(-1, 1 + self.neg_prop)
        loss_batch = score_batch[:, 0] / torch.sum(score_batch, dim=-1)
        loss_batch = torch.log(loss_batch)
        loss = -torch.sum(loss_batch)

        return loss


if __name__ == '__main__':
    mind = MIND()
    g = mind.graphs['train']
    assert isinstance(g, dgl.DGLHeteroGraph)
    user_id_vocab_size, title_vocab_size, topic_vocab_size,news_id_vocab_size = \
        max(list(mind.user_id_vocab.values())), \
        max(list(mind.word_vocab.values()), key=lambda x: x[0])[0], \
        max(list(mind.topic_vocab.values())), \
        max(list(mind.news_id_vocab.values()))

    print('1')
    dataloader = DataLoader(
        g, NegativeSampler(g, 4), UserSampler(user_limit=15, news_limit=10), NewsSampler(news_limit=15), 
        user_news_max_len=10, neighbor_users_max_len=15, neighbor_news_max_len=15)

    print('2')
    user_user_id_batch, \
    (user_news_title_batch, user_news_topic_batch, _), \
    (neighbor_users_user_id_batch, _), \
    (news_title_batch, news_topic_batch), \
    (neighbor_news_title_batch, neighbor_news_topic_batch, neighbor_news_news_id_batch, _), label_batch = next(dataloader.load(batch_size=10))

    print(g.ndata['news_id']['news'].max())
    print(neighbor_news_news_id_batch.max().long())
    
    golve = GolVe()
    model = GERL(
        user_id_vocab_size, 250, 128, 
        title_vocab_size, 300, 6, 128, 
        topic_vocab_size, 100, 128, 
        news_id_vocab_size, 100, 128, 
        word_vec=None)
    result = model(
        user_user_id_batch.long(), 
        user_news_title_batch.long(), user_news_topic_batch.long(), 
        neighbor_users_user_id_batch.long(), 
        news_title_batch, news_topic_batch.long(), 
        neighbor_news_title_batch.long(), neighbor_news_topic_batch.long(), neighbor_news_news_id_batch.long())

    print(result.shape)
    print(result[: 5])

    print(label_batch)
    loss_fn = GERLLoss(neg_prop=4)
    print(loss_fn(label_batch))


