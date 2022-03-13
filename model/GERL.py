import sys
import os
sys.path.append('/home/sthan/Codefield/python/GERL/')

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from utils.nn import TransFormer, Attention
from data import MIND, GolVe
from dataloading import DataLoader, NewsSampler, UserSampler, NegativeSampler


class GERL(nn.Module):
    def __init__(
        self, 
        user_id_vocab_size, user_id_embed_dim, user_id_out_feat, 
        title_vocab_size, title_embed_dim, title_num_heads, title_out_feat,
        topic_vocab_size, topic_embed_dim,
        transformer_out_feat,
        news_id_vocab_size, news_id_embed_dim, news_id_out_feat, 
        word_vec):
        super(GERL, self).__init__()
        self.user_id_embed = nn.Embedding(user_id_vocab_size, user_id_embed_dim, padding_idx=0)
        self.user_id_attention = Attention(user_id_embed_dim, user_id_out_feat)
        self.title_embed = nn.Embedding.from_pretrained(word_vec, freeze=True, padding_idx=0)
        self.topic_embed = nn.Embedding(topic_vocab_size, topic_embed_dim, padding_idx=0)
        self.transformer = TransFormer(title_embed_dim, title_num_heads, title_out_feat)
        self.transformer_attention = Attention(title_embed_dim + topic_embed_dim, transformer_out_feat)
        self.news_id_embed = nn.Embedding(news_id_vocab_size, news_id_embed_dim, padding_idx=0)
        self.news_id_attention = Attention(news_id_embed_dim, news_id_out_feat)

        print(self.title_embed.weight)

    def userEncoder(
        self, 
        user_user_id_batch, 
        user_news_title_batch, user_news_topic_batch, 
        neighbor_users_user_id_batch):
        user_user_id_batch = self.user_id_embed(user_user_id_batch)

        user_news_title_batch, user_news_topic_batch = \
            self.title_embed(user_news_title_batch), self.topic_embed(user_news_topic_batch)
        transformer_batch = torch.zeros(
            user_news_title_batch.shape[0], user_news_title_batch.shape[1], 
            user_news_title_batch.shape[-1] + user_news_topic_batch.shape[-1])
        for i in range(0, user_news_title_batch.shape[1]):
            transformer_batch[:, i] = self.transformer(
                user_news_title_batch[:, i], user_news_topic_batch[:, i])
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
        neighbor_transformer_batch = torch.zeros(
            neighbor_news_title_batch.shape[0], neighbor_news_title_batch.shape[1], 
            neighbor_news_title_batch.shape[-1] + neighbor_news_topic_batch.shape[-1])
        for i in range(0, neighbor_news_title_batch.shape[1]):
            neighbor_transformer_batch[:, i] = self.transformer(
                neighbor_news_title_batch[:, i], neighbor_news_topic_batch[:, i])
        neighbor_transformer_batch = self.transformer_attention(neighbor_transformer_batch)

        neighbor_news_news_id_batch = self.news_id_embed(neighbor_news_news_id_batch)
        neighbor_news_news_id_batch = self.news_id_attention(neighbor_news_news_id_batch)

        news_code_batch = torch.cat(
            (news_transformer_batch, neighbor_transformer_batch, neighbor_news_news_id_batch), dim=-1)

        return news_code_batch


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
    (neighbor_news_title_batch, neighbor_news_topic_batch, neighbor_news_news_id_batch, _), _ = next(dataloader.load(batch_size=10))
    
    golve = GolVe()
    model = GERL(
        user_id_vocab_size, 100, 128, 
        title_vocab_size, 300, 4, 256, 
        topic_vocab_size, 100, 
        256, 
        news_id_vocab_size, 100, 100,
        word_vec=golve.buildEmbedding(mind.word_vocab))


    result = model.userEncoder(
        user_user_id_batch.long(), 
        user_news_title_batch.long(), user_news_topic_batch.long(), 
        neighbor_users_user_id_batch.long())
    print(result.shape)
    result = model.newsEncoder(
        news_title_batch.long(), news_topic_batch.long(), 
        neighbor_news_title_batch.long(), neighbor_news_topic_batch.long(), neighbor_news_news_id_batch.long())
    print(result.shape)


