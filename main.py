import sys
sys.path.append('/home/sthan/Codefield/python/GERL')

import torch
import torch.nn as nn
import dgl
from tqdm import tqdm

from model.GERL import GERL, GERLLoss
from model.data import MIND, GolVe
from model.dataloading import DataLoader, UserSampler, NewsSampler, NegativeSampler


def train(model, loss_fn, optimizer, batch_iter, device):
    assert isinstance(model, nn.Module)
    model.train()
    batch_iter = tqdm(batch_iter, total=20559)

    for batch in batch_iter:
        user_user_id_batch, \
        (user_news_title_batch, user_news_topic_batch, _), \
        (neighbor_users_user_id_batch, _), \
        (news_title_batch, news_topic_batch), \
        (neighbor_news_title_batch, neighbor_news_topic_batch, neighbor_news_news_id_batch, _), \
        label_batch \
        = batch
        score = model(
            user_user_id_batch.long().to(device), 
            user_news_title_batch.long().to(device), user_news_topic_batch.long().to(device), 
            neighbor_users_user_id_batch.long().to(device),
            news_title_batch.long().to(device), news_topic_batch.long().to(device), 
            neighbor_news_title_batch.long().to(device), neighbor_news_topic_batch.long().to(device), neighbor_news_news_id_batch.long().to(device))
        loss = loss_fn(score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_iter.set_postfix(loss=loss.item())


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mind = MIND()
    golve = GolVe()
    train_graph, dev_graph = mind.graphs['train'], mind.graphs['dev']
    user_id_vocab_size, title_vocab_size, topic_vocab_size,news_id_vocab_size = \
        max(list(mind.user_id_vocab.values())), \
        max(list(mind.word_vocab.values()), key=lambda x: x[0])[0], \
        max(list(mind.topic_vocab.values())), \
        max(list(mind.news_id_vocab.values()))

    train_dataloader = DataLoader(
        g=train_graph, 
        negative_sampler=NegativeSampler(train_graph, neg_prop=4), 
        user_sampler=UserSampler(user_limit=15, news_limit=10), 
        news_sampler=NewsSampler(news_limit=15), 
        user_news_max_len=10, neighbor_users_max_len=15, neighbor_news_max_len=15)
    dev_dataloader = DataLoader(
        g=train_graph, 
        negative_sampler=NegativeSampler(dev_graph, neg_prop=4), 
        user_sampler=UserSampler(user_limit=15, news_limit=10), 
        news_sampler=NewsSampler(news_limit=15), 
        user_news_max_len=10, neighbor_users_max_len=15, neighbor_news_max_len=15)

    model = GERL(
        user_id_vocab_size=user_id_vocab_size, user_id_embed_dim=250, user_id_out_feat=128, 
        title_vocab_size=title_vocab_size, title_embed_dim=300, title_num_heads=4, title_out_feat=128, 
        topic_vocab_size=topic_vocab_size, topic_embed_dim=100, transformer_out_feat=128, 
        news_id_vocab_size=news_id_vocab_size, news_id_embed_dim=100, news_id_out_feat=128, 
        word_vec=golve.buildEmbedding(mind.word_vocab))
    assert isinstance(model, nn.Module)
    model.to(device)
    loss_fn = GERLLoss(neg_prop=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(
        model=model, loss_fn=loss_fn, optimizer=optimizer, 
        batch_iter=train_dataloader.load(batch_size=50),
        device=device)
    
