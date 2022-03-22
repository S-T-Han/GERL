import sys
sys.path.append('/home/sthan/Codefield/python/GERL')

import torch
import torch.nn as nn
import dgl
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from model.GERL import GERL, GERLLoss
from model.data import MIND, MIND_small, GolVe
from model.dataloading import DataLoader, EvalLoader, UserSampler, NewsSampler, NegativeSampler


def train(model, loss_fn, optimizer, batch_iter, device, total_len=1027941, batch_size=64):
    assert isinstance(model, nn.Module)
    model.train()
    batch_iter = tqdm(batch_iter, total=(total_len // batch_size))

    i = 0
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

        batch_iter.set_postfix(loss = loss.item())

        if i % 2000 == 0:
            torch.save(model.state_dict(), 'saved_models/epoch_{}_batch_{}'.format(epoch, i))
        i += 1


def run(epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mind = MIND()
    golve = GolVe()
    train_graph, dev_graph = mind.graphs['train'], mind.graphs['dev']
    user_id_vocab_size, title_vocab_size, topic_vocab_size,news_id_vocab_size = \
        max(list(mind.user_id_vocab.values())) + 1, \
        max(list(mind.word_vocab.values()), key=lambda x: x[0])[0] + 1, \
        max(list(mind.topic_vocab.values())) + 1, \
        max(list(mind.news_id_vocab.values())) + 1

    train_dataloader = DataLoader(
        g=train_graph, 
        negative_sampler=NegativeSampler(train_graph, neg_prop=4), 
        user_sampler=UserSampler(user_limit=15, news_limit=10), 
        news_sampler=NewsSampler(news_limit=15), 
        user_news_max_len=10, neighbor_users_max_len=15, neighbor_news_max_len=15)
    dev_dataloader = DataLoader(
        g=dev_graph, 
        negative_sampler=NegativeSampler(dev_graph, neg_prop=4), 
        user_sampler=UserSampler(user_limit=15, news_limit=10), 
        news_sampler=NewsSampler(news_limit=15), 
        user_news_max_len=10, neighbor_users_max_len=15, neighbor_news_max_len=15)

    model = GERL(
        user_id_vocab_size=user_id_vocab_size, user_id_embed_dim=200, user_id_out_feat=128, 
        title_vocab_size=title_vocab_size, title_embed_dim=300, title_num_heads=2, title_out_feat=128, 
        topic_vocab_size=topic_vocab_size, topic_embed_dim=10, transformer_out_feat=128, 
        news_id_vocab_size=news_id_vocab_size, news_id_embed_dim=90, news_id_out_feat=128, 
        word_vec=golve.buildEmbedding(mind.word_vocab))
    assert isinstance(model, nn.Module)
    model.to(device)
    loss_fn = nn.BCELoss()
    custom_loss_fn = GERLLoss(neg_prop=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0
    for epoch in range(0, epochs):
        print('epoch: {}'.format(epoch))
        print('Training...')
        train(
            model=model, loss_fn=custom_loss_fn, optimizer=optimizer, 
            batch_iter=train_dataloader.load(64), device=device,
            epoch=epoch)

        torch.save(model.state_dict(), 'saved_models/epoch_{}.pt'.format(epoch))


def run_small(epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mind = MIND_small()
    golve = GolVe()
    train_graph, dev_graph = mind.graphs['train'], mind.graphs['dev']
    user_id_vocab_size, title_vocab_size, topic_vocab_size,news_id_vocab_size = \
        max(list(mind.user_id_vocab.values())) + 1, \
        max(list(mind.word_vocab.values()), key=lambda x: x[0])[0] + 1, \
        max(list(mind.topic_vocab.values())) + 1, \
        max(list(mind.news_id_vocab.values())) + 1

    train_dataloader = DataLoader(
        g=train_graph, 
        negative_sampler=NegativeSampler(train_graph, neg_prop=4), 
        user_sampler=UserSampler(user_limit=15, news_limit=10), 
        news_sampler=NewsSampler(news_limit=15), 
        user_news_max_len=10, neighbor_users_max_len=15, neighbor_news_max_len=15)
    dev_dataloader = DataLoader(
        g=dev_graph, 
        negative_sampler=NegativeSampler(dev_graph, neg_prop=4), 
        user_sampler=UserSampler(user_limit=15, news_limit=10), 
        news_sampler=NewsSampler(news_limit=15), 
        user_news_max_len=10, neighbor_users_max_len=15, neighbor_news_max_len=15)

    model = GERL(
        user_id_vocab_size=user_id_vocab_size, user_id_embed_dim=200, user_id_out_feat=128, 
        title_vocab_size=title_vocab_size, title_embed_dim=300, title_num_heads=2, title_out_feat=128, 
        topic_vocab_size=topic_vocab_size, topic_embed_dim=10, transformer_out_feat=128, 
        news_id_vocab_size=news_id_vocab_size, news_id_embed_dim=90, news_id_out_feat=128, 
        word_vec=golve.buildEmbedding(mind.word_vocab))
    assert isinstance(model, nn.Module)
    model.to(device)
    loss_fn = nn.BCELoss()
    custom_loss_fn = GERLLoss(neg_prop=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0
    for epoch in range(0, epochs):
        print('epoch: {}'.format(epoch))
        print('Training...')
        train(
            model=model, loss_fn=custom_loss_fn, optimizer=optimizer, 
            batch_iter=train_dataloader.load(64), device=device,
            epoch=epoch, total_len=mind.graphs['train'].num_edges())

        torch.save(model.state_dict(), 'saved_models_small/epoch_{}.pt'.format(epoch))


def eval_small(model, model_path, batch_iter, batch_size, num_user, num_news, device):
    assert isinstance(model, nn.Module)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    pred = torch.zeros(num_user, num_news, dtype=torch.long)
    batch_iter = tqdm(batch_iter, total=(num_news * num_user // batch_size))
    for batch in batch_iter:
        user_user_id_batch, \
        (user_news_title_batch, user_news_topic_batch, _), \
        (neighbor_users_user_id_batch, _), \
        (news_title_batch, news_topic_batch), \
        (neighbor_news_title_batch, neighbor_news_topic_batch, neighbor_news_news_id_batch, _), \
        label_batch, \
        (user_nid, start, end), \
        = batch

        score = model(
            user_user_id_batch.to(device), 
            user_news_title_batch.to(device), user_news_topic_batch.to(device), 
            neighbor_users_user_id_batch.to(device), 
            news_title_batch.to(device), news_topic_batch.to(device), 
            neighbor_news_title_batch.to(device), neighbor_news_topic_batch.to(device), neighbor_news_news_id_batch.to(device))
        pred[user_nid][start: end] = score.squeeze(dim=-1)

        print('user_nid: {}, start: {}, end: {}'.format(user_nid, start, end))
        print(score.squeeze(dim=-1))
        print(label_batch)

    print(pred[: 5])
    print(pred.shape)


if __name__ == "__main__":
    # run_small()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mind = MIND_small()
    golve = GolVe()
    train_graph, dev_graph = mind.graphs['train'], mind.graphs['dev']
    user_id_vocab_size, title_vocab_size, topic_vocab_size,news_id_vocab_size = \
        max(list(mind.user_id_vocab.values())) + 1, \
        max(list(mind.word_vocab.values()), key=lambda x: x[0])[0] + 1, \
        max(list(mind.topic_vocab.values())) + 1, \
        max(list(mind.news_id_vocab.values())) + 1
    
    model = GERL(
        user_id_vocab_size=user_id_vocab_size, user_id_embed_dim=200, user_id_out_feat=128, 
        title_vocab_size=title_vocab_size, title_embed_dim=300, title_num_heads=2, title_out_feat=128, 
        topic_vocab_size=topic_vocab_size, topic_embed_dim=10, transformer_out_feat=128, 
        news_id_vocab_size=news_id_vocab_size, news_id_embed_dim=90, news_id_out_feat=128, 
        word_vec=golve.buildEmbedding(mind.word_vocab))
    dev_dataloader = EvalLoader(
        g=dev_graph, 
        user_sampler=UserSampler(user_limit=15, news_limit=10), 
        news_sampler=NewsSampler(news_limit=15), 
        user_news_max_len=10, neighbor_users_max_len=15, neighbor_news_max_len=15)

    eval_small(
        model, 'saved_models_small/epoch_2.pt', 
        dev_dataloader.load(256), 256, 
        dev_graph.num_nodes('user'), dev_graph.num_nodes('news'), device)



