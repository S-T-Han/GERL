import sys
sys.path.append('/home/sthan/Codefield/python/GERL')

import torch
import torch.nn as nn
import dgl
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from model.GERL import GERL, GERLLoss
from model.data import MIND, GolVe
from model.dataloading import DataLoader, UserSampler, NewsSampler, NegativeSampler


def train(model, loss_fn, optimizer, batch_iter, device):
    assert isinstance(model, nn.Module)
    model.train()
    batch_iter = tqdm(batch_iter, total=(1027941 // 50))

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


def eval(model, batch_iter, device, best_acc=0):
    assert isinstance(model, nn.Module)
    model.eval()

    total_acc = []
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
        y_pred = (score > 0.5).int().cpu().numpy()
        y_true = label_batch.int().cpu().numpy()
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        total_acc.append(acc)

    avg_acc = np.array(total_acc).mean()
    if avg_acc > best_acc:
        torch.save(model.state_dict(), 'saved_models/acc_{}.pt'.format(best_acc))

    return best_acc


if __name__ == "__main__":
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
    for epoch in range(0, 10):
        print('epoch: {}'.format(epoch))
        print('Training...')
        train(
            model=model, loss_fn=custom_loss_fn, optimizer=optimizer, 
            batch_iter=train_dataloader.load(50), device=device)
