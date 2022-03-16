import sys
sys.path.append('/home/sthan/Codefield/python/GERL')

import torch
import dgl
import numpy as np
from model.data import MIND

import time


class DataLoader():
    def __init__(
        self, 
        g, negative_sampler, user_sampler, news_sampler, 
        user_news_max_len, neighbor_users_max_len, neighbor_news_max_len):
        assert isinstance(g, dgl.DGLHeteroGraph)

        self.g = g
        self.negative_sampler = negative_sampler
        self.user_sampler, self.news_sampler = user_sampler, news_sampler
        self.pos_users, self.pos_news = g.edges()
        self.user_news_max_len, self.neighbor_users_max_len, self.neighbor_news_max_len = \
                user_news_max_len, neighbor_users_max_len, neighbor_news_max_len

        self.num_edges = g.num_edges()
        self.eid = np.arange(self.num_edges)
        
    def load(self, batch_size):
        eid = np.random.permutation(self.eid)
        I = np.arange(0, self.num_edges, batch_size)
        for start, end in zip(I[: -2], I[1: ]):
            eid_batch = eid[start: end]
            pos_user_nid_batch, pos_news_nid_batch = self.g.find_edges(eid_batch, etype='clicked')
            neg_user_nid_batch, neg_news_nid_batch = self.negative_sampler(self.g, eid_batch)
            user_nid_batch, news_nid_batch = \
                torch.cat((pos_user_nid_batch, neg_user_nid_batch)), torch.cat((pos_news_nid_batch, neg_news_nid_batch))
            label_batch = torch.cat((torch.ones_like(pos_user_nid_batch), torch.zeros_like(neg_user_nid_batch)))
            total_batch_size = len(user_nid_batch)
            pos_size, neg_prop = len(pos_news_nid_batch), len(neg_news_nid_batch) // len(pos_news_nid_batch)
            reordered_idx = np.array([
                i // (neg_prop + 1) if i % (neg_prop + 1) == 0 else (i + pos_size - (i // (neg_prop + 1)) - 1) 
                for i in range(total_batch_size)])
            user_nid_batch, news_nid_batch, label_batch = user_nid_batch[reordered_idx], news_nid_batch[reordered_idx], label_batch[reordered_idx]
           
            user_user_id_batch, news_title_batch, news_topic_batch = \
                self.g.ndata['user_id']['user'][user_nid_batch], self.g.ndata['title']['news'][news_nid_batch], \
                self.g.ndata['topic']['news'][news_nid_batch]
            user_news_title_batch, user_news_topic_batch, neighbor_users_user_id_batch = \
                torch.zeros((total_batch_size, self.user_news_max_len, 30)), torch.zeros((total_batch_size, self.user_news_max_len)), \
                torch.zeros((total_batch_size, self.neighbor_users_max_len))
            user_news_effective_len, neighbor_users_effective_len = \
                torch.zeros(total_batch_size), torch.zeros(total_batch_size)
            neighbor_news_title_batch, neighbor_news_topic_batch, neighbor_news_news_id_batch = \
                torch.zeros((total_batch_size, self.neighbor_news_max_len, 30)), torch.zeros((total_batch_size, self.neighbor_news_max_len)), \
                torch.zeros((total_batch_size, self.neighbor_news_max_len))
            neighbor_news_effective_len = torch.zeros(total_batch_size)

            for i, (user_nid, news_nid) in enumerate((zip(user_nid_batch, news_nid_batch))):
                user_news_nid, neighbor_users_nid = self.user_sampler(self.g, user_nid)
                user_news_effective_len[i] = len(user_news_nid)
                user_news_title_batch[i][0: len(user_news_nid)] = self.g.ndata['title']['news'][user_news_nid]
                user_news_topic_batch[i][0: len(user_news_nid)] = self.g.ndata['topic']['news'][user_news_nid]
                neighbor_users_effective_len[i] = len(neighbor_users_nid)
                neighbor_users_user_id_batch[i][0: len(neighbor_users_nid)] = self.g.ndata['user_id']['user'][neighbor_users_nid]

                neighbor_news_nid = self.news_sampler(self.g, news_nid)
                neighbor_news_effective_len[i] = len(neighbor_news_nid)
                neighbor_news_title_batch[i][0: len(neighbor_news_nid)] = self.g.ndata['title']['news'][neighbor_news_nid]
                neighbor_news_topic_batch[i][0: len(neighbor_news_nid)] = self.g.ndata['topic']['news'][neighbor_news_nid]
                neighbor_news_news_id_batch[i][0: len(neighbor_news_nid)] = self.g.ndata['news_id']['news'][neighbor_news_nid]
                
            yield user_user_id_batch, \
                (user_news_title_batch, user_news_topic_batch, user_news_effective_len), (neighbor_users_user_id_batch, neighbor_users_effective_len), \
                (news_title_batch, news_topic_batch), \
                (neighbor_news_title_batch, neighbor_news_topic_batch, neighbor_news_news_id_batch, neighbor_news_effective_len), \
                label_batch


class NegativeSampler():
    def __init__(self, g, neg_prop):
        assert isinstance(g, dgl.DGLHeteroGraph)
        self.weights = {'clicked': torch.ones(g.num_nodes(ntype='news'))}
        self.neg_prop = neg_prop

    def __call__(self, g, eids):
        assert isinstance(g, dgl.DGLHeteroGraph)
        users_nid, _ = g.find_edges(eids, etype='clicked')
        users_nid_total = users_nid.repeat_interleave(self.neg_prop)
        neg_news_nid_total = []
        for user_nid in users_nid:
            _, pos_news_nid = g.out_edges(user_nid)
            weights = self.weights['clicked']
            weights[pos_news_nid] = 0
            neg_news_nid = weights.multinomial(self.neg_prop, replacement=True)
            neg_news_nid_total.append(neg_news_nid)
        neg_news_nid_total = torch.cat(neg_news_nid_total)

        return users_nid_total, neg_news_nid_total


class UserSampler():
    def __init__(self, user_limit, news_limit):
        self.user_limit, self.news_limit = user_limit, news_limit

    def __call__(self, g, user_nid):
        assert isinstance(g, dgl.DGLHeteroGraph)
        g1 = dgl.sampling.sample_neighbors(
            g, {'user': user_nid}, {'clicked': -1}, edge_dir='out')
        _, news_nid = g1.edges(order='eid')
        
        shuffled_idx = np.random.permutation(np.arange(0, len(news_nid)))
        news_nid = news_nid[shuffled_idx]
        news_nid = news_nid[: min(self.news_limit, len(news_nid))]

        g2 = dgl.sampling.sample_neighbors(
            g, {'news': news_nid}, {'clicked': -1}, edge_dir='in')
        neighbor_users_nid, _ = g2.edges(order='eid')

        neighbor_users_nid = neighbor_users_nid.unique()
        shuffled_idx = np.random.permutation(np.arange(0, len(neighbor_users_nid)))
        neighbor_users_nid = neighbor_users_nid[shuffled_idx]
        neighbor_users_nid = neighbor_users_nid[: min(self.user_limit, len(neighbor_users_nid))]

        return news_nid, neighbor_users_nid


class NewsSampler():
    def __init__(self, news_limit):
        self.news_limit = news_limit

    def __call__(self, g, news_nid):
        assert isinstance(g, dgl.DGLHeteroGraph)
        g1 = dgl.sampling.sample_neighbors(
            g, {'news': news_nid}, {'clicked': -1}, edge_dir='in')
        users_nid, _ = g1.edges(order='eid')
        g2 = dgl.sampling.sample_neighbors(
            g, {'user': users_nid}, {'clicked': -1}, edge_dir='out')
        _, neighbor_news_nid = g2.edges(order='eid')

        neighbor_news_nid = neighbor_news_nid.unique()
        shuffled_idx = np.random.permutation(np.arange(0, len(neighbor_news_nid)))
        neighbor_news_nid = neighbor_news_nid[shuffled_idx]
        neighbor_news_nid = neighbor_news_nid[: min(self.news_limit, len(neighbor_news_nid))]

        return neighbor_news_nid


if __name__ == '__main__':
    mind = MIND()
    g = mind.graphs['train']
    dataloader = DataLoader(
        g, NegativeSampler(g, 4), UserSampler(user_limit=15, news_limit=10), NewsSampler(news_limit=15), 
        user_news_max_len=10, neighbor_users_max_len=15, neighbor_news_max_len=15)
    t1 = time.time()
    i = 0
    for everything in dataloader.load(batch_size=5):
        for one_thing in everything:
            print(one_thing)
        break
    t2 = time.time()
    print(i)
    print("{}ms".format((t2 -t1) * 1000))
    
