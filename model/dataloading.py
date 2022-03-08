import torch
import dgl
import numpy as np
import scipy.sparse as sp
from data import MIND


class DataLoader():
    def __init__(self, g, negative_sampler, user_sampler, news_sampler):
        assert isinstance(g, dgl.DGLHeteroGraph)
        self.graph = g
        self.negative_sampler = negative_sampler
        self.user_sampler, self.news_sampler = user_sampler, news_sampler
        self.pos_users, self.pos_news = g.edges()
        self.num_edges = g.num_edges()
        self.eid = np.arange(self.num_edges)
        
    def load(self, batch_size):
        eid = np.random.permutation(self.eid)
        I = np.arange(0, self.num_edges, batch_size)
        for start, end in zip(I[: -2], I[1: ]):
            eid_batch = eid[start: end]
            pos_users_nid_batch, pos_news_nid_batch = self.graph.find_edges(eid_batch, etype='clicked')
            neg_users_nid_batch, neg_news_nid_batch = self.negative_sampler(self.graph, eid_batch)
            users_nid_batch, news_nid_batch = \
                torch.cat((pos_users_nid_batch, neg_users_nid_batch)), torch.cat((pos_news_nid_batch, neg_news_nid_batch))
            labels_batch = torch.cat((torch.ones_like(pos_users_nid_batch), torch.zeros_like(neg_users_nid_batch)))
            shuffled_idx = np.random.permutation(np.arange(0, len(users_nid_batch)))
            users_nid_batch, labels_batch = users_nid_batch[shuffled_idx], labels_batch[shuffled_idx]
            
            # users_neighbor, news_neighbor_news = [], []
            for user_nid, news_nid in zip(users_nid_batch, news_nid_batch):
                # users_neighbor.append(self.user_sampler(g, user_nid))
                # news_neighbor_news.append(self.news_sampler(g, news_nid))
                user_news_nid, user_neighbor_users_nid = self.user_sampler(g, user_nid)
                users_news_title = g.ndata['title']['news'][user_news_nid]
                users_neighbor_users_id = g.ndata['user_id']['user'][user_neighbor_users_nid]
                news_neighbor_news_nid = self.news_sampler(g, news_nid)
                news_neighbor_news_id = g.ndata['news_id']['news'][news_neighbor_news_nid]
                news_neighbor_news_title = g.ndata['title']['news'][news_neighbor_news_nid]
                
                print('one trun start')
                print(user_nid, news_nid)
                print(user_news_nid, user_neighbor_users_nid)
                print(users_news_title.shape)
                print(users_neighbor_users_id.shape)
                print(news_neighbor_news_nid)
                print(news_neighbor_news_id.shape)
                print(news_neighbor_news_title.shape)
                print('one turn over')
                print()


            # yield users_nid_batch, users_neighbor, news_nid_batch, news_neighbor_news, labels_batch
            yield 1


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
    def __init__(self, user_limit):
        self.user_limit = user_limit

    def __call__(self, g, user_nid):
        assert isinstance(g, dgl.DGLHeteroGraph)
        g1 = dgl.sampling.sample_neighbors(
            g, {'user': user_nid}, {'clicked': -1}, edge_dir='out')
        _, news_nid = g1.edges(order='eid')
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
    dataloader = DataLoader(g, NegativeSampler(g, 1), UserSampler(user_limit=15), NewsSampler(news_limit=15))
    a = next(dataloader.load(2))
    