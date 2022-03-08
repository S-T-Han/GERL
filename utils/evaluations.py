import numpy as np


def NDCG(pos_item, rank_list):
    """每个用户的NDCG@k指标，k即rank_list的长度。（对于所有用户使用的是NDCG@k的算数平均值）

    Args:
        pos_item (iterable): 用户阅读过的新闻
        rank_list (iterable): 推荐的新闻，按得分从高到低排列（若ranklist的长度为k，则指标为NDCG@k，例：len(ranklist=5)->NDCG@5）
    """
    pos_item = set(pos_item)
    rank_score = np.array([int(news in pos_item) for news in rank_list])
    dcg = np.sum((2 ** rank_score - 1) / np.log2(np.arange(2, len(rank_score) + 2)))
    idcg = np.sum(np.ones_like(pos_item) / np.log2(np.arange(2, len(rank_score) + 2)))

    return dcg / idcg

def MRR(pos_item, rank_list):
    """每个用户的MRR指标（对于所有用户使用的是MRR的算数平均值）

    Args:
        pos_item (iterable): 用户阅读过的新闻
        rank_list (iterable): 推荐的新闻，按得分从高到低排列
    """
    item_to_rank = {item: (rank + 1) for rank, item in enumerate(rank_list)}
    ranks = np.array([1 / item_to_rank.get(item, np.inf) for item in pos_item])

    return ranks.sum()


if __name__ == '__main__':
    pass