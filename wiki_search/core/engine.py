from collections import namedtuple
from typing import List, Tuple

import torch

from wiki_search.dataset import Dataset, Document
from wiki_search.core.bert_ranking import BertRanking


SearchResult = namedtuple('SearchResult', ['score', 'document'])


class Engine(object):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._bert = BertRanking()

    def search(self, text: str) -> List[SearchResult]:
        query = self._vectorize(text)
        res = self._retrieve(query)
        res = self._sim_rank(query, res)
        return [SearchResult(score=x[0], document=x[1]) for x in res]

    def search_bert(self, text: str) -> List[SearchResult]:
        query = self._vectorize(text)
        res = self._retrieve(query)

        def compute_rank(doc: Document):
            score = self._bert.score(text, doc)
            return SearchResult(score=score, document=doc)

        res = [compute_rank(doc) for doc in res]
        res = sorted(res, key=lambda x: x.score, reverse=True)

        return res

    def _vectorize(self, x):
        if isinstance(x, Document):
            return self._dataset.vectorize_doc(x)
        if isinstance(x, str):
            return self._dataset.vectorize(x)
        raise NotImplementedError(f'Unsupported argument type: {type(x)}')

    def _retrieve(self, query_vector: torch.FloatTensor) -> List[Document]:
        mask = (query_vector > 1e-8).to(torch.bool)
        word_occ = (self._dataset.word_count > 0).to(torch.bool)
        word_occ = word_occ[:, mask]
        match = torch.all(word_occ, dim=1)
        matched = torch.nonzero(match, as_tuple=False).view(-1)
        matched = matched.tolist()

        docs = [self._dataset.data[i] for i in matched]

        return docs

    def _sim_rank(self, query: torch.FloatTensor, candidates: List[Document]) -> List[Tuple[float, Document]]:
        # query: [W]
        vectors = [self._dataset.vectorize_doc(doc) for doc in candidates]
        mat = torch.stack(vectors, dim=0)  # [K, W]
        query = torch.unsqueeze(query, dim=1)
        score = mat @ query  # [K, 1]
        score = torch.squeeze(score, dim=1)
        score = score.tolist()

        res = list(zip(score, candidates))
        res = sorted(res, key=lambda x: x[0], reverse=True)

        return res
