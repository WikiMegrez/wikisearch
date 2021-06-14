from collections import Counter
from typing import List, Set, Dict, Tuple
import json
from os import walk
import os.path as osp

import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import ray

from wiki_search.dataset.document import Document

stopwords = set(stopwords.words('english'))


def load_doc(json_path: str) -> Document:
    doc = json.load(open(json_path))
    name = doc['entryId']
    title = doc['title']
    main_desc = doc['mainDesc']
    main_image = doc['mainImage']
    out_links = doc['outlinks']
    other_images = doc['otherImages']

    return Document(
        name=name, title=title, raw_main_desc=main_desc,
        main_image=main_image, out_links=out_links, other_images=other_images)


def tokenize_text(text: str) -> List[str]:
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    words = [w for w in words if len(w) > 0]
    words = [w for w in words if w[0].isalpha()]
    words = [w for w in words if w not in stopwords]
    return words


def doc_split_word(doc: Document) -> Document:
    main_desc = doc.raw_main_desc
    words = tokenize_text(main_desc)
    doc.main_desc_words = words
    return doc


@ray.remote
def doc_split_word_runner(doc: Document) -> Document:
    return doc_split_word(doc)


def doc_word_set(doc: Document) -> Set[str]:
    return set(doc.main_desc_words)


@ray.remote
def doc_word_set_runner(doc: Document) -> Set[str]:
    return doc_word_set(doc)


def split_words(data: List[Document]):
    docs = [doc_split_word_runner.remote(doc) for doc in data]
    docs = ray.get(docs)
    return docs


def derive_edges(doc: Document, d2i: Dict[str, int]):
    src = d2i[doc.name]
    dests = [d2i[x] for x in doc.out_links if x in d2i]
    edges = [[src, x] for x in dests]
    edge_index = torch.tensor(edges).to(torch.long)
    edge_index = edge_index.t()
    return edge_index


@ray.remote
def derive_edges_runner(doc: Document, d2i: Dict[str, int]):
    return derive_edges(doc, d2i)


def ray_reduce(func, xs):
    n = len(xs)
    if n == 1:
        return xs[0]
    if n == 2:
        return func.remote(xs[0], xs[1])
    mid = n // 2
    left = ray_reduce(func, xs[:mid])
    right = ray_reduce(func, xs[mid:])
    return func.remote(left, right)


@ray.remote
def merge_set_runner(l: set, r: set) -> set:
    return l.union(r)


def compute_word_set(data: List[Document]):
    sets = [doc_word_set_runner.remote(doc) for doc in data]
    res = ray_reduce(merge_set_runner, sets)
    return ray.get(res)


@ray.remote
def doc_word_count_runner(doc: Document, w2i: Dict[str, int]) -> torch.LongTensor:
    counter = Counter(doc.main_desc_words)
    res = [0 for _ in range(len(w2i))]
    for k, v in counter.items():
        i = w2i[k]
        res[i] = v
    return torch.tensor(res).to(torch.long)


def compute_word_count_mat(data: List[Document], w2i: Dict[str, int]) -> torch.LongTensor:
    xs = [doc_word_count_runner.remote(doc, w2i) for doc in data]
    xs = ray.get(xs)
    return torch.stack(xs, dim=0).to(torch.long)


def compute_idf_mat(word_count: torch.LongTensor) -> torch.FloatTensor:
    mat = word_count.t()  # [W, T]
    mat = (mat > 0).sum(dim=1).to(torch.float32)
    mat = torch.pow(mat, -1)
    mat = word_count.size()[0] * mat
    mat = torch.log(mat).to(torch.float32)

    return mat


def compute_tfidf_mat(tf: torch.LongTensor, idf: torch.FloatTensor) -> torch.FloatTensor:
    # tf: [T, W]
    # idf: [W]
    tf = tf.to(torch.float32)
    idf = torch.unsqueeze(idf, dim=0)
    return tf * idf


class Dataset(object):
    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self.data = None
        self.word_set = None
        self.d2i = None
        self.i2w = None
        self.w2i = None
        self.word_count = None
        self.idf = None
        self.tfidf: torch.FloatTensor = None
        self.edge_index = None

        self.preprocess()

        self.num_docs = self.tfidf.size()[0]
        self.num_words = self.tfidf.size()[1]

    def construct_graph(self):
        edges = [derive_edges_runner.remote(doc, self.d2i) for doc in self.data]
        edges = ray.get(edges)
        edge_index = torch.cat(edges, dim=1)

        return edge_index

    def vectorize(self, text: str) -> torch.FloatTensor:
        words = tokenize_text(text)
        res = torch.zeros((self.num_words,), dtype=torch.float32)
        for w in words:
            if w in self.w2i:
                i = self.w2i[w]
                res[i] += 1.0
        return res.to(torch.float32)

    def vectorize_doc(self, doc: Document) -> torch.FloatTensor:
        return doc.main_desc

    def preprocess(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        print('Preprocessing data ...')
        print('Loading data ...')
        path = osp.join(self.processed_dir, 'raw_data.pt')
        if osp.exists(osp.join(self.processed_dir, 'raw_data.pt')):
            self.data = torch.load(path)
        else:
            self.data = self.load_data()
            torch.save(self.data, path)

        print('Building map from document to index ...')
        self.d2i = dict()
        for i, doc in enumerate(self.data):
            self.d2i[doc.name] = i

        print('Splitting words ...')
        path = osp.join(self.processed_dir, 'splitted_data.pt')
        if osp.exists(path):
            self.data = torch.load(path)
        else:
            self.data = split_words(self.data)
            torch.save(self.data, path)

        print('Computing word dictionary ...')
        path = osp.join(self.processed_dir, 'word_set.pt')
        if osp.exists(path):
            self.word_set = torch.load(path)
        else:
            self.word_set = compute_word_set(self.data)
            torch.save(self.word_set, path)

        print('Building map from index to word ...')
        self.i2w = sorted(list(self.word_set))

        print('Building map from word to index ...')
        self.w2i = dict()
        for i, w in enumerate(self.i2w):
            self.w2i[w] = i

        print('Building word count vector ...')
        path = osp.join(self.processed_dir, 'word_count.pt')
        if osp.exists(path):
            self.word_count = torch.load(path)
        else:
            self.word_count = compute_word_count_mat(self.data, self.w2i)
            torch.save(self.word_count, path)

        print('Building IDF vector ...')
        path = osp.join(self.processed_dir, 'idf.pt')
        if osp.exists(path):
            self.idf = torch.load(path)
        else:
            self.idf = compute_idf_mat(self.word_count)
            torch.save(self.idf, path)

        print('Building TFIDF vector ...')
        path = osp.join(self.processed_dir, 'tfidf.pt')
        if osp.exists(path):
            self.tfidf = torch.load(path)
        else:
            self.tfidf = compute_tfidf_mat(self.word_count, self.idf)
            torch.save(self.tfidf, path)

        for doc in self.data:
            i = self.d2i[doc.name]
            doc.main_desc = self.tfidf[i]

        # print('Building graph edge index ...')
        # path = osp.join(self.processed_dir, 'edge_index.pt')
        # if osp.exists(path):
        #     self.edge_index = torch.load(path)
        # else:
        #     self.edge_index = self.construct_graph()
        #     torch.save(self.edge_index, path)

    @property
    def raw_dir(self):
        return self._data_dir

    @property
    def processed_dir(self):
        return osp.join(self.raw_dir, 'processed')

    def load_data(self):
        res = []
        for root, dirs, files in walk(self.raw_dir):
            for name in files:
                path = osp.join(root, name)
                res.append(load_doc(path))
        res = sorted(res, key=lambda x: x.name)
        return res
